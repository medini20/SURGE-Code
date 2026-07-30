"""Neural Improvement Heuristic for Vertex Colouring (min-conflict k-colouring).

The GNN *is* the search algorithm. Starting from an initial colouring, an RL
agent iteratively rewrites it: at each step it picks a (vertex, colour) move,
learning to escape local optima better than classical local search (TabuCol).

Key properties
--------------
* Fully **batched & vectorised** environment (runs B graphs in parallel on GPU).
* **Size-generalisable** GNN: message passing + a per-node colour head whose
  weights are shared across vertices, so one trained model runs on any N.
* Custom **masked PPO** trainer (only conflicted vertices are recolourable).

This module is standalone (no stable-baselines3), just torch + networkx.
"""
import argparse
import os
import time
import math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Graph generation
# ======================================================================
def plant_clique_np(adj, k, rng, w_lo=1.0, w_hi=1.0):
    """Plants a (k+1)-clique into `adj` in place. By pigeonhole, a clique
    that size needs k+1 colours, so chi(G) > k comes for free without
    ever solving for the chromatic number."""
    n = adj.shape[0]
    assert n >= k + 1, f"need at least k+1={k+1} nodes to plant a (k+1)-clique, got n={n}"
    nodes = rng.choice(n, size=k + 1, replace=False)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = int(nodes[i]), int(nodes[j])
            w = 1.0 if (w_lo == w_hi == 1.0) else float(rng.uniform(w_lo, w_hi))
            adj[u, v] = w
            adj[v, u] = w
    return adj


def sample_graphs(batch, n, p_lo, p_hi, device, rng, w_lo=1.0, w_hi=1.0, force_k=None):
    """Returns a batch of dense Erdos-Renyi adjacency matrices, (B, N, N).
    Edge weights are i.i.d. Uniform(w_lo, w_hi); leave the defaults for
    plain unweighted graphs. Pass force_k to plant a (force_k+1)-clique
    into every graph, guaranteeing chi(G) > force_k."""
    A = torch.zeros(batch, n, n, device=device)
    for b in range(batch):
        p = float(rng.uniform(p_lo, p_hi))
        G = nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(0, 2**31 - 1)))
        if G.number_of_edges() > 0:
            e = np.array(G.edges(), dtype=np.int64).T
            idx = torch.from_numpy(e).to(device)
            w = 1.0 if (w_lo == w_hi == 1.0) else torch.from_numpy(
                rng.uniform(w_lo, w_hi, size=e.shape[1]).astype(np.float32)).to(device)
            A[b, idx[0], idx[1]] = w
            A[b, idx[1], idx[0]] = w
        if force_k is not None:
            adj_np = A[b].cpu().numpy()
            plant_clique_np(adj_np, force_k, rng, w_lo, w_hi)
            A[b] = torch.from_numpy(adj_np).to(device)
    return A


def gen_graphs(n_graphs, n, lo, hi, seed, w_lo=1.0, w_hi=1.0, force_k=None):
    """Same as `sample_graphs` but returns a plain list of numpy arrays,
    for the numpy/Gurobi baselines."""
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n_graphs):
        p = float(rng.uniform(lo, hi))
        G = nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(0, 2**31 - 1)))
        adj = nx.to_numpy_array(G, dtype=np.float32)
        if not (w_lo == w_hi == 1.0) and G.number_of_edges() > 0:
            mask = adj > 0
            w = rng.uniform(w_lo, w_hi, size=adj.shape).astype(np.float32)
            w = np.triu(w, k=1); w = w + w.T   # symmetric, zero diagonal
            adj = np.where(mask, w, 0.0).astype(np.float32)
        if force_k is not None:
            plant_clique_np(adj, force_k, rng, w_lo, w_hi)
        graphs.append(adj)
    return graphs


class FixedGraphPool:
    """Serves shuffled batches from a pre-generated pool of graphs with
    variable node counts, instead of sampling a fresh random graph every
    iteration. Graphs are zero-padded to a common max size, and each
    graph's true node count is stored separately in `nodes`. Padding
    nodes have no edges, so they can never be in conflict and never get
    picked (see `ColoringEnv.action_mask()`). `next_batch` also hands back
    the true node counts so the policy can pool/condition on size
    correctly (see `GNNPolicy`).

    Also keeps a real epoch count: one full shuffled pass through the pool.
    """

    def __init__(self, adj, nodes):
        # `adj` is usually a memmap (mmap_mode="r"), so only the slice we
        # pick in next_batch() actually gets pulled into RAM.
        self.adj = adj
        self.nodes = np.asarray(nodes)
        self.rng = np.random.default_rng(0)
        self.perm = self.rng.permutation(len(self.adj))
        self.cursor = 0
        self.epochs = 0

    def next_batch(self, batch, device):
        if batch > len(self.adj):
            idx = self.rng.integers(0, len(self.adj), size=batch)  # sample with replacement
        else:
            if self.cursor + batch > len(self.perm):
                self.epochs += 1
                self.perm = self.rng.permutation(len(self.adj))
                self.cursor = 0
            idx = self.perm[self.cursor:self.cursor + batch]
            self.cursor += batch
        adj_np = np.array(self.adj[idx], dtype=np.float32)
        nodes_np = self.nodes[idx].astype(np.int64)
        return (torch.from_numpy(adj_np).to(device),
                torch.from_numpy(nodes_np).to(device))


def load_fixed_dataset(path, device):
    """Loads a dataset directory produced by `generate_overconstrained_dataset.py`.
    Returns (train_pool, val_pool, n_lo, n_hi, k), where k is the colour
    budget the dataset guarantees chi(G) > k for. Adjacency arrays stay
    memmapped rather than being read fully into RAM."""
    meta = np.load(os.path.join(path, "meta.npz"))
    n_lo, n_hi, k = int(meta["n_lo"]), int(meta["n_hi"]), int(meta["k"])
    train_adj = np.load(os.path.join(path, "train_adj.npy"), mmap_mode="r")
    train_nodes = np.load(os.path.join(path, "train_nodes.npy"))
    val_adj = np.load(os.path.join(path, "val_adj.npy"), mmap_mode="r")
    val_nodes = np.load(os.path.join(path, "val_nodes.npy"))
    train_pool = FixedGraphPool(train_adj, train_nodes)
    val_pool = FixedGraphPool(val_adj, val_nodes)
    return train_pool, val_pool, n_lo, n_hi, k


# ======================================================================
# Vectorised colouring dynamics
# ======================================================================
def neighbour_colour_counts(A, x, k):
    """(B,N,k): for each node, how many neighbours currently use each colour."""
    onehot = F.one_hot(x, num_classes=k).float()          # (B,N,k)
    return torch.bmm(A, onehot)                            # (B,N,k)


def total_conflicts(A, x, k):
    """(B,): number of monochromatic edges per graph."""
    nc = neighbour_colour_counts(A, x, k)
    own = nc.gather(-1, x.unsqueeze(-1)).squeeze(-1)       # (B,N) conflicts per node
    return own.sum(dim=1) / 2.0


RECENCY_CAP = 20.0  # just keeps this feature on a consistent scale regardless of N or horizon


class ColoringEnv:
    """Batched iterative-improvement MDP for k-colouring.

    State  : adjacency A (fixed per episode) + current colouring x (B,N) +
             recency (B,N) = steps since each vertex was last recoloured.
    Action : flat index a in [0, N*k) -> (node=a//k, colour=a%k).
    Reward : conflicts_before - conflicts_after.
    Mask   : only conflicted vertices may be recoloured, never to their
             current colour. 0 conflicts = done.

    `node_counts`, if given, holds each graph's true node count for a
    zero-padded, variable-size batch (see `FixedGraphPool`). Just leave
    it out for same-size batches.
    """

    def __init__(self, A, k, horizon, node_counts=None):
        self.A = A
        self.B, self.N = A.shape[0], A.shape[1]
        self.k = k
        self.horizon = horizon
        self.deg = A.sum(-1)                               # (B,N)
        self.node_counts = (node_counts if node_counts is not None
                             else torch.full((self.B,), self.N, device=A.device, dtype=torch.long))

    def reset(self, x0):
        self.x = x0.clone()
        self.t = 0
        self.recency = torch.zeros(self.B, self.N, device=self.A.device)
        self.conf = total_conflicts(self.A, self.x, self.k)   # (B,)
        return self.x

    def action_mask(self):
        """(B, N*k) boolean mask of legal moves."""
        nc = neighbour_colour_counts(self.A, self.x, self.k)   # (B,N,k)
        own = nc.gather(-1, self.x.unsqueeze(-1)).squeeze(-1)  # (B,N)
        conflicted = own > 0                                   # (B,N)
        mask = conflicted.unsqueeze(-1).expand(-1, -1, self.k).clone()  # (B,N,k)
        mask.scatter_(2, self.x.unsqueeze(-1), False)          # no-op recolours aren't legal moves
        return mask.reshape(self.B, self.N * self.k)

    def step(self, action, active):
        """Apply (node,colour); return reward, done. `active` masks finished graphs."""
        node = torch.div(action, self.k, rounding_mode="floor")   # (B,)
        colour = action % self.k                                  # (B,)
        conf_before = self.conf
        new_x = self.x.clone()
        idx = torch.arange(self.B, device=self.A.device)
        applied = active
        new_x[idx[applied], node[applied]] = colour[applied]
        self.x = new_x
        self.recency = self.recency + 1
        self.recency[idx[applied], node[applied]] = 0.0
        self.conf = total_conflicts(self.A, self.x, self.k)
        reward = (conf_before - self.conf) * active.float()
        self.t += 1
        solved = self.conf <= 0
        done = solved | (self.t >= self.horizon)
        return reward, done, solved


def coloured_neighbour_weights(A, x, coloured, k):
    """(B,N,k): for each vertex, the weighted conflict cost of each
    candidate colour c, counting only neighbours that are already
    coloured (uncoloured ones don't contribute anything). Used by both
    `ColoringEnvConstruct.step` and `GNNPolicyConstruct.features`."""
    oh = F.one_hot(x, num_classes=k).float() * coloured.unsqueeze(-1).float()
    return torch.bmm(A, oh)                                     # (B,N,k)


# ======================================================================
# Constructive MDP: colour the graph one vertex at a time, never coming
# back to fix one later. The environment picks the actual colour, not the
# policy -- it tries the first colour that doesn't clash with an
# already-coloured neighbour, and if nothing's free, falls back to
# whichever colour does the least damage. All the policy has to learn is
# which vertex to colour next. Basically DSATUR, but with a learned
# ordering instead of the fixed saturation-degree rule. Since every real
# vertex gets coloured exactly once, an episode just naturally ends after
# `node_count` steps -- no horizon needed.
# ======================================================================
class ColoringEnvConstruct:
    """Batched constructive MDP for k-colouring.

    State  : adjacency A + partial colouring x (B,N, meaningless where not
             yet coloured) + coloured (B,N) bool.
    Action : index of one not-yet-coloured (real) vertex to colour next.
    Reward : set by the environment's fixed colouring rule (see module
             comment above), not chosen by the policy.
    Mask   : `action_mask()` -- real, not-yet-coloured vertices.

    `node_counts`, if given, holds true node counts for a zero-padded,
    variable-size batch. Leave it out for same-size batches.
    """

    def __init__(self, A, k, node_counts=None, reward_weight=1.0):
        self.A = A
        self.B, self.N = A.shape[0], A.shape[1]
        self.k = k
        self.reward_weight = reward_weight
        self.node_counts = (node_counts if node_counts is not None
                             else torch.full((self.B,), self.N, device=A.device, dtype=torch.long))
        self._real = (torch.arange(self.N, device=A.device).view(1, self.N)
                      < self.node_counts.view(self.B, 1))         # (B,N) bool

    def reset(self):
        self.x = torch.zeros(self.B, self.N, dtype=torch.long, device=self.A.device)
        self.coloured = torch.zeros(self.B, self.N, dtype=torch.bool, device=self.A.device)
        self.t = 0
        return self.x

    def action_mask(self):
        """(B,N) boolean: real, not-yet-coloured vertices."""
        return self._real & ~self.coloured

    def step(self, v_idx, active):
        """Colour vertex v_idx (B,) -- for graphs where active is True --
        via the fixed rule. Returns (reward (B,), done (B,))."""
        cw = coloured_neighbour_weights(self.A, self.x, self.coloured, self.k)  # (B,N,k)
        idx = torch.arange(self.B, device=self.A.device)
        cw_v = cw[idx, v_idx]                                     # (B,k) conflict weight per candidate colour
        available = cw_v <= 1e-9
        has_available = available.any(dim=1)                      # (B,)
        colour_idx = torch.arange(self.k, device=self.A.device).view(1, self.k)
        first_avail = torch.where(available, colour_idx, torch.full_like(colour_idx, self.k)).amin(dim=1)
        min_conflict_colour = cw_v.argmin(dim=1)
        min_conflict_weight = cw_v.gather(1, min_conflict_colour.unsqueeze(1)).squeeze(1)
        chosen_colour = torch.where(has_available, first_avail, min_conflict_colour)
        reward = torch.where(has_available,
                              torch.full_like(min_conflict_weight, self.reward_weight),
                              -min_conflict_weight)

        apply = active
        new_x = self.x.clone()
        new_coloured = self.coloured.clone()
        new_x[idx[apply], v_idx[apply]] = chosen_colour[apply]
        new_coloured[idx[apply], v_idx[apply]] = True
        self.x = new_x
        self.coloured = new_coloured
        reward = reward * active.float()
        self.t += 1
        done = ~self.action_mask().any(dim=1)      # nothing real left to colour
        return reward, done


# ======================================================================
# GNN policy (size-generalisable) + value head
# ======================================================================
class GNNPolicy(nn.Module):
    def __init__(self, k, hidden=128, layers=4):
        super().__init__()
        self.k = k
        f_in = 2 * k + 5                       # colour one-hot, hist, +4 scalars, +recency
        self.inp = nn.Linear(f_in, hidden)
        self.msg = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])
        self.colour_head = nn.Linear(hidden, k)
        # Prior strength and the value head are conditioned on log(graph
        # size) instead of being one fixed global scalar/head, so they can
        # adapt to very different conflict-count scales at small vs. large
        # N without stepping on each other during training. log(N) also
        # extrapolates fine to sizes never seen in training.
        self.prior_net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.prior_net[-1].weight)
        nn.init.constant_(self.prior_net[-1].bias, 1.0)  # softplus(1.0) ~= 1.31, a reasonably moderate start
        self.val = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def _size_feat(self, n, device, dtype):
        return torch.full((1, 1), math.log(n), device=device, dtype=dtype)

    def prior_at(self, n):
        """Scalar prior strength at graph size n, for logging/inspection."""
        device = next(self.parameters()).device
        with torch.no_grad():
            return F.softplus(self.prior_net(self._size_feat(n, device, torch.float32))).item()

    def features(self, A, x, recency, node_counts=None):
        B, N = x.shape
        k = self.k
        if node_counts is None:
            node_counts = torch.full((B,), N, device=A.device, dtype=torch.long)
        deg = A.sum(-1, keepdim=True)                       # (B,N,1)
        deg_safe = deg.clamp_min(1.0)
        nc = neighbour_colour_counts(A, x, k)               # (B,N,k)
        onehot = F.one_hot(x, num_classes=k).float()        # (B,N,k)
        own = nc.gather(-1, x.unsqueeze(-1))                # (B,N,1) conflicts at node
        sat = (nc > 0).float().sum(-1, keepdim=True) / k    # (B,N,1)
        hist = nc / deg_safe                                # (B,N,k) normalised
        recency_norm = recency.clamp(max=RECENCY_CAP).unsqueeze(-1) / RECENCY_CAP  # (B,N,1)
        deg_rel = deg / node_counts.view(B, 1, 1).to(deg.dtype)  # relative to the TRUE node count
        feats = torch.cat([
            onehot, hist,
            deg_rel,
            own / deg_safe,
            (own > 0).float(),
            sat,
            recency_norm,
        ], dim=-1)                                          # (B,N,2k+5)
        return feats, nc, deg_safe

    def forward(self, A, x, recency, node_counts=None):
        B, N = x.shape
        if node_counts is None:
            node_counts = torch.full((B,), N, device=A.device, dtype=torch.long)
        feats, nc, deg_safe = self.features(A, x, recency, node_counts)
        A_norm = A / deg_safe                               # row-normalised aggregation
        h = torch.relu(self.inp(feats))
        for lin in self.msg:
            agg = torch.bmm(A_norm, h)                      # mean of neighbours
            h = torch.relu(lin(torch.cat([h, agg], dim=-1))) + h
        size_feat = torch.log(node_counts.to(h.dtype)).unsqueeze(-1)         # (B,1)
        prior_strength = F.softplus(self.prior_net(size_feat)).view(B, 1, 1)  # (B,1,1)
        # Kept as (B,N,k), not flattened. ColoringEnv's single flat N*k
        # categorical does that itself via `flatten_logits()`.
        logits = self.colour_head(h) - prior_strength * nc   # (B,N,k)
        valid = (torch.arange(N, device=A.device).view(1, N) < node_counts.view(B, 1)).to(h.dtype)  # (B,N)
        h_mean = (h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        val_in = torch.cat([h_mean, size_feat], dim=-1)      # (B, hidden+1)
        value = self.val(val_in).squeeze(-1)                # (B,)
        return logits, value


class GNNPolicyConstruct(nn.Module):
    """Policy for `ColoringEnvConstruct`: scores every not-yet-coloured
    vertex and picks which one to colour next. The colour itself is
    decided by the environment (see `ColoringEnvConstruct.step`), so
    unlike `GNNPolicy` there's just one selection logit per vertex, not k
    colour logits. Same GraphSAGE message-passing body as `GNNPolicy`; the
    features differ because the colouring is *partial* here -- there's an
    explicit "not yet coloured" bucket to account for.
    """

    def __init__(self, k, hidden=128, layers=4):
        super().__init__()
        self.k = k
        f_in = 2 * k + 6                       # own colour-or-uncoloured one-hot (k+1), neighbour
        self.inp = nn.Linear(f_in, hidden)     # colour histogram (k), +4 scalars (see features())
        self.msg = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])
        self.select_head = nn.Linear(hidden, 1)
        self.val = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def features(self, A, x, coloured, node_counts):
        B, N = x.shape
        k = self.k
        deg = A.sum(-1, keepdim=True)                                # (B,N,1) degree over ALL neighbours
        deg_safe = deg.clamp_min(1.0)
        nc = coloured_neighbour_weights(A, x, coloured, k)           # (B,N,k) conflict weight per candidate colour
        own_idx = torch.where(coloured, x, torch.full_like(x, k))    # k = "not yet coloured" bucket
        own_oh = F.one_hot(own_idx, num_classes=k + 1).float()       # (B,N,k+1)
        hist = nc / deg_safe                                          # (B,N,k) normalised
        coloured_deg = torch.bmm(A, coloured.float().unsqueeze(-1))   # (B,N,1) weighted degree to coloured nbrs
        frac_coloured_nb = coloured_deg / deg_safe                    # (B,N,1) how "settled" this neighbourhood is
        sat_degree = (nc > 1e-9).float().sum(-1, keepdim=True) / k    # (B,N,1) distinct colours already blocking
        has_avail = (nc <= 1e-9).any(dim=-1, keepdim=True).float()    # (B,N,1) would this vertex get a clean colour now?
        min_cw_norm = nc.min(dim=-1, keepdim=True).values / deg_safe  # (B,N,1) cost if forced, normalised
        deg_rel = deg / node_counts.view(B, 1, 1).to(deg.dtype)       # (B,N,1) relative to true graph size
        feats = torch.cat([own_oh, hist, deg_rel, frac_coloured_nb,
                            sat_degree, has_avail, min_cw_norm], dim=-1)   # (B,N,2k+6)
        return feats

    def forward(self, A, x, coloured, node_counts=None):
        B, N = x.shape
        if node_counts is None:
            node_counts = torch.full((B,), N, device=A.device, dtype=torch.long)
        feats = self.features(A, x, coloured, node_counts)
        deg_safe = A.sum(-1, keepdim=True).clamp_min(1.0)
        A_norm = A / deg_safe
        h = torch.relu(self.inp(feats))
        for lin in self.msg:
            agg = torch.bmm(A_norm, h)
            h = torch.relu(lin(torch.cat([h, agg], dim=-1))) + h
        logits = self.select_head(h).squeeze(-1)              # (B,N) one score per vertex
        valid = (torch.arange(N, device=A.device).view(1, N) < node_counts.view(B, 1)).to(h.dtype)  # (B,N)
        h_mean = (h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        progress = ((coloured.float() * valid).sum(dim=1, keepdim=True)
                    / valid.sum(dim=1, keepdim=True).clamp_min(1))    # how far through the graph we are
        size_feat = torch.log(node_counts.to(h.dtype)).unsqueeze(-1)
        val_in = torch.cat([h_mean, size_feat, progress], dim=-1)   # (B, hidden+2)
        value = self.val(val_in).squeeze(-1)
        return logits, value


def flatten_logits(logits):
    """(B,N,k) -> (B,N*k), for ColoringEnv's single flat categorical per step."""
    B, N, k = logits.shape
    return logits.reshape(B, N * k)


def masked_categorical(logits, mask):
    """Categorical over legal actions; rows with no legal action get uniform."""
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg))
    empty = ~mask.any(dim=-1)      # guard fully-masked rows so we don't get NaNs
    masked[empty] = 0.0
    return torch.distributions.Categorical(logits=masked)


# ======================================================================
# Classical baselines
# ======================================================================
def dsatur_np(adj, k):
    n = adj.shape[0]
    a = np.full(n, -1, dtype=np.int64)
    for _ in range(n):
        oh = np.zeros((n, k), dtype=np.float32)
        m = a >= 0
        oh[m, a[m]] = 1.0
        nc = adj @ oh
        score = (nc > 0).sum(1) * n + adj.sum(1)
        score[m] = -1
        v = int(np.argmax(score))
        a[v] = int(np.argmin(nc[v]))
    return a


def conflicts_np(adj, a):
    """Total conflict cost (sum of weights of monochromatic edges). Kept
    as a float rather than an int so weighted fractional costs don't get
    truncated away."""
    return float(np.sum(adj * (a[:, None] == a[None, :]))) / 2.0


def tabucol_np(adj, a, k, iters, tenure=10, seed=0):
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    a = a.copy().astype(np.int64)
    tabu = np.zeros((n, k), dtype=np.int64)
    best, best_c = a.copy(), conflicts_np(adj, a)
    for it in range(iters):
        oh = np.zeros((n, k), dtype=np.float32); oh[np.arange(n), a] = 1.0
        nc = adj @ oh
        cur = nc[np.arange(n), a]
        c = float(cur.sum()) / 2.0
        if c <= 1e-9:
            return 0.0
        V = np.where(cur > 0)[0]
        delta = nc[V] - cur[V][:, None]
        delta[np.arange(len(V)), a[V]] = 1e9
        blocked = (tabu[V] > it) & ((c + delta) >= best_c)
        eff = np.where(blocked, 1e9, delta)
        flat = int(np.argmin(eff)); vi, col = divmod(flat, k); v = int(V[vi])
        if eff[vi, col] >= 1e9:
            v = int(rng.choice(V)); col = int(rng.integers(0, k))
        tabu[v, a[v]] = it + tenure + int(rng.integers(0, 5))
        a[v] = col
        nc_new = conflicts_np(adj, a)
        if nc_new < best_c:
            best_c = nc_new; best = a.copy()
    return best_c


# ======================================================================
# Rollout + PPO
# ======================================================================
@torch.no_grad()
def run_agent(policy, A, k, horizon, greedy=False, restarts=1, x0=None, node_counts=None):
    """Runs the learned policy as a local search and returns the best
    conflict count found per graph. `restarts` random initialisations run
    in parallel and the best is kept. `x0`, if given, is a (G,N) initial
    colouring (e.g. from DSATUR) used instead of a random start -- handy
    for isolating search quality from cold-start recovery. `node_counts`:
    see `ColoringEnv`.
    """
    G = A.shape[0]
    At = A.repeat(restarts, 1, 1)                       # (restarts*G, N, N)
    nct = node_counts.repeat(restarts) if node_counts is not None else None
    env = ColoringEnv(At, k, horizon, node_counts=nct)
    B = At.shape[0]
    if x0 is None:
        x = torch.randint(0, k, (B, At.shape[1]), device=At.device)
    else:
        x = x0.repeat(restarts, 1)
    env.reset(x)
    best = env.conf.clone()
    done = torch.zeros(B, dtype=torch.bool, device=At.device)
    for _ in range(horizon):
        logits, _ = policy(env.A, env.x, env.recency, env.node_counts)
        dist = masked_categorical(flatten_logits(logits), env.action_mask())
        action = dist.probs.argmax(-1) if greedy else dist.sample()
        reward, step_done, solved = env.step(action, ~done)
        best = torch.minimum(best, env.conf)
        done = done | step_done
        if done.all():
            break
    return best.reshape(restarts, G).min(dim=0).values      # (G,)


@torch.no_grad()
def run_agent_construct(policy, A, k, horizon=None, greedy=False, restarts=1, x0=None, node_counts=None):
    """Constructive-MDP version of `run_agent` -- see `ColoringEnvConstruct`.
    `horizon` and `x0` are accepted but not used; they're only there so
    call sites written for `run_agent`'s signature don't need a separate
    path. An episode here always runs for `A.shape[1]` steps, which is
    enough to colour every real vertex once, and construction always
    starts from an empty colouring anyway.
    """
    G = A.shape[0]
    At = A.repeat(restarts, 1, 1)
    nct = node_counts.repeat(restarts) if node_counts is not None else None
    env = ColoringEnvConstruct(At, k, node_counts=nct)
    B, N = At.shape[0], At.shape[1]
    env.reset()
    done = torch.zeros(B, dtype=torch.bool, device=At.device)
    for _ in range(N):
        logits, _ = policy(env.A, env.x, env.coloured, env.node_counts)
        mask = env.action_mask()
        dist = masked_categorical(logits, mask)
        v_idx = dist.probs.argmax(-1) if greedy else dist.sample()
        _, step_done = env.step(v_idx, ~done)
        done = done | step_done
        if done.all():
            break
    final = total_conflicts(env.A, env.x, k)
    return final.reshape(restarts, G).min(dim=0).values      # (G,)


def train_v2(args):
    """PPO training for `ColoringEnv` (the repair MDP) against a fixed
    dataset. With `--dataset`, draws shuffled batches from a pre-generated
    pool and tracks a real epoch count. Without it, generates a single
    fixed --n batch online each iteration -- fine for a quick smoke test,
    not meant for a real run.
    """
    device = args.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    k = args.colors

    train_pool = val_pool = None
    n_ref = args.n
    if args.dataset:
        train_pool, val_pool, n_lo, n_hi, dataset_k = load_fixed_dataset(args.dataset, device)
        assert dataset_k == k, (
            f"--dataset was generated with k={dataset_k} (chi(G) > {dataset_k} guaranteed), "
            f"but --colors={k} was requested; regenerate the dataset with --colors {k}.")
        n_ref = n_lo
        print(f"Loaded fixed dataset {args.dataset}: N in [{n_lo},{n_hi}], k={dataset_k} "
              f"(train {len(train_pool.adj)}, val {len(val_pool.adj)})", flush=True)

    policy = GNNPolicy(k, hidden=args.hidden, layers=args.layers).to(device)
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ck["state_dict"])
        print(f"Resumed weights from {args.resume}", flush=True)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    log = []

    for it in range(1, args.iters + 1):
        if args.lr_final is not None:
            frac = (it - 1) / max(1, args.iters - 1)
            for g in opt.param_groups:
                g["lr"] = args.lr + (args.lr_final - args.lr) * frac

        if train_pool is not None:
            A, node_counts = train_pool.next_batch(args.batch, device)
            N = A.shape[1]
        else:
            N = args.n
            A = sample_graphs(args.batch, N, args.p_lo, args.p_hi, device, rng, args.w_lo, args.w_hi, force_k=k)
            node_counts = torch.full((args.batch,), N, device=device, dtype=torch.long)
        batch, horizon = args.batch, args.horizon
        # Scale reward per graph so returns stay comparable across a batch
        # of mixed sizes. Without this the value head and prior_net would
        # see return magnitude swing purely from graph size, which causes
        # training on one size range to interfere with another.
        reward_scale = n_ref / node_counts.float()          # (batch,)

        env = ColoringEnv(A, k, horizon, node_counts=node_counts)
        x0 = torch.randint(0, k, (batch, N), device=device)
        env.reset(x0)
        init_conf = env.conf.clone()

        S_x, S_rec, S_mask, S_act, S_logp, S_val, S_rew, S_active = [], [], [], [], [], [], [], []
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(horizon):
            active = ~done
            x_cur = env.x.clone()
            rec_cur = env.recency.clone()
            mask = env.action_mask()
            with torch.no_grad():
                logits, value = policy(env.A, env.x, env.recency, env.node_counts)
                dist = masked_categorical(flatten_logits(logits), mask)
                action = dist.sample()
                logp = dist.log_prob(action)
            reward, step_done, solved = env.step(action, active)
            S_x.append(x_cur); S_rec.append(rec_cur); S_mask.append(mask); S_act.append(action)
            S_logp.append(logp); S_val.append(value); S_rew.append(reward * reward_scale)
            S_active.append(active.float())
            done = done | step_done
            if done.all():
                break

        T = len(S_act)
        with torch.no_grad():
            _, last_val = policy(env.A, env.x, env.recency, env.node_counts)
        # GAE
        adv = torch.zeros(batch, device=device)
        advs = [None] * T
        for t in reversed(range(T)):
            nextval = last_val if t == T - 1 else S_val[t + 1]
            nonterm = S_active[t]  # 1 if this transition counted
            delta = S_rew[t] + args.gamma * nextval * nonterm - S_val[t]
            adv = delta + args.gamma * args.lam * adv * nonterm
            advs[t] = adv.clone()
        returns = [advs[t] + S_val[t] for t in range(T)]

        # Flattened. Row i belongs to timestep t=i//batch, graph b=i%batch,
        # so env.A[mb % batch] finds the right adjacency without copying.
        bx = torch.cat(S_x, 0)
        brec = torch.cat(S_rec, 0)
        bmask = torch.cat(S_mask, 0)
        bact = torch.cat(S_act, 0)
        blogp = torch.cat(S_logp, 0)
        badv = torch.cat(advs, 0)
        bret = torch.cat(returns, 0)
        bactive = torch.cat(S_active, 0)
        badv = (badv - badv[bactive > 0].mean()) / (badv[bactive > 0].std() + 1e-8)

        idx_all = torch.arange(bx.shape[0], device=device)
        for _ in range(args.epochs):
            perm = idx_all[torch.randperm(idx_all.shape[0], device=device)]
            for s in range(0, perm.shape[0], args.minibatch):
                mb = perm[s:s + args.minibatch]
                a_idx = mb % batch
                logits, value = policy(env.A[a_idx], bx[mb], brec[mb], env.node_counts[a_idx])
                dist = masked_categorical(flatten_logits(logits), bmask[mb])
                new_logp = dist.log_prob(bact[mb])
                ratio = torch.exp(new_logp - blogp[mb])
                w = bactive[mb]
                s1 = ratio * badv[mb]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * badv[mb]
                pi_loss = -(torch.min(s1, s2) * w).sum() / w.sum().clamp_min(1)
                v_loss = (((value - bret[mb]) ** 2) * w).sum() / w.sum().clamp_min(1)
                ent = (dist.entropy() * w).sum() / w.sum().clamp_min(1)
                loss = pi_loss + args.vf * v_loss - args.ent * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        if it % args.log_every == 0 or it == 1:
            # `args.epochs` is PPO's gradient epochs per collected batch;
            # `dataset_epoch` is a separate thing, a full shuffled pass
            # over the fixed --dataset pool.
            if val_pool is not None:
                A_eval, nodes_eval = val_pool.next_batch(min(batch, len(val_pool.adj)), device)
                with torch.no_grad():
                    final = run_agent(policy, A_eval, k, horizon, greedy=True, node_counts=nodes_eval)
                epoch_str = f" | dataset_epoch {train_pool.epochs:3d}"
            else:
                with torch.no_grad():
                    final = run_agent(policy, A, k, horizon, greedy=True, node_counts=node_counts)
                epoch_str = ""
            n_mean = float(node_counts.float().mean())
            msg = (f"it {it:5d} | N~{n_mean:5.1f} (min {int(node_counts.min())}-max {int(node_counts.max())}) "
                   f"| batch {batch:4d} | horizon {horizon:4d} "
                   f"| init_conf {init_conf.mean():7.2f} | agent_best {final.mean():7.2f} "
                   f"| prior {policy.prior_at(max(1, round(n_mean))):.2f} | lr {opt.param_groups[0]['lr']:.2e}"
                   f"{epoch_str}")
            print(msg, flush=True)
            log.append((it, n_mean, float(init_conf.mean()), float(final.mean())))

        if args.save and (it % args.save_every == 0 or it == args.iters):
            torch.save({"state_dict": policy.state_dict(),
                        "k": k, "hidden": args.hidden, "layers": args.layers,
                        "mdp": "repair"}, args.save)

    return policy


def train_construct(args):
    """PPO training for `ColoringEnvConstruct` (the constructive MDP):
    build the colouring one vertex at a time; the environment colours
    each chosen vertex with its fixed rule, and the policy only learns
    which vertex to pick. Otherwise mirrors `train_v2`. No `--horizon`
    here -- an episode just runs for the graph's real node count.
    """
    device = args.device
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    k = args.colors

    train_pool = val_pool = None
    if args.dataset:
        train_pool, val_pool, n_lo, n_hi, dataset_k = load_fixed_dataset(args.dataset, device)
        assert dataset_k == k, (
            f"--dataset was generated with k={dataset_k} (chi(G) > {dataset_k} guaranteed), "
            f"but --colors={k} was requested; regenerate the dataset with --colors {k}.")
        print(f"[construct] Loaded fixed dataset {args.dataset}: N in [{n_lo},{n_hi}], k={dataset_k} "
              f"(train {len(train_pool.adj)}, val {len(val_pool.adj)})", flush=True)

    policy = GNNPolicyConstruct(k, hidden=args.hidden, layers=args.layers).to(device)
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ck["state_dict"])
        print(f"Resumed weights from {args.resume}", flush=True)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    for it in range(1, args.iters + 1):
        if args.lr_final is not None:
            frac = (it - 1) / max(1, args.iters - 1)
            for g in opt.param_groups:
                g["lr"] = args.lr + (args.lr_final - args.lr) * frac

        if train_pool is not None:
            A, node_counts = train_pool.next_batch(args.batch, device)
            N = A.shape[1]
        else:
            N = args.n
            A = sample_graphs(args.batch, N, args.p_lo, args.p_hi, device, rng, args.w_lo, args.w_hi, force_k=k)
            node_counts = torch.full((args.batch,), N, device=device, dtype=torch.long)
        batch = args.batch

        env = ColoringEnvConstruct(A, k, node_counts=node_counts, reward_weight=args.reward_weight)
        env.reset()

        S_x, S_col, S_mask, S_act, S_logp, S_val, S_rew, S_active = [], [], [], [], [], [], [], []
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(N):
            active = ~done
            x_cur = env.x.clone()
            col_cur = env.coloured.clone()
            mask = env.action_mask()
            with torch.no_grad():
                logits, value = policy(env.A, env.x, env.coloured, env.node_counts)
                dist = masked_categorical(logits, mask)
                action = dist.sample()
                logp = dist.log_prob(action)
            reward, step_done = env.step(action, active)
            S_x.append(x_cur); S_col.append(col_cur); S_mask.append(mask); S_act.append(action)
            S_logp.append(logp); S_val.append(value); S_rew.append(reward)
            S_active.append(active.float())
            done = done | step_done
            if done.all():
                break

        T = len(S_act)
        with torch.no_grad():
            _, last_val = policy(env.A, env.x, env.coloured, env.node_counts)
        # GAE, same as in train_v2, one scalar reward per graph per step
        adv = torch.zeros(batch, device=device)
        advs = [None] * T
        for t in reversed(range(T)):
            nextval = last_val if t == T - 1 else S_val[t + 1]
            nonterm = S_active[t]
            delta = S_rew[t] + args.gamma * nextval * nonterm - S_val[t]
            adv = delta + args.gamma * args.lam * adv * nonterm
            advs[t] = adv.clone()
        returns = [advs[t] + S_val[t] for t in range(T)]

        bx = torch.cat(S_x, 0)
        bcol = torch.cat(S_col, 0)
        bmask = torch.cat(S_mask, 0)
        bact = torch.cat(S_act, 0)
        blogp = torch.cat(S_logp, 0)
        badv = torch.cat(advs, 0)
        bret = torch.cat(returns, 0)
        bactive = torch.cat(S_active, 0)
        badv = (badv - badv[bactive > 0].mean()) / (badv[bactive > 0].std() + 1e-8)

        idx_all = torch.arange(bx.shape[0], device=device)
        for _ in range(args.epochs):
            perm = idx_all[torch.randperm(idx_all.shape[0], device=device)]
            for s in range(0, perm.shape[0], args.minibatch):
                mb = perm[s:s + args.minibatch]
                a_idx = mb % batch
                logits, value = policy(env.A[a_idx], bx[mb], bcol[mb], env.node_counts[a_idx])
                dist = masked_categorical(logits, bmask[mb])
                new_logp = dist.log_prob(bact[mb])
                ratio = torch.exp(new_logp - blogp[mb])
                w = bactive[mb]
                s1 = ratio * badv[mb]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * badv[mb]
                pi_loss = -(torch.min(s1, s2) * w).sum() / w.sum().clamp_min(1)
                v_loss = (((value - bret[mb]) ** 2) * w).sum() / w.sum().clamp_min(1)
                ent = (dist.entropy() * w).sum() / w.sum().clamp_min(1)
                loss = pi_loss + args.vf * v_loss - args.ent * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        if it % args.log_every == 0 or it == 1:
            if val_pool is not None:
                A_eval, nodes_eval = val_pool.next_batch(min(batch, len(val_pool.adj)), device)
                with torch.no_grad():
                    final = run_agent_construct(policy, A_eval, k, node_counts=nodes_eval)
                epoch_str = f" | dataset_epoch {train_pool.epochs:3d}"
            else:
                with torch.no_grad():
                    final = run_agent_construct(policy, A, k, node_counts=node_counts)
                epoch_str = ""
            n_mean = float(node_counts.float().mean())
            msg = (f"[construct] it {it:5d} | N~{n_mean:5.1f} "
                   f"(min {int(node_counts.min())}-max {int(node_counts.max())}) "
                   f"| batch {batch:4d} "
                   f"| agent_final_conflicts {final.mean():7.2f} | lr {opt.param_groups[0]['lr']:.2e}"
                   f"{epoch_str}")
            print(msg, flush=True)

        if args.save and (it % args.save_every == 0 or it == args.iters):
            torch.save({"state_dict": policy.state_dict(),
                        "k": k, "hidden": args.hidden, "layers": args.layers,
                        "mdp": "construct"}, args.save)

    return policy


def evaluate(args):
    device = args.device
    ck = torch.load(args.model, map_location=device)
    mdp = ck.get("mdp", "repair")      # picks the right policy class and rollout fn
    policy_cls = GNNPolicyConstruct if mdp == "construct" else GNNPolicy
    agent_fn = {"repair": run_agent, "construct": run_agent_construct}[mdp]
    policy = policy_cls(ck["k"], hidden=ck["hidden"], layers=ck["layers"]).to(device)
    policy.load_state_dict(ck["state_dict"]); policy.eval()
    k = ck["k"]
    rng = np.random.default_rng(123)

    A = sample_graphs(args.eval_graphs, args.n, args.p, args.p, device, rng, args.w_lo, args.w_hi, force_k=k)
    t0 = time.time()
    agent = agent_fn(policy, A, k, args.horizon, greedy=False,
                      restarts=args.restarts).cpu().numpy()
    agent_t = (time.time() - t0) / args.eval_graphs
    adj_np = A.cpu().numpy()
    ds, ds_tabu, tabu_rand = [], [], []
    for b in range(args.eval_graphs):
        adj = adj_np[b]
        d = dsatur_np(adj, k)
        ds.append(conflicts_np(adj, d))
        ds_tabu.append(tabucol_np(adj, d, k, args.tabu_iters))
        r = rng.integers(0, k, adj.shape[0])
        tabu_rand.append(tabucol_np(adj, r, k, args.tabu_iters))
    print(f"\np={args.p} (N={args.n}, {args.eval_graphs} graphs, mdp={mdp}):")
    print(f"  DSATUR              {np.mean(ds):6.2f}")
    print(f"  Neural-LS (ours)    {np.mean(agent):6.2f}   ({agent_t*1000:.1f} ms/graph)")
    print(f"  DSATUR+TabuCol      {np.mean(ds_tabu):6.2f}")
    print(f"  Random+TabuCol      {np.mean(tabu_rand):6.2f}")


def build_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--n", type=int, default=50)
        sp.add_argument("--colors", type=int, default=7)
        sp.add_argument("--hidden", type=int, default=128)
        sp.add_argument("--layers", type=int, default=4)
        sp.add_argument("--horizon", type=int, default=120)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--w_lo", type=float, default=1.0,
                         help="lower bound of per-edge weight ~Uniform(w_lo,w_hi). Default (1.0,1.0) = unweighted.")
        sp.add_argument("--w_hi", type=float, default=1.0,
                         help="upper bound of per-edge weight; see --w_lo.")

    t = sub.add_parser("train"); common(t)
    t.add_argument("--mdp", choices=["repair", "construct"], default="repair",
                    help="repair (default): local-search repair, one (vertex,colour) move per "
                         "step from a random full colouring (ColoringEnv/train_v2). construct: "
                         "build the colouring one not-yet-coloured vertex at a time; the "
                         "environment picks each vertex's colour, the policy only picks the "
                         "vertex order (ColoringEnvConstruct/train_construct).")
    t.add_argument("--reward_weight", type=float, default=1.0,
                    help="--mdp construct only: reward for a clean vertex assignment. A forced "
                         "(conflicting) assignment is always -conflict_weight_incurred.")
    t.add_argument("--iters", type=int, default=3000)
    t.add_argument("--dataset", default=None,
                    help="path to a fixed dataset directory from generate_overconstrained_dataset.py "
                         "(--out <dir>). If given, training draws shuffled batches from this "
                         "pre-generated pool and logs a real epoch count. Omit for a quick "
                         "online-generated smoke test (uses --n).")
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--resume", default=None, help="checkpoint to load weights from before training")
    t.add_argument("--p_lo", type=float, default=0.10, help="only used without --dataset")
    t.add_argument("--p_hi", type=float, default=0.50, help="only used without --dataset")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--lr_final", type=float, default=None,
                    help="if set, linearly anneal LR from --lr to this value (default: constant LR)")
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--lam", type=float, default=0.95)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--vf", type=float, default=0.5)
    t.add_argument("--ent", type=float, default=0.01)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--minibatch", type=int, default=1024)
    t.add_argument("--log_every", type=int, default=10)
    t.add_argument("--save", default="models/neural_color.pt")
    t.add_argument("--save_every", type=int, default=100)

    e = sub.add_parser("eval"); common(e)
    e.add_argument("--model", default="models/neural_color.pt")
    e.add_argument("--eval_graphs", type=int, default=50)
    e.add_argument("--tabu_iters", type=int, default=2000)
    e.add_argument("--restarts", type=int, default=8)
    e.add_argument("--p", type=float, default=0.15,
                    help="fixed Erdos-Renyi edge probability for eval graphs (matches training).")
    return p


if __name__ == "__main__":
    args = build_args().parse_args()
    if args.cmd == "train":
        (train_construct if args.mdp == "construct" else train_v2)(args)
    else:
        evaluate(args)
