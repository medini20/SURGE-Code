"""Generates a fixed train/val dataset of graphs, all guaranteed to need
more than k colours, matching VColRL's dataset structure: graph size N is
drawn per-graph from a continuous range (default 50-100) at a single fixed
edge probability (default 0.15), split into train and held-out validation
sets (defaults: --count 25000, --val_count 1000). Generated once, then
repeatedly passed over in shuffled order during training instead of
generating graphs online.

Every graph gets a small (~k+4-8 node) randomly-structured subgraph planted
into it, checked with a fast Gurobi solve to confirm it needs more than k
colours on its own. Since a subgraph's chromatic number is a lower bound
on the whole graph's, that's just as solid a guarantee as planting a
(k+1)-clique, except the certificate looks different every time instead of
being the same clique motif over and over. Falls back to a plain
(k+1)-clique (exact by pigeonhole) if nothing in the density schedule
works out.

Graphs have different node counts, so they get padded with all-zero
rows/cols up to --n_hi, with the true node count stored alongside in a
separate array. A padding node has no edges, so it's never in conflict and
never gets picked by `ColoringEnv.action_mask()` -- the true count is only
needed by the policy's pooling/size-conditioning (see neural_coloring.GNNPolicy).

Each graph is written straight to a memory-mapped .npy file on disk, one
graph at a time, so RAM use stays flat no matter how big --count is.

Output layout (a directory, not a single file):
    <out>/meta.npz         -- n_lo, n_hi, k
    <out>/train_adj.npy    -- (n_train, n_hi, n_hi) float32, zero-padded
    <out>/train_nodes.npy  -- (n_train,) int32, true node count per graph
    <out>/val_adj.npy, val_nodes.npy -- same, for validation

Loaded by `neural_coloring.load_fixed_dataset` (train_v2 --dataset <out> ...).
"""
import argparse
import os
import numpy as np
import networkx as nx

from neural_coloring import plant_clique_np

_GUROBI_ENV = None


def _gurobi_env():
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        import gurobipy as gp
        _GUROBI_ENV = gp.Env(empty=True)
        _GUROBI_ENV.setParam("OutputFlag", 0)
        _GUROBI_ENV.start()
    return _GUROBI_ENV


def _small_subgraph_feasibility(edges, m, k, time_limit):
    """Exact Gurobi check on an m-node subgraph -- m doesn't depend on the
    overall graph size N, so this stays fast. Returns "INFEASIBLE" (this
    subgraph, and therefore any graph containing it, needs more than k
    colours), "FEASIBLE" (k-colourable, doesn't qualify), or "UNRESOLVED"
    (hit the time limit, inconclusive, never treated as a guarantee)."""
    import gurobipy as gp
    from gurobipy import GRB
    model = gp.Model("feas", env=_gurobi_env())
    model.Params.TimeLimit = time_limit
    x = model.addVars(m, k, vtype=GRB.BINARY)
    model.addConstrs(x.sum(i, "*") == 1 for i in range(m))
    for (i, j) in edges:
        for c in range(k):
            model.addConstr(x[i, c] + x[j, c] <= 1)
    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        return "INFEASIBLE"
    elif model.Status == GRB.OPTIMAL:
        return "FEASIBLE"
    return "UNRESOLVED"


def plant_random_certificate(adj, k, rng, stats, m_slack_range=(3, 7),
                              density_schedule=(0.55, 0.65, 0.75, 0.85, 0.95), time_limit=3.0):
    """Embeds a small, randomly-structured subgraph into `adj` in place
    that's provably guaranteed to need more than k colours. `stats` (a
    dict with "random"/"fallback_clique"/"density_sum" keys) gets updated
    in place so the caller can report how often each path got used."""
    n = adj.shape[0]
    m_slack = int(rng.integers(*m_slack_range))
    m = k + 1 + m_slack
    assert n >= m, f"need at least k+1+m_slack={m} nodes, got n={n} (raise --n_lo or lower m_slack_range)"
    nodes = rng.choice(n, size=m, replace=False)

    for p in density_schedule:
        mask = np.triu(rng.random((m, m)) < p, k=1)
        edges = list(zip(*np.where(mask)))
        if not edges:
            continue
        if _small_subgraph_feasibility(edges, m, k, time_limit) == "INFEASIBLE":
            for (i, j) in edges:
                u, v = int(nodes[i]), int(nodes[j])
                adj[u, v] = 1.0
                adj[v, u] = 1.0
            stats["random"] += 1
            stats["density_sum"] += p
            return adj

    stats["fallback_clique"] += 1
    plant_clique_np(adj, k, rng)
    return adj


def gen_one_graph(rng, n_lo, n_hi, p, force_k, stats):
    """Generate a single graph with N drawn uniformly from [n_lo, n_hi] at a
    single fixed edge probability p (matching VColRL's dataset exactly).
    Returns (padded_adj, n) where padded_adj is (n_hi, n_hi) with all-zero
    padding rows/cols beyond the first n."""
    n = int(rng.integers(n_lo, n_hi + 1))
    G = nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(0, 2**31 - 1)))
    adj = nx.to_numpy_array(G, dtype=np.float32)
    if force_k is not None:
        plant_random_certificate(adj, force_k, rng, stats)
    padded = np.zeros((n_hi, n_hi), dtype=np.float32)
    padded[:n, :n] = adj
    return padded, n


def validate_sample(adj_pool, nodes_pool, k, n_validate, time_limit=15.0):
    """Spot-check n_validate random graphs with a Gurobi feasibility solve on
    the *real* (unpadded) portion of each graph. Returns (n_checked,
    n_infeasible, n_unresolved, n_feasible)."""
    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    def is_k_colorable(adj, k):
        n = adj.shape[0]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0]
        m = gp.Model("feas", env=env)
        m.Params.TimeLimit = time_limit
        x = m.addVars(n, k, vtype=GRB.BINARY)
        m.addConstrs(x.sum(i, "*") == 1 for i in range(n))
        for (i, j) in edges:
            for c in range(k):
                m.addConstr(x[i, c] + x[j, c] <= 1)
        m.setObjective(0, GRB.MINIMIZE)
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            return "FEASIBLE"
        elif m.Status == GRB.INFEASIBLE:
            return "INFEASIBLE"
        return "UNRESOLVED"

    idx = np.random.default_rng(0).choice(len(adj_pool), size=min(n_validate, len(adj_pool)), replace=False)
    results = []
    for i in idx:
        n = int(nodes_pool[i])
        results.append(is_k_colorable(adj_pool[i][:n, :n], k))
    n_infeas = results.count("INFEASIBLE")
    n_unres = results.count("UNRESOLVED")
    n_feas = results.count("FEASIBLE")
    return len(results), n_infeas, n_unres, n_feas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_lo", type=int, default=50, help="minimum node count (inclusive)")
    ap.add_argument("--n_hi", type=int, default=100, help="maximum node count (inclusive); also the pad size")
    ap.add_argument("--p", type=float, default=0.15, help="fixed Erdos-Renyi edge probability (VColRL uses 0.15)")
    ap.add_argument("--colors", type=int, default=7, help="k -- every graph will need > k colours")
    ap.add_argument("--count", type=int, default=25000, help="total number of graphs to generate")
    ap.add_argument("--val_count", type=int, default=1000,
                     help="number of graphs held out for validation (VColRL: 1000 out of 15000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                     help="output directory (created if missing): meta.npz + train_adj.npy/train_nodes.npy "
                          "+ val_adj.npy/val_nodes.npy. Omit to skip saving (dry run).")
    ap.add_argument("--validate", type=int, default=30,
                     help="spot-check this many graphs with Gurobi before trusting the rest")
    args = ap.parse_args()

    assert args.count > args.val_count, "--count must be larger than --val_count"
    n_train = args.count - args.val_count
    n_val = args.val_count

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    train_adj_path = os.path.join(args.out, "train_adj.npy") if args.out else None
    train_nodes_path = os.path.join(args.out, "train_nodes.npy") if args.out else None
    val_adj_path = os.path.join(args.out, "val_adj.npy") if args.out else None
    val_nodes_path = os.path.join(args.out, "val_nodes.npy") if args.out else None

    if train_adj_path and os.path.exists(train_adj_path) and os.path.exists(val_adj_path):
        print(f"{args.out}/train_adj.npy and val_adj.npy already exist -- delete them to regenerate. Exiting.")
        return

    print(f"Generating {n_train} train + {n_val} val graphs, N in [{args.n_lo},{args.n_hi}], p={args.p}, "
          f"each guaranteed chi > {args.colors} via a randomly-structured, Gurobi-verified small certificate...")

    rng = np.random.default_rng(args.seed)
    stats = {"random": 0, "fallback_clique": 0, "density_sum": 0.0}

    if train_adj_path:
        train_adj = np.lib.format.open_memmap(train_adj_path, mode="w+", dtype=np.float32,
                                               shape=(n_train, args.n_hi, args.n_hi))
        val_adj = np.lib.format.open_memmap(val_adj_path, mode="w+", dtype=np.float32,
                                             shape=(n_val, args.n_hi, args.n_hi))
    else:
        train_adj = np.empty((n_train, args.n_hi, args.n_hi), dtype=np.float32)
        val_adj = np.empty((n_val, args.n_hi, args.n_hi), dtype=np.float32)
    train_nodes = np.empty(n_train, dtype=np.int32)
    val_nodes = np.empty(n_val, dtype=np.int32)

    for j in range(n_val):
        adj, n = gen_one_graph(rng, args.n_lo, args.n_hi, args.p, args.colors, stats)
        val_adj[j] = adj
        val_nodes[j] = n
        if (j + 1) % 200 == 0:
            print(f"  val {j + 1}/{n_val}")
    for j in range(n_train):
        adj, n = gen_one_graph(rng, args.n_lo, args.n_hi, args.p, args.colors, stats)
        train_adj[j] = adj
        train_nodes[j] = n
        if (j + 1) % 2000 == 0:
            print(f"  train {j + 1}/{n_train}")

    if isinstance(train_adj, np.memmap):
        train_adj.flush(); val_adj.flush()

    n_cert = stats["random"] + stats["fallback_clique"]
    avg_p = stats["density_sum"] / stats["random"] if stats["random"] else 0.0
    print(f"\nCertificate structure: {stats['random']}/{n_cert} randomly-structured "
          f"(avg density {avg_p:.2f}), {stats['fallback_clique']}/{n_cert} fell back to a plain clique.")

    any_bug = False
    if args.validate > 0:
        n_checked, n_infeas, n_unres, n_feas = validate_sample(train_adj, train_nodes, args.colors, args.validate)
        print(f"Spot-check: {n_checked} random graphs, Gurobi feasibility solve (whole graph) for a "
              f"proper {args.colors}-colouring:")
        print(f"  INFEASIBLE (confirms chi>{args.colors}, as guaranteed): {n_infeas}/{n_checked}")
        print(f"  FEASIBLE (would indicate a bug in the certificate):     {n_feas}/{n_checked}")
        print(f"  UNRESOLVED (time limit, inconclusive either way):       {n_unres}/{n_checked}")
        if n_feas > 0:
            print("  *** BUG: a planted-certificate graph was found FEASIBLE -- do not trust this dataset. ***")
            any_bug = True
        elif n_infeas == n_checked:
            print("  All spot-checked graphs confirmed over-constrained.")

    if not args.out:
        print("\n(not saved: --out omitted)")
    elif any_bug:
        print("\n*** Validation failed -- fix the bug before using this dataset. ***")
    else:
        np.save(train_nodes_path, train_nodes)
        np.save(val_nodes_path, val_nodes)
        np.savez(os.path.join(args.out, "meta.npz"), n_lo=args.n_lo, n_hi=args.n_hi, k=args.colors)
        print(f"\nDataset ready in {args.out}/. Use: train_v2 --dataset {args.out} --colors {args.colors}")


if __name__ == "__main__":
    main()
