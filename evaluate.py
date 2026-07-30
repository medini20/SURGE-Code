"""Canonical benchmark: Neural-LS (ours) vs Gurobi vs DSATUR vs TabuCol on
min-conflict k-colouring, all methods run on the same graph instances.

Two modes: `--dataset <dir>` evaluates on the real held-out validation
split from a fixed dataset built by generate_overconstrained_dataset.py
(--n/--p get ignored, defaults to the whole split unless you pass
--graphs). Without --dataset, it just generates `--graphs` random graphs
at fixed N=`--n` and edge probability `--p` (0.15 by default, matching
training).

Also runs a Wilcoxon signed-rank test to check whether the gap to
DSATUR+TabuCol is actually significant.

The model checkpoint is a plain PyTorch state dict (models/neural_color.pt),
produced by `python neural_coloring.py train ...`.
"""
import argparse
import json
import time
import numpy as np
import torch

from neural_coloring import (
    GNNPolicy, GNNPolicyConstruct, run_agent, run_agent_construct,
    dsatur_np, conflicts_np, tabucol_np, gen_graphs, load_fixed_dataset,
)

_GUROBI_ENV = None


def _gurobi_env():
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        import gurobipy as gp
        _GUROBI_ENV = gp.Env(empty=True)
        _GUROBI_ENV.setParam("OutputFlag", 0)
        _GUROBI_ENV.start()
    return _GUROBI_ENV


def solve_ilp(adj, k, time_limit=10.0):
    """Minimum-conflict (weighted) k-colouring solved exactly as a Gurobi
    MILP; returns (cost, solve_time). The objective sums adj[i,j]*y[i,j],
    so for unweighted (0/1) adjacency this is just a violated-edge count."""
    import gurobipy as gp
    from gurobipy import GRB
    n = adj.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0]
    m = gp.Model("gc", env=_gurobi_env())
    m.Params.TimeLimit = time_limit
    x = m.addVars(n, k, vtype=GRB.BINARY)
    y = m.addVars(edges, vtype=GRB.BINARY)
    m.addConstrs((x.sum(i, "*") == 1 for i in range(n)))
    for (i, j) in edges:
        for c in range(k):
            m.addConstr(x[i, c] + x[j, c] - 1 <= y[i, j])
    m.setObjective(gp.quicksum(float(adj[i, j]) * y[i, j] for (i, j) in edges), GRB.MINIMIZE)
    t0 = time.time()
    m.optimize()
    dt = time.time() - t0
    if m.SolCount > 0:
        return float(m.ObjVal), dt
    return float("inf"), dt


def run_tier(policy, k, args, tier, graphs, seed, node_counts=None, agent_fn=run_agent):
    """Evaluate all methods on `graphs` (list of 2D numpy adjacency arrays).

    `node_counts`, if given, is each graph's true node count for a batch
    that's been zero-padded to a common max size. Baselines run on the
    cropped `adj[:n, :n]`, while Neural-LS runs on the full padded batch
    with `node_counts` passed through for correct pooling/masking. Leave
    it out for a same-size batch.

    `agent_fn` is `run_agent` or `run_agent_construct`, whichever matches
    the MDP `policy` was trained under (`main()` figures this out from
    `ck["mdp"]`). One quirk: `run_agent_construct` ignores `x0`, so
    "Neural-LS (DSATUR-init)" just equals plain "Neural-LS" for those
    checkpoints -- that's expected, not a bug.
    """
    n_graphs = len(graphs)
    A = torch.tensor(np.stack(graphs), device=args.device)
    nct = (torch.tensor(node_counts, dtype=torch.long, device=args.device)
           if node_counts is not None else None)

    t0 = time.time()
    agent = agent_fn(policy, A, k, args.horizon, greedy=False, restarts=args.restarts,
                      node_counts=nct).cpu().numpy()
    agent_time_ms = (time.time() - t0) / n_graphs * 1000

    ds_c, ds_t, tabu_c, tabu_t, gurobi_c, gurobi_t, ds_all_padded = [], [], [], [], [], [], []
    for i, adj in enumerate(graphs):
        n = int(node_counts[i]) if node_counts is not None else adj.shape[0]
        sub = adj[:n, :n]
        t0 = time.time(); d = dsatur_np(sub, k); ds_t.append(time.time() - t0)
        ds_c.append(conflicts_np(sub, d))
        t0 = time.time(); tc = tabucol_np(sub, d, k, args.tabu_iters); tabu_t.append(time.time() - t0)
        tabu_c.append(tc)
        if not args.skip_gurobi:
            gc, gt = solve_ilp(sub, k, time_limit=args.gurobi_time)
            gurobi_c.append(gc); gurobi_t.append(gt)
        padded_d = np.zeros(adj.shape[0], dtype=np.int64)
        padded_d[:n] = d
        ds_all_padded.append(padded_d)

    # Warm-started variant, same DSATUR start TabuCol gets, so we can see
    # search quality separately from cold-start recovery.
    x0 = torch.tensor(np.stack(ds_all_padded), dtype=torch.long, device=args.device)
    t0 = time.time()
    agent_warm = agent_fn(policy, A, k, args.horizon, greedy=False,
                           restarts=args.restarts, x0=x0, node_counts=nct).cpu().numpy()
    agent_warm_time_ms = (time.time() - t0) / n_graphs * 1000

    if node_counts is not None:
        n_desc = f"{int(min(node_counts))}-{int(max(node_counts))}"
    else:
        n_desc = str(graphs[0].shape[0])

    row = {
        "n_graphs": n_graphs, "n_nodes": n_desc, "k": k,
        "Neural-LS": {"mean": float(np.mean(agent)), "time_ms": agent_time_ms,
                      "raw": [float(v) for v in agent]},
        "Neural-LS (DSATUR-init)": {"mean": float(np.mean(agent_warm)), "time_ms": agent_warm_time_ms,
                                     "raw": [float(v) for v in agent_warm]},
        "DSATUR": {"mean": float(np.mean(ds_c)), "time_ms": float(np.mean(ds_t)) * 1000,
                   "raw": [float(v) for v in ds_c]},
        "DSATUR+TabuCol": {"mean": float(np.mean(tabu_c)), "time_ms": float(np.mean(tabu_t)) * 1000,
                           "raw": [float(v) for v in tabu_c]},
    }
    if not args.skip_gurobi:
        row["Gurobi"] = {"mean": float(np.mean(gurobi_c)), "time_ms": float(np.mean(gurobi_t)) * 1000,
                          "raw": [float(v) for v in gurobi_c]}

    try:
        from scipy.stats import wilcoxon
        for name, arr in [("Neural-LS", agent), ("Neural-LS (DSATUR-init)", agent_warm)]:
            diff = np.array(arr) - np.array(tabu_c)
            row[f"wilcoxon_p_{name}_vs_tabucol"] = float(wilcoxon(arr, tabu_c)[1]) if np.any(diff != 0) else 1.0
    except ImportError:
        print("  (scipy not installed -- skipping significance test; `pip install scipy` to enable)")

    print(f"\n{tier} (N={n_desc}, k={k}, {n_graphs} graphs):")
    print(f"  {'method':<24} {'conflicts':>10}  {'time/graph':>12}")
    for name in ["Gurobi", "DSATUR", "DSATUR+TabuCol", "Neural-LS", "Neural-LS (DSATUR-init)"]:
        if name not in row:
            continue
        v = row[name]
        print(f"  {name:<24} {v['mean']:>10.2f}  {v['time_ms']:>10.1f}ms")
    for name in ["Neural-LS", "Neural-LS (DSATUR-init)"]:
        key = f"wilcoxon_p_{name}_vs_tabucol"
        if key in row:
            print(f"  Wilcoxon {name} vs DSATUR+TabuCol: p={row[key]:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/neural_color.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset", default=None,
                     help="path to a fixed dataset directory (generate_overconstrained_dataset.py "
                          "--out ...); evaluates on its held-out val_adj.npy/val_nodes.npy split "
                          "instead of freshly-generated synthetic graphs. --n/--p are ignored.")
    ap.add_argument("--n", type=int, default=50, help="synthetic mode only; ignored with --dataset")
    ap.add_argument("--graphs", type=int, default=None,
                     help="number of graphs to evaluate. Default: 100 in synthetic mode, or the "
                          "*entire* validation split in --dataset mode.")
    ap.add_argument("--tabu_iters", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--gurobi_time", type=float, default=10.0)
    ap.add_argument("--skip_gurobi", action="store_true")
    ap.add_argument("--p", type=float, default=0.15,
                     help="synthetic mode only; fixed Erdos-Renyi edge probability for eval graphs, "
                          "matching VColRL/the training distribution exactly. Ignored with --dataset.")
    ap.add_argument("--w_lo", type=float, default=1.0,
                     help="lower bound of per-edge weight ~Uniform(w_lo,w_hi); "
                          "default (1.0,1.0) = unweighted.")
    ap.add_argument("--w_hi", type=float, default=1.0, help="upper bound of per-edge weight.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location=args.device)
    mdp = ck.get("mdp", "repair")      # tells us which policy class and rollout fn to use
    policy_cls = GNNPolicyConstruct if mdp == "construct" else GNNPolicy
    agent_fn = run_agent_construct if mdp == "construct" else run_agent
    print(f"Checkpoint mdp={mdp}")
    policy = policy_cls(ck["k"], hidden=ck["hidden"], layers=ck["layers"]).to(args.device)
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    k = ck["k"]

    results = {}
    if args.dataset:
        _, val_pool, n_lo, n_hi, dataset_k = load_fixed_dataset(args.dataset, args.device)
        if dataset_k != k:
            print(f"WARNING: dataset was generated for k={dataset_k}, but model checkpoint is k={k}")
        n_eval = len(val_pool.adj) if args.graphs is None else min(args.graphs, len(val_pool.adj))
        print(f"Evaluating on {n_eval}/{len(val_pool.adj)} graphs from the real validation split "
              f"in {args.dataset} (N in [{n_lo},{n_hi}])")
        graphs = [np.array(val_pool.adj[i], dtype=np.float32) for i in range(n_eval)]
        node_counts = np.asarray(val_pool.nodes[:n_eval])
        tier = "val_split"
        results[tier] = run_tier(policy, k, args, tier, graphs, seed=args.seed,
                                  node_counts=node_counts, agent_fn=agent_fn)
    else:
        n_graphs = args.graphs if args.graphs is not None else 100
        graphs = gen_graphs(n_graphs, args.n, args.p, args.p, seed=args.seed,
                             w_lo=args.w_lo, w_hi=args.w_hi, force_k=k)
        tier = f"p={args.p}"
        results[tier] = run_tier(policy, k, args, tier, graphs, seed=args.seed, agent_fn=agent_fn)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
