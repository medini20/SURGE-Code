"""Quality-vs-wall-clock curves: Neural-LS vs DSATUR+TabuCol at matched
per-graph time budgets, at the same fixed edge density (p, default 0.15)
the model was trained on.

Neural-LS: one batched GPU rollout with restarts running in parallel,
checkpointed by horizon step -- per-graph time is just batch_elapsed /
n_graphs. TabuCol: a single-threaded per-graph trace (the real per-instance
cost), checkpointed by iteration count and averaged over the graph set.

Both curves share the same x-axis (ms/graph) so they can go on one plot.
"""
import argparse
import json
import time
import numpy as np
import torch

from neural_coloring import GNNPolicy, ColoringEnv, masked_categorical, dsatur_np, conflicts_np, gen_graphs


@torch.no_grad()
def neural_curve(policy, A, k, max_horizon, restarts, checkpoints, x0=None):
    G = A.shape[0]
    At = A.repeat(restarts, 1, 1)
    env = ColoringEnv(At, k, max_horizon)
    B = At.shape[0]
    if x0 is None:
        x = torch.randint(0, k, (B, At.shape[1]), device=At.device)
    else:
        x = x0.repeat(restarts, 1)
    env.reset(x)
    best = env.conf.clone()
    done = torch.zeros(B, dtype=torch.bool, device=At.device)
    cps = set(checkpoints)
    curve = []
    torch.cuda.synchronize() if At.is_cuda else None
    t0 = time.time()
    for t in range(1, max_horizon + 1):
        logits, _ = policy(env.A, env.x, env.recency)
        dist = masked_categorical(logits, env.action_mask())
        action = dist.sample()
        _, step_done, _ = env.step(action, ~done)
        best = torch.minimum(best, env.conf)
        done = done | step_done
        if t in cps:
            torch.cuda.synchronize() if At.is_cuda else None
            elapsed = time.time() - t0
            per_graph_best = best.reshape(restarts, G).min(dim=0).values
            curve.append({"horizon": t, "ms_per_graph": elapsed / G * 1000,
                          "mean_conflicts": float(per_graph_best.mean())})
        if done.all():
            break
    return curve


def tabucol_track(adj, a, k, iters, checkpoints, tenure=10, seed=0):
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    a = a.copy().astype(np.int64)
    tabu = np.zeros((n, k), dtype=np.int64)
    best, best_c = a.copy(), conflicts_np(adj, a)
    cps = set(checkpoints)
    t0 = time.time()
    trace = {}
    for it in range(iters):
        oh = np.zeros((n, k), dtype=np.float32); oh[np.arange(n), a] = 1.0
        nc = adj @ oh
        cur = nc[np.arange(n), a]
        c = int(cur.sum()) // 2
        if c == 0:
            best_c = 0
        else:
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
        if (it + 1) in cps:
            trace[it + 1] = (time.time() - t0, best_c)
    for cp in cps:
        if cp not in trace:
            trace[cp] = (time.time() - t0, best_c)
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/neural_color.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--graphs", type=int, default=50)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--max_horizon", type=int, default=800)
    ap.add_argument("--max_tabu_iters", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--init", choices=["random", "dsatur"], default="random")
    ap.add_argument("--p", type=float, default=0.15,
                     help="fixed Erdos-Renyi edge probability for eval graphs, matching "
                          "VColRL/the training distribution exactly.")
    ap.add_argument("--out_json", default="budget_curve.json")
    ap.add_argument("--out_png", default="budget_curve.png")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location=args.device)
    policy = GNNPolicy(ck["k"], hidden=ck["hidden"], layers=ck["layers"]).to(args.device)
    policy.load_state_dict(ck["state_dict"]); policy.eval()
    k = ck["k"]

    graphs = gen_graphs(args.graphs, args.n, args.p, args.p, seed=args.seed, force_k=k)
    A = torch.tensor(np.stack(graphs), device=args.device)

    x0 = None
    if args.init == "dsatur":
        ds_all = [dsatur_np(adj, k) for adj in graphs]
        x0 = torch.tensor(np.stack(ds_all), dtype=torch.long, device=args.device)

    horizon_cps = sorted(set([5, 10, 20, 40, 80, 120, 160, 240, 320, 480, 640,
                               960, 1280, 1600, 2000, 2400, 2800, args.max_horizon]))
    horizon_cps = [c for c in horizon_cps if c <= args.max_horizon]
    print(f"Running Neural-LS curve (init={args.init})...")
    neural = neural_curve(policy, A, k, args.max_horizon, args.restarts, horizon_cps, x0=x0)
    for row in neural:
        print(f"  horizon={row['horizon']:4d}  {row['ms_per_graph']:8.2f} ms/graph  conflicts={row['mean_conflicts']:.2f}")

    tabu_cps = sorted(set([50, 100, 250, 500, 1000, 2000, 4000, 8000, args.max_tabu_iters]))
    tabu_cps = [c for c in tabu_cps if c <= args.max_tabu_iters]
    print("\nRunning TabuCol curve (per-graph, single-threaded)...")
    inits = [dsatur_np(adj, k) for adj in graphs]
    accum = {cp: {"time": 0.0, "conf": 0.0} for cp in tabu_cps}
    for gi, (adj, init) in enumerate(zip(graphs, inits)):
        trace = tabucol_track(adj, init, k, args.max_tabu_iters, tabu_cps, seed=gi)
        for cp in tabu_cps:
            t, c = trace[cp]
            accum[cp]["time"] += t
            accum[cp]["conf"] += c
    tabucol = []
    for cp in tabu_cps:
        ms_per_graph = accum[cp]["time"] / args.graphs * 1000
        mean_conf = accum[cp]["conf"] / args.graphs
        tabucol.append({"iters": cp, "ms_per_graph": ms_per_graph, "mean_conflicts": mean_conf})
        print(f"  iters={cp:6d}  {ms_per_graph:8.2f} ms/graph  conflicts={mean_conf:.2f}")

    with open(args.out_json, "w") as f:
        json.dump({"neural_ls": neural, "tabucol": tabucol,
                    "n": args.n, "graphs": args.graphs, "restarts": args.restarts}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4.5))
    plt.plot([r["ms_per_graph"] for r in neural], [r["mean_conflicts"] for r in neural],
              "o-", label="Neural-LS (ours)")
    plt.plot([r["ms_per_graph"] for r in tabucol], [r["mean_conflicts"] for r in tabucol],
              "s-", label="DSATUR+TabuCol")
    plt.xscale("log")
    plt.xlabel("wall-clock time per graph (ms, log scale)")
    plt.ylabel("mean conflicts")
    plt.title(f"Quality vs. budget (N={args.n}, k={k}, p={args.p})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"\nSaved curve data -> {args.out_json}\nSaved plot -> {args.out_png}")


if __name__ == "__main__":
    main()
