"""Run the trained Neural-LS policy on a single input graph and return the
actual node colouring, not just its cost (`run_agent` in neural_coloring.py
only tracks best conflict count for benchmarking, so this wraps the same
rollout and also keeps track of which colouring produced it)."""
import argparse
import numpy as np
import torch

from neural_coloring import GNNPolicy, ColoringEnv, masked_categorical


@torch.no_grad()
def solve(policy, A, k, horizon, restarts=16, greedy=False, x0=None, device="cpu"):
    """Same multi-restart rollout as `run_agent`, but returns (colouring, cost)."""
    G, N = A.shape[0], A.shape[1]
    At = A.repeat(restarts, 1, 1)                       # (restarts*G, N, N)
    env = ColoringEnv(At, k, horizon)
    B = At.shape[0]
    if x0 is None:
        x = torch.randint(0, k, (B, N), device=device)
    else:
        x = x0.repeat(restarts, 1)
    env.reset(x)
    best_conf = env.conf.clone()
    best_x = env.x.clone()
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(horizon):
        logits, _ = policy(env.A, env.x, env.recency)
        dist = masked_categorical(logits, env.action_mask())
        action = dist.probs.argmax(-1) if greedy else dist.sample()
        _, step_done, _ = env.step(action, ~done)
        improved = env.conf < best_conf
        best_conf = torch.where(improved, env.conf, best_conf)
        best_x = torch.where(improved.unsqueeze(-1), env.x, best_x)
        done = done | step_done
        if done.all():
            break

    best_conf = best_conf.reshape(restarts, G)          # (restarts, G)
    best_x = best_x.reshape(restarts, G, N)              # (restarts, G, N)
    conf_out, idx = best_conf.min(dim=0)                 # (G,), (G,)
    x_out = best_x[idx, torch.arange(G, device=device)]  # (G, N)
    return x_out, conf_out


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    policy = GNNPolicy(ck["k"], hidden=ck["hidden"], layers=ck["layers"]).to(device)
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    return policy, ck["k"]


def adjacency_from_edges(edges, n):
    A = np.zeros((n, n), dtype=np.float32)
    for u, v, *w in edges:
        weight = float(w[0]) if w else 1.0
        A[u, v] = weight
        A[v, u] = weight
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/neural_color_continued2.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--edges", default=None,
                     help="path to edge-list file: 'u v' or 'u v weight' per line, 0-indexed nodes")
    ap.add_argument("--n", type=int, default=None,
                     help="number of nodes; inferred from --edges if omitted")
    ap.add_argument("--random_n", type=int, default=None,
                     help="generate a random Erdos-Renyi graph instead of reading --edges")
    ap.add_argument("--random_p", type=float, default=0.3)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling (deterministic)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    policy, k = load_model(args.model, args.device)

    if args.edges:
        edges = []
        with open(args.edges) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                edges.append((int(parts[0]), int(parts[1])) + tuple(parts[2:]))
        n = args.n or (max(max(u, v) for u, v, *_ in edges) + 1)
        A_np = adjacency_from_edges(edges, n)
    elif args.random_n:
        import networkx as nx
        rng = np.random.default_rng(args.seed)
        G = nx.erdos_renyi_graph(args.random_n, args.random_p, seed=int(rng.integers(0, 2**31 - 1)))
        A_np = nx.to_numpy_array(G, dtype=np.float32)
    else:
        raise SystemExit("Provide either --edges <file> or --random_n <N>")

    torch.manual_seed(args.seed)
    A = torch.tensor(A_np, device=args.device).unsqueeze(0)   # (1, N, N)
    x_out, conf_out = solve(policy, A, k, args.horizon, restarts=args.restarts,
                             greedy=args.greedy, device=args.device)
    colours = x_out[0].cpu().numpy()

    print(f"Nodes: {len(colours)}   Colours available (k={k})   Colours used: {len(set(colours.tolist()))}")
    print(f"Conflicts (total weight of monochromatic edges): {conf_out[0].item():.3f}")
    print("Colouring (node: colour):")
    for i, c in enumerate(colours):
        print(f"  {i}: {c}")


if __name__ == "__main__":
    main()
