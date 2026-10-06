"""Single-global-model baselines for the main go/no-go comparison:
FedAvg, FedProx, Krum, Trimmed Mean. Malicious agents apply the same model
poisoning attack used against TACT (attacks/poisoning.py), for a fair
comparison. One method + one seed per invocation, writing to its own output
file -- a crash here never affects other method/seed runs.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent
from attacks.poisoning import poison_update, poison_update_bounded
from baselines.fltrust import carve_root_set, fltrust_aggregate
from baselines.krum import krum_select
from baselines.trimmed_mean import trimmed_mean_aggregate
from data.cifar10 import load_cifar10
from data.partition import dirichlet_partition
from models.resnet9 import ResNet9


def is_finite_state(state):
    return all(torch.isfinite(v).all() for v in state.values() if torch.is_floating_point(v))


def fedavg_full_average(states, weights):
    total = sum(weights)
    out = {}
    for k in states[0]:
        if not torch.is_floating_point(states[0][k]):
            out[k] = states[0][k]
            continue
        acc = sum(states[i][k] * weights[i] for i in range(len(states))) / total
        out[k] = acc
    return out


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fedavg", "fedprox", "krum", "trimmedmean", "fltrust"], required=True)
    ap.add_argument("--num_agents", type=int, default=10)
    ap.add_argument("--num_malicious", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poison_lambda", type=float, default=1.0)
    ap.add_argument("--poison_bounded", action="store_true")
    ap.add_argument("--poison_max_norm", type=float, default=800.0)
    ap.add_argument("--fedprox_mu", type=float, default=0.01)
    ap.add_argument("--fltrust_root_size", type=int, default=250, help="Size of the server's held-out clean root set.")
    ap.add_argument(
        "--fltrust_server_epochs",
        type=int,
        default=5,
        help="Local epochs for the server's own root-set training. Must be "
        "larger than client local_epochs: with only 1 epoch over a root set "
        "this small (~3 SGD steps), the server's reference update direction "
        "is an unreliable, high-variance estimate -- empirically confirmed "
        "to produce NEGATIVE cosine similarity with every client, including "
        "honest ones (verified directly: 1 epoch gives cos in [-0.58,-0.49] "
        "for all clients; 3+ epochs gives a stable, consistent [+0.55,+0.70]).",
    )
    ap.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="Re-run even if --out already has a complete result")
    args = ap.parse_args()

    if not args.force and os.path.exists(args.out):
        with open(args.out) as f:
            rows = f.readlines()
        if len(rows) >= args.rounds + 1:  # header + all rounds
            print(f"SKIP (already complete): {args.out}")
            return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    print(f"[{args.method} seed={args.seed}] device={device}")

    train_set, test_set = load_cifar10(root=args.data_root)
    labels = np.array(train_set.targets)

    server_agent = None
    if args.method == "fltrust":
        # Hold out a small, class-balanced root set BEFORE client
        # partitioning, so it never overlaps with any client's data (or the
        # test set). Client partitioning then proceeds identically to every
        # other baseline, just over the slightly reduced remaining pool --
        # dirichlet_partition returns positions within whatever label array
        # it's given, so remaining_indices maps those back to the original
        # dataset's indices.
        root_indices, remaining_indices = carve_root_set(labels, args.fltrust_root_size, seed=args.seed)
        remaining_labels = labels[remaining_indices]
        rel_client_idxs = dirichlet_partition(remaining_labels, args.num_agents, args.alpha, seed=args.seed)
        client_idxs = [remaining_indices[rel].tolist() for rel in rel_client_idxs]
        print(f"[fltrust] root set size={len(root_indices)}, client pool reduced by same amount")
    else:
        client_idxs = dirichlet_partition(labels, args.num_agents, args.alpha, seed=args.seed)

    malicious_ids = set(range(args.num_malicious))

    model_fn = lambda: ResNet9(num_classes=10)
    agents = [
        Agent(
            i,
            train_set,
            client_idxs[i],
            model_fn,
            device,
            batch_size=args.batch_size,
            lr=args.lr,
            is_malicious=(i in malicious_ids),
            seed=args.seed,
        )
        for i in range(args.num_agents)
    ]
    global_model = model_fn().to(device)
    global_state = agents[0].get_state()  # shared init, same rationale as tact_prototype.py
    for a in agents:
        a.set_state(global_state)
    global_model.load_state_dict(global_state)

    if args.method == "fltrust":
        # The server is just another Agent instance (reusing all existing,
        # already-tested local-training machinery), trained each round on
        # its own held-out root set instead of any client's data.
        server_agent = Agent(
            -1, train_set, root_indices.tolist(), model_fn, device,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed, val_frac=0.0,
        )
        server_agent.set_state(global_state)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "test_acc", "trust_honest", "trust_malicious", "round_time_s"])

        for rnd in range(1, args.rounds + 1):
            t0 = time.time()
            prev_state = global_state

            states, sizes = [], []
            for i, agent in enumerate(agents):
                agent.set_state(global_state)
                if args.method == "fedprox" and i not in malicious_ids:
                    agent.local_train(epochs=args.local_epochs, prox_mu=args.fedprox_mu, global_state=global_state)
                else:
                    agent.local_train(epochs=args.local_epochs)
                st = agent.get_state()
                if i in malicious_ids:
                    if args.poison_bounded:
                        st = poison_update_bounded(prev_state, st, lam=args.poison_lambda, max_norm=args.poison_max_norm)
                    else:
                        st = poison_update(prev_state, st, lam=args.poison_lambda)
                states.append(st)
                sizes.append(len(agent.indices))

            trust_honest, trust_malicious = float("nan"), float("nan")

            if args.method == "fltrust":
                server_agent.set_state(global_state)
                server_agent.local_train(epochs=args.fltrust_server_epochs)
                server_state = server_agent.get_state()
                global_state, trust_scores, _fallback_used = fltrust_aggregate(prev_state, server_state, states)
                honest_ts = [trust_scores[i] for i in range(args.num_agents) if i not in malicious_ids]
                malicious_ts = [trust_scores[i] for i in range(args.num_agents) if i in malicious_ids]
                trust_honest = sum(honest_ts) / len(honest_ts) if honest_ts else float("nan")
                trust_malicious = sum(malicious_ts) / len(malicious_ts) if malicious_ts else float("nan")
            else:
                # Basic input hygiene: a client submission with non-finite values
                # is unusable by construction (e.g. lam=1.0 poisoning can numerically
                # explode an already well-trained update). Any real aggregator would
                # reject this outright; naive averaging/Krum have no other protection
                # against it (unlike trimmed mean's built-in tolerance for outliers).
                finite_mask = [is_finite_state(s) for s in states]
                if not all(finite_mask):
                    states = [s for s, ok in zip(states, finite_mask) if ok]
                    sizes = [w for w, ok in zip(sizes, finite_mask) if ok]

                if args.method in ("fedavg", "fedprox"):
                    global_state = fedavg_full_average(states, sizes)
                elif args.method == "krum":
                    global_state, _ = krum_select(states, args.num_malicious)
                elif args.method == "trimmedmean":
                    global_state = trimmed_mean_aggregate(states, trim_k=args.num_malicious)

            global_model.load_state_dict(global_state)
            test_acc = eval_model(global_model, test_loader, device)
            elapsed = time.time() - t0
            print(
                f"[{args.method} seed={args.seed}] round {rnd:3d}/{args.rounds} | acc {test_acc:.4f} | "
                f"trust_h {trust_honest:.3f} | trust_m {trust_malicious:.3f} | {elapsed:.1f}s"
            )
            writer.writerow([rnd, test_acc, trust_honest, trust_malicious, elapsed])
            f.flush()

    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
