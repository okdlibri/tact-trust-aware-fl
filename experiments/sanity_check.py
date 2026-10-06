"""Milestone: basic multi-agent environment + FedAvg baseline (no trust, no attacks yet).

Loop per Section 40 of the execution plan (local_train -> aggregate -> evaluate),
minus the trust/collaboration steps which are added next.
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
from data.cifar10 import load_cifar10
from data.partition import dirichlet_partition
from models.resnet9 import ResNet9


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


def fedavg_aggregate(global_state, deltas_and_sizes, device):
    total = sum(n for _, n in deltas_and_sizes)
    new_state = {k: v.clone().to(device) for k, v in global_state.items()}
    for k in new_state:
        if not torch.is_floating_point(new_state[k]):
            # Non-float buffers (e.g. BatchNorm's num_batches_tracked) aren't
            # gradient-like quantities to average; leave them as-is.
            continue
        for delta, n in deltas_and_sizes:
            w = n / total
            new_state[k] += w * delta[k]
    return new_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_agents", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "sanity_check.csv"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    print(f"Device: {device}")

    train_set, test_set = load_cifar10(root=args.data_root)
    labels = np.array(train_set.targets)
    client_idxs = dirichlet_partition(labels, args.num_agents, args.alpha, seed=args.seed)
    print("Client shard sizes:", [len(x) for x in client_idxs])

    model_fn = lambda: ResNet9(num_classes=10)
    global_model = model_fn().to(device)
    global_state = global_model.state_dict()

    agents = [
        Agent(i, train_set, client_idxs[i], model_fn, device, batch_size=args.batch_size, lr=args.lr)
        for i in range(args.num_agents)
    ]

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "train_loss", "test_acc", "round_time_s"])

        for rnd in range(1, args.rounds + 1):
            t0 = time.time()
            deltas_and_sizes = []
            losses = []
            for agent in agents:
                agent.load_global(global_state)
                loss = agent.local_train(epochs=args.local_epochs)
                losses.append(loss)
                delta, n = agent.get_delta(global_state)
                deltas_and_sizes.append((delta, n))

            global_state = fedavg_aggregate(global_state, deltas_and_sizes, device)
            global_model.load_state_dict(global_state)
            test_acc = eval_model(global_model, test_loader, device)
            elapsed = time.time() - t0
            avg_loss = sum(losses) / len(losses)
            print(f"Round {rnd:3d}/{args.rounds} | loss {avg_loss:.4f} | test_acc {test_acc:.4f} | {elapsed:.1f}s")
            writer.writerow([rnd, avg_loss, test_acc, elapsed])
            f.flush()

    print("Done. Results ->", args.out)


if __name__ == "__main__":
    main()
