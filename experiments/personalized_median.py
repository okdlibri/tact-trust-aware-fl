"""Personalized control #2 (requested in paper/feedback.txt): same decentralized
architecture as TACT (every agent keeps and evolves its own model, same local
training / broadcast / Agent code path, same malicious-agent poisoning logic),
but the trust mechanism is replaced by a generic, memoryless, no-state robust
statistic: coordinate-wise median over all broadcast states. This isolates
whether TACT's gain over single-global-model baselines (Table 1) comes from
the trust mechanism specifically, or merely from the personalized/decentralized
architecture plus *any* robust aggregation rule.

Note on "personalized": because every honest agent observes the identical
broadcast set {theta_1, ..., theta_n} and applies the same permutation-symmetric
rule (coordinate median) with no private state, every honest agent's aggregate
is mathematically identical each round -- there is no genuine per-agent
divergence, unlike TACT (whose T_ij depends on each agent's own private
validation split and update history). We compute the median once per round and
assign it to every honest agent accordingly, for efficiency; this is a
transparent, expected property of using a symmetric rule here, not a bug.
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
from attacks.alie import alie_attack
from attacks.poisoning import poison_update, poison_update_bounded
from baselines.median import median_aggregate
from data.cifar10 import load_cifar10
from data.partition import dirichlet_partition
from models.resnet9 import ResNet9


def is_finite_state(state):
    return all(torch.isfinite(v).all() for v in state.values() if torch.is_floating_point(v))


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--attack_type", type=str, default="reversal", choices=["reversal", "alie"])
    ap.add_argument("--alie_z", type=float, default=1.0)
    ap.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "personalized_median.csv"))
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

    malicious_ids = set(range(args.num_malicious))
    honest_ids = [i for i in range(args.num_agents) if i not in malicious_ids]
    print(f"Malicious agents: {sorted(malicious_ids)} | Honest agents: {honest_ids}")

    model_fn = lambda: ResNet9(num_classes=10)
    agents = [
        Agent(
            i, train_set, client_idxs[i], model_fn, device,
            batch_size=args.batch_size, lr=args.lr,
            is_malicious=(i in malicious_ids), seed=args.seed,
        )
        for i in range(args.num_agents)
    ]
    shared_init = agents[0].get_state()
    for agent in agents[1:]:
        agent.set_state(shared_init)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "mean_honest_test_acc", "round_time_s"])

        for rnd in range(1, args.rounds + 1):
            t0 = time.time()

            prev_states = {i: agents[i].get_state() for i in range(args.num_agents)}
            for agent in agents:
                agent.local_train(epochs=args.local_epochs)

            broadcast_states = {}
            if args.attack_type == "alie" and malicious_ids:
                # Statistics over ALL broadcast states (no oracle on who is honest) --
                # see the matching note in tact_prototype.py.
                all_trained = [agents[i].get_state() for i in range(args.num_agents)]
                alie_state = alie_attack(all_trained, z=args.alie_z)
                for i in malicious_ids:
                    agents[i].set_state(alie_state)
                    broadcast_states[i] = alie_state
                for i in honest_ids:
                    broadcast_states[i] = agents[i].get_state()
            else:
                for i in range(args.num_agents):
                    trained_state = agents[i].get_state()
                    if i in malicious_ids:
                        if args.poison_bounded:
                            trained_state = poison_update_bounded(
                                prev_states[i], trained_state, lam=args.poison_lambda, max_norm=args.poison_max_norm
                            )
                        else:
                            trained_state = poison_update(prev_states[i], trained_state, lam=args.poison_lambda)
                        agents[i].set_state(trained_state)
                    broadcast_states[i] = trained_state

            # Coordinate-wise median over ALL n broadcast states, no trust/history,
            # no self-weighting -- identical aggregate for every honest agent.
            # Same basic numerical hygiene as every other aggregator in this codebase
            # (baseline_run.py's is_finite_state, TACT's hard safety rail): a non-finite
            # state must never enter the statistic, or it can propagate and corrupt the
            # entire shared consensus irrecoverably (this is not optional defense against
            # a clever attack -- it is the same invariant any aggregator needs regardless
            # of sophistication, see Section TACT/judge_peer discussion in the paper).
            valid_states = [broadcast_states[i] for i in range(args.num_agents) if is_finite_state(broadcast_states[i])]
            consensus_state = median_aggregate(valid_states)
            for i in honest_ids:
                agents[i].set_state(consensus_state)

            accs = [agents[i].evaluate(test_loader) for i in honest_ids]
            mean_acc = sum(accs) / len(accs)
            elapsed = time.time() - t0

            print(f"Round {rnd:3d}/{args.rounds} | honest_acc {mean_acc:.4f} | {elapsed:.1f}s")
            writer.writerow([rnd, mean_acc, elapsed])
            f.flush()

    print("Done. Results ->", args.out)


if __name__ == "__main__":
    main()
