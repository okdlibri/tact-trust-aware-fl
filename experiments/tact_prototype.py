"""Milestone: TACT prototype (decentralized, personalized, trust-weighted).

Each honest agent keeps its OWN model. Every round it judges every peer
(honest or malicious) using its own held-out local data, updates trust,
decides accept/verify/downweight/reject, and folds peers into its own model
accordingly. Malicious agents apply a model-poisoning attack
(attacks/poisoning.py) and do not bother aggregating peers (their own
utility is irrelevant to the attack).
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
from collaboration.policy import decide, weighted_aggregate
from data.cifar10 import load_cifar10
from data.partition import dirichlet_partition
from models.resnet9 import ResNet9
from trust.trust_estimator import update_trust


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
    ap.add_argument(
        "--poison_bounded",
        action="store_true",
        help="Use the norm-bounded poisoning attack (realistic/stealthy magnitude, never "
        "numerically explodes) instead of the plain unbounded reversal.",
    )
    ap.add_argument("--poison_max_norm", type=float, default=800.0)
    ap.add_argument(
        "--attack_type",
        type=str,
        default="reversal",
        choices=["reversal", "alie"],
        help="'reversal' (default, unchanged): poison_update/poison_update_bounded as before. "
        "'alie': the 'a little is enough' attack (Baruch et al. 2019) -- overrides --poison_bounded.",
    )
    ap.add_argument("--alie_z", type=float, default=1.0)
    ap.add_argument("--trust_momentum", type=float, default=0.8)
    ap.add_argument("--trust_beta", type=float, default=1.0)
    ap.add_argument("--trust_gamma", type=float, default=0.7)
    ap.add_argument(
        "--trust_temperature",
        type=float,
        default=3.0,
        help="Softens peer confidence before judging (classifiers are overconfident even when "
        "wrong on another agent's non-IID data, which would otherwise misread honest "
        "heterogeneity as malicious). Does not affect the malicious-agent detection, which is "
        "caught by a separate hard NaN/Inf safety rail regardless of temperature.",
    )
    ap.add_argument("--no_uncertainty", action="store_true", help="TACT-no-UQ ablation")
    ap.add_argument(
        "--no_adaptive_policy",
        action="store_true",
        help="TACT-no-adaptation ablation: use continuous weight=trust directly instead of "
        "the discrete accept/verify/downweight/reject action tiers.",
    )
    ap.add_argument(
        "--no_trust",
        action="store_true",
        help="TACT-no-trust ablation: ignore trust for weighting (uniform weight=1 for every "
        "numerically valid peer); trust is still computed/logged but has no effect.",
    )
    ap.add_argument(
        "--attack_onset",
        type=int,
        default=1,
        help="Round at which malicious agents start poisoning (default 1 = from the start). "
        "Rounds before onset, malicious agents train and broadcast honestly, letting trust "
        "build up normally -- tests whether TACT detects a behavior CHANGE, not just a "
        "persistently bad agent.",
    )
    ap.add_argument(
        "--warmup_rounds",
        type=int,
        default=3,
        help="Rounds of coordinate-median (Byzantine-robust) bootstrap aggregation before "
        "trust starts gating weights. Right after common init, models are near-random and "
        "produce degenerate, uninformative behavioral evidence; a naive uniform-weight "
        "warm-up is itself not robust to the attacker, so median is used instead purely to "
        "reach a usable common baseline before trust-gated personalization takes over.",
    )
    ap.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "tact_prototype.csv"))
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
    # Weight-space averaging is only meaningful if agents start from a common
    # initialization (independent random inits are not permutation-aligned,
    # so blending them collapses to noise). All agents share init weights,
    # matching standard FedAvg practice, then diverge only via training.
    shared_init = agents[0].get_state()
    for agent in agents[1:]:
        agent.set_state(shared_init)

    # trust[i][j]: how much honest agent i trusts agent j. Self-trust is
    # always 1.0 and not stored. Initialized neutral (0.5).
    trust = {i: {j: 0.5 for j in range(args.num_agents) if j != i} for i in honest_ids}

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["round", "mean_honest_test_acc", "trust_honest", "trust_malicious", "false_isolation_rate", "round_time_s"]
        )

        for rnd in range(1, args.rounds + 1):
            t0 = time.time()

            prev_states = {i: agents[i].get_state() for i in range(args.num_agents)}
            for agent in agents:
                agent.local_train(epochs=args.local_epochs)

            broadcast_states = {}
            if args.attack_type == "alie" and rnd >= args.attack_onset and malicious_ids:
                # Statistics computed over ALL broadcast states (not the ground-truth
                # honest subset) -- the attacker observes every public broadcast under
                # full connectivity, but does not get an oracle telling it who is honest.
                all_trained = [agents[i].get_state() for i in range(args.num_agents)]
                alie_state = alie_attack(all_trained, z=args.alie_z)
                for i in malicious_ids:
                    agents[i].set_state(alie_state)  # all colluding attackers broadcast the identical crafted state
                    broadcast_states[i] = alie_state
                for i in honest_ids:
                    broadcast_states[i] = agents[i].get_state()
            else:
                for i in range(args.num_agents):
                    trained_state = agents[i].get_state()
                    if i in malicious_ids and rnd >= args.attack_onset:
                        if args.poison_bounded:
                            trained_state = poison_update_bounded(
                                prev_states[i], trained_state, lam=args.poison_lambda, max_norm=args.poison_max_norm
                            )
                        else:
                            trained_state = poison_update(prev_states[i], trained_state, lam=args.poison_lambda)
                        agents[i].set_state(trained_state)  # attacker also adopts the poisoned state
                    broadcast_states[i] = trained_state

            trust_honest_vals, trust_malicious_vals = [], []
            false_isolation_rate = float("nan")
            if rnd <= args.warmup_rounds:
                # Byzantine-robust bootstrap: same median consensus state for everyone.
                boot_state = median_aggregate([broadcast_states[i] for i in range(args.num_agents)])
                for i in range(args.num_agents):
                    agents[i].set_state(boot_state)
                for i in honest_ids:
                    for j in range(args.num_agents):
                        if j == i:
                            continue
                        (trust_honest_vals if j in honest_ids else trust_malicious_vals).append(trust[i][j])
            else:
                honest_reject_count, honest_pair_count = 0, 0
                for i in honest_ids:
                    peer_states_and_weights = []
                    for j in range(args.num_agents):
                        if j == i:
                            continue
                        evidence = agents[i].judge_peer(broadcast_states[j], temperature=args.trust_temperature)
                        t_ij = update_trust(
                            trust[i][j],
                            evidence,
                            momentum=args.trust_momentum,
                            beta=args.trust_beta,
                            gamma=args.trust_gamma,
                            use_uncertainty=not args.no_uncertainty,
                        )
                        trust[i][j] = t_ij
                        if not evidence.get("valid", True):
                            w_ij = 0.0  # hard safety rail: never blend a non-finite peer state
                            action = "reject"
                        elif args.no_trust:
                            w_ij, action = 1.0, "accept"
                        elif args.no_adaptive_policy:
                            w_ij, action = max(0.0, min(1.0, t_ij)), "accept"
                        else:
                            action, w_ij = decide(t_ij)
                        peer_states_and_weights.append((broadcast_states[j], w_ij))
                        (trust_honest_vals if j in honest_ids else trust_malicious_vals).append(t_ij)
                        if j in honest_ids:
                            honest_pair_count += 1
                            if action == "reject":
                                honest_reject_count += 1

                    new_state = weighted_aggregate(broadcast_states[i], peer_states_and_weights)
                    agents[i].set_state(new_state)
                false_isolation_rate = honest_reject_count / honest_pair_count if honest_pair_count else float("nan")

            accs = [agents[i].evaluate(test_loader) for i in honest_ids]
            mean_acc = sum(accs) / len(accs)
            mean_trust_honest = sum(trust_honest_vals) / len(trust_honest_vals) if trust_honest_vals else float("nan")
            mean_trust_malicious = (
                sum(trust_malicious_vals) / len(trust_malicious_vals) if trust_malicious_vals else float("nan")
            )
            elapsed = time.time() - t0

            print(
                f"Round {rnd:3d}/{args.rounds} | honest_acc {mean_acc:.4f} | "
                f"trust_honest {mean_trust_honest:.3f} | trust_malicious {mean_trust_malicious:.3f} | "
                f"false_iso {false_isolation_rate:.3f} | {elapsed:.1f}s"
            )
            writer.writerow([rnd, mean_acc, mean_trust_honest, mean_trust_malicious, false_isolation_rate, elapsed])
            f.flush()

    print("Done. Results ->", args.out)


if __name__ == "__main__":
    main()
