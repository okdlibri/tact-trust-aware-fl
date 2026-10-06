"""feedback.txt item 1: the critical personalized-control sweep. Isolates
whether TACT's gain over single-global-model baselines (Table 1) comes from
the trust mechanism specifically, or from the decentralized/personalized
architecture plus *any* non-trust weighting rule.

Three controls, all sharing TACT's exact local-training/broadcast/averaging
code path, differing only in the peer weight:
  1. Personalized FedAvg (every valid peer weight 1, including adversaries)
     = tact_prototype.py --no_trust (already exists for seed 0 unbounded at
     momentum 0.8; here we add momentum=0.3 and seeds 1,2, both attacks, so
     it is directly comparable to Table 1).
  2. Personalized coordinate-median (no trust state at all) = new
     personalized_median.py.
  3. "TACT with trust computed but not acted on" is the same object as (1)
     here: trust is still computed/logged (judge_peer + update_trust run every
     round), it just never changes the aggregation weight.

Same seeds as Table 1 (0, 1, 2), alpha=0.5, 10 agents, 2 malicious, 50 rounds,
both the unbounded and norm-bounded attack.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RESULTS_DIR = os.path.join(HERE, "..", "results", "personalized_controls")

COMMON = "--num_agents 10 --num_malicious 2 --alpha 0.5 --rounds 50"
SEEDS = [0, 1, 2]

CONFIGS = {}
for seed in SEEDS:
    for attack_name, attack_flag in [("unbounded", ""), ("bounded", "--poison_bounded")]:
        name = f"personalized_median_{attack_name}_seed{seed}"
        CONFIGS[name] = {
            "script": "personalized_median.py",
            "args": f"{COMMON} --seed {seed} {attack_flag}".strip(),
        }
        name = f"personalized_fedavg_{attack_name}_seed{seed}"
        CONFIGS[name] = {
            "script": "tact_prototype.py",
            "args": f"{COMMON} --seed {seed} --trust_momentum 0.3 --no_trust {attack_flag}".strip(),
        }


def is_complete(path, rounds=50):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for _ in f) >= rounds + 1


def run_group(names):
    for name in names:
        out = os.path.join(RESULTS_DIR, f"{name}.csv")
        if is_complete(out):
            print(f"SKIP {name} (already complete)")
            continue
        os.makedirs(RESULTS_DIR, exist_ok=True)
        cfg = CONFIGS[name]
        cmd = [PY, os.path.join(HERE, cfg["script"])] + f"{cfg['args']} --out {out}".split()
        print(f"RUN {name}: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd)
        print(f"{'DONE' if result.returncode == 0 else 'FAILED'} {name} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=str, default="all", help="'all', or 'gpuN' with --num_groups")
    ap.add_argument("--num_groups", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    all_names = sorted(CONFIGS.keys())
    if args.list:
        for n in all_names:
            print(n)
        sys.exit(0)
    if args.group == "all":
        group = all_names
    else:
        idx = int(args.group.replace("gpu", ""))
        group = [n for i, n in enumerate(all_names) if i % args.num_groups == idx]
    run_group(group)
