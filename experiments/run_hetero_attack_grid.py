"""Compact heterogeneity x attack factorial:
alpha in {1.0, 0.5, 0.1} x attack in {none, bounded}, frozen
TACT configuration (momentum=0.3). Goal: establish whether TACT degrades
gracefully as heterogeneity increases, and whether adversarial behavior
becomes harder to distinguish from honest distributional variation as
heterogeneity increases -- a single coherent grid rather than separate,
harder-to-compare heterogeneity and attack experiments.

Reuses the already-valid alpha=10 zero-attack point from the original
heterogeneity sweep (results/extra/hetero_alpha10.csv) instead of
re-running it, since that config never involved a malicious agent and is
unaffected by anything since.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RESULTS_DIR = os.path.join(HERE, "..", "results", "hetero_attack_grid")

ALPHAS = [1.0, 0.5, 0.1]
COMMON = "--num_agents 10 --rounds 50 --seed 0 --trust_momentum 0.3"

CONFIGS = {}
for alpha in ALPHAS:
    name_none = f"alpha{alpha}_none"
    CONFIGS[name_none] = f"{COMMON} --alpha {alpha} --num_malicious 0"
    name_bounded = f"alpha{alpha}_bounded"
    CONFIGS[name_bounded] = f"{COMMON} --alpha {alpha} --num_malicious 2 --poison_bounded"


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
        cmd = [PY, os.path.join(HERE, "tact_prototype.py")] + f"{CONFIGS[name]} --out {out}".split()
        print(f"RUN {name}: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd)
        print(f"{'DONE' if result.returncode == 0 else 'FAILED'} {name} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["gpu0", "gpu1"], required=True)
    args = ap.parse_args()
    all_names = list(CONFIGS.keys())
    half = len(all_names) // 2
    group = all_names[:half] if args.group == "gpu0" else all_names[half:]
    run_group(group)
