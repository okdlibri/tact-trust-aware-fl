"""Re-run the UQ/adaptation/trust ablations against the SELECTED base
configuration (momentum=0.3) instead of the original default (momentum=0.8),
for consistency with the main evaluation table. The old momentum=0.8
ablation files are kept as-is (reference point, not overwritten) -- these
write to new, clearly-named files.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RESULTS_DIR = os.path.join(HERE, "..", "results", "extra")

BASE = "--num_agents 10 --num_malicious 2 --alpha 0.5 --rounds 50 --seed 0 --trust_momentum 0.3"

CONFIGS = {
    "ablation_no_uq_m0.3": "--no_uncertainty",
    "ablation_no_adaptation_m0.3": "--no_adaptive_policy",
    "ablation_no_trust_m0.3": "--no_trust",
}


def is_complete(path, rounds=50):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for _ in f) >= rounds + 1


def main():
    for name, flag in CONFIGS.items():
        out = os.path.join(RESULTS_DIR, f"{name}.csv")
        if is_complete(out):
            print(f"SKIP {name} (already complete)")
            continue
        cmd = [PY, os.path.join(HERE, "tact_prototype.py")] + f"{BASE} {flag} --out {out}".split()
        print(f"RUN {name}: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd)
        print(f"{'DONE' if result.returncode == 0 else 'FAILED'} {name} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
