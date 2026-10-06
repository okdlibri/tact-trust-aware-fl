"""Isolates what the soft trust mechanism contributes versus the hard
NaN-safety rail: TACT-full vs TACT-no-trust, both under the norm-bounded
poisoning attack (which never produces NaN, unlike the plain unbounded
attack used in the MVP/ablation runs), same config as the ablation study
(10 agents, 2/10 malicious, alpha=0.5, poisoning from round 1).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RESULTS_DIR = os.path.join(HERE, "..", "results", "bounded")

COMMON = ["--num_agents", "10", "--num_malicious", "2", "--alpha", "0.5", "--rounds", "50", "--seed", "0", "--poison_bounded"]

CONFIGS = [
    {"name": "bounded_tact_full", "args": COMMON},
    {"name": "bounded_tact_no_trust", "args": COMMON + ["--no_trust"]},
]


def is_complete(path, rounds=50):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        n = sum(1 for _ in f)
    return n >= rounds + 1


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "bounded_summary.csv")

    for cfg in CONFIGS:
        out_path = os.path.join(RESULTS_DIR, f"{cfg['name']}.csv")
        if is_complete(out_path):
            print(f"[bounded] SKIP {cfg['name']} (already complete)")
            continue
        cmd = [PY, os.path.join(HERE, "tact_prototype.py")] + cfg["args"] + ["--out", out_path]
        print(f"[bounded] RUN {cfg['name']} -> {out_path}")
        t0 = time.time()
        try:
            result = subprocess.run(cmd, timeout=3 * 3600)
            ok = result.returncode == 0
        except Exception as e:
            print(f"[bounded] ERROR {cfg['name']}: {e}")
            ok = False
        elapsed = time.time() - t0
        print(f"[bounded] {'DONE' if ok else 'FAILED'} {cfg['name']} in {elapsed/60:.1f} min")
        with open(summary_path, "a") as sf:
            sf.write(f"{cfg['name']},{'ok' if ok else 'failed'},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("[bounded] All configs attempted.")


if __name__ == "__main__":
    main()
