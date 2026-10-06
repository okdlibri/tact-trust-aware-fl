"""Run one named group of configs from all_configs.py sequentially, each as
its own subprocess writing to its own file -- a crash in one never affects
the others, and already-complete runs are skipped (safe to re-invoke)."""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from all_configs import CONFIGS, GROUPS

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def is_complete(path, rounds=50):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        n = sum(1 for _ in f)
    return n >= rounds + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(GROUPS.keys()))
    args = ap.parse_args()

    names = GROUPS[args.group]
    summary_path = os.path.join(HERE, "..", "results", f"group_{args.group}_summary.csv")
    print(f"[{args.group}] {len(names)} configs: {names}")

    for name in names:
        cfg = CONFIGS[name]
        out = cfg["out"]
        if is_complete(out):
            print(f"[{args.group}] SKIP {name} (already complete)")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cmd = [PY, os.path.join(HERE, cfg["script"])] + cfg["args"].split()
        print(f"[{args.group}] RUN {name}: {' '.join(cmd)}")
        t0 = time.time()
        try:
            result = subprocess.run(cmd, timeout=3 * 3600)
            ok = result.returncode == 0
        except Exception as e:
            print(f"[{args.group}] ERROR {name}: {e}")
            ok = False
        elapsed = time.time() - t0
        print(f"[{args.group}] {'DONE' if ok else 'FAILED'} {name} in {elapsed/60:.1f} min")
        with open(summary_path, "a") as sf:
            sf.write(f"{name},{'ok' if ok else 'failed'},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"[{args.group}] All configs attempted.")


if __name__ == "__main__":
    main()
