"""Driver for the Section 16 MVP go/no-go comparison:
FedAvg, FedProx, Krum, Trimmed Mean, TACT x 3 seeds, 10 agents (8 honest / 2
malicious), alpha=0.5, model poisoning, 50 rounds.

Each method+seed runs as its own subprocess writing to its own output file.
A crash in one never affects the others, and any run already complete on
disk is skipped, so this script is safe to re-invoke after a partial failure
or interruption without losing finished work.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RESULTS_DIR = os.path.join(HERE, "..", "results", "mvp")

METHODS = ["fedavg", "fedprox", "krum", "trimmedmean", "tact"]


def out_path(method, seed):
    return os.path.join(RESULTS_DIR, f"{method}_seed{seed}.csv")


def is_complete(path, rounds):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        n = sum(1 for _ in f)
    return n >= rounds + 1


def run_one(method, seed, common_args):
    path = out_path(method, seed)
    if is_complete(path, common_args["rounds"]):
        print(f"[sweep] SKIP {method} seed={seed} (already complete)")
        return True

    if method == "tact":
        cmd = [
            PY,
            os.path.join(HERE, "tact_prototype.py"),
            "--num_agents", str(common_args["num_agents"]),
            "--num_malicious", str(common_args["num_malicious"]),
            "--alpha", str(common_args["alpha"]),
            "--rounds", str(common_args["rounds"]),
            "--seed", str(seed),
            "--poison_lambda", str(common_args["poison_lambda"]),
            "--out", path,
        ]
    else:
        cmd = [
            PY,
            os.path.join(HERE, "baseline_run.py"),
            "--method", method,
            "--num_agents", str(common_args["num_agents"]),
            "--num_malicious", str(common_args["num_malicious"]),
            "--alpha", str(common_args["alpha"]),
            "--rounds", str(common_args["rounds"]),
            "--seed", str(seed),
            "--poison_lambda", str(common_args["poison_lambda"]),
            "--out", path,
        ]

    print(f"[sweep] RUN {method} seed={seed} -> {path}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, timeout=common_args["per_run_timeout"])
        ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[sweep] TIMEOUT {method} seed={seed} after {common_args['per_run_timeout']}s")
        ok = False
    except Exception as e:
        print(f"[sweep] ERROR {method} seed={seed}: {e}")
        ok = False
    elapsed = time.time() - t0
    print(f"[sweep] {'DONE' if ok else 'FAILED'} {method} seed={seed} in {elapsed/60:.1f} min")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_agents", type=int, default=10)
    ap.add_argument("--num_malicious", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--poison_lambda", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", type=str, nargs="+", default=METHODS)
    ap.add_argument("--per_run_timeout", type=int, default=3 * 3600)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    common_args = {
        "num_agents": args.num_agents,
        "num_malicious": args.num_malicious,
        "alpha": args.alpha,
        "rounds": args.rounds,
        "poison_lambda": args.poison_lambda,
        "per_run_timeout": args.per_run_timeout,
    }

    summary_path = os.path.join(RESULTS_DIR, "sweep_summary.csv")
    with open(summary_path, "a", newline="") as sf:
        summary_writer = csv.writer(sf)
        for method in args.methods:
            for seed in args.seeds:
                ok = run_one(method, seed, common_args)
                summary_writer.writerow([method, seed, "ok" if ok else "failed", time.strftime("%Y-%m-%d %H:%M:%S")])
                sf.flush()

    print("[sweep] All runs attempted. See", summary_path)


if __name__ == "__main__":
    main()
