"""Driver for the three follow-up experiments requested after the MVP go/no-go:

1. Heterogeneity sweep: num_malicious=0, alpha in {10,1,0.5,0.1}.
   Tests whether heterogeneity alone causes false isolation of honest agents.
2. Adaptive attacker: malicious agents behave honestly for 20 rounds, then
   start poisoning, under high heterogeneity (alpha=0.1). Tests detection delay.
3. Ablation study: TACT-no-UQ, TACT-no-history, TACT-no-adaptation,
   TACT-no-trust, same MVP config (10 agents, 8 honest/2 malicious, alpha=0.5,
   poisoning from round 1) as the already-collected TACT-full runs, for direct
   comparison.

Each config runs as its own subprocess writing to its own file -- a crash in
one never affects the others, and already-complete runs are skipped, so this
is safe to re-invoke after an interruption.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RESULTS_DIR = os.path.join(HERE, "..", "results", "extra")

COMMON = ["--num_agents", "10", "--rounds", "50"]

CONFIGS = []

# 1. Heterogeneity sweep: 0% malicious, varying alpha.
for alpha in [10, 1, 0.5, 0.1]:
    CONFIGS.append(
        {
            "name": f"hetero_alpha{alpha}",
            "args": COMMON + ["--num_malicious", "0", "--alpha", str(alpha), "--seed", "0"],
        }
    )

# 2. Adaptive attacker: high heterogeneity + delayed-onset poisoning.
# 20% malicious, honest for 20 rounds, attack from round 21.
CONFIGS.append(
    {
        "name": "adaptive_alpha0.1_mal20pct_onset21",
        "args": COMMON
        + ["--num_malicious", "2", "--alpha", "0.1", "--attack_onset", "21", "--seed", "0"],
    }
)

# 3. Ablations: same MVP config (alpha=0.5, 2/10 malicious, poisoning from round 1).
ABLATION_BASE = COMMON + ["--num_malicious", "2", "--alpha", "0.5", "--seed", "0"]
CONFIGS += [
    {"name": "ablation_no_uq", "args": ABLATION_BASE + ["--no_uncertainty"]},
    {"name": "ablation_no_history", "args": ABLATION_BASE + ["--trust_momentum", "0"]},
    {"name": "ablation_no_adaptation", "args": ABLATION_BASE + ["--no_adaptive_policy"]},
    {"name": "ablation_no_trust", "args": ABLATION_BASE + ["--no_trust"]},
]


def is_complete(path, rounds=50):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        n = sum(1 for _ in f)
    return n >= rounds + 1


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "extra_summary.csv")

    for cfg in CONFIGS:
        out_path = os.path.join(RESULTS_DIR, f"{cfg['name']}.csv")
        if is_complete(out_path):
            print(f"[extra] SKIP {cfg['name']} (already complete)")
            continue

        cmd = [PY, os.path.join(HERE, "tact_prototype.py")] + cfg["args"] + ["--out", out_path]
        print(f"[extra] RUN {cfg['name']} -> {out_path}")
        print(f"[extra] cmd: {' '.join(cmd)}")
        t0 = time.time()
        try:
            result = subprocess.run(cmd, timeout=3 * 3600)
            ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"[extra] TIMEOUT {cfg['name']}")
            ok = False
        except Exception as e:
            print(f"[extra] ERROR {cfg['name']}: {e}")
            ok = False
        elapsed = time.time() - t0
        print(f"[extra] {'DONE' if ok else 'FAILED'} {cfg['name']} in {elapsed/60:.1f} min")
        with open(summary_path, "a") as sf:
            sf.write(f"{cfg['name']},{'ok' if ok else 'failed'},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("[extra] All configs attempted. See", summary_path)


if __name__ == "__main__":
    main()
