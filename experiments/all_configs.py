"""Single source of truth for every run that needs (re-)doing after the
BatchNorm-running-stat poisoning bug fix (2026-10-02). Each entry maps a
unique name to a full CLI invocation of either tact_prototype.py or
baseline_run.py. Used by run_group.py to execute a named subset (so work can
be split across multiple machines/GPUs) with the same per-config isolation
and skip-if-complete behavior as the original sweep scripts.
"""
import os

RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

MVP_COMMON = "--num_agents 10 --num_malicious 2 --alpha 0.5 --rounds 50"
ABLATION_COMMON = MVP_COMMON + " --seed 0"

CONFIGS = {}

# --- MVP go/no-go sweep: 5 methods x 3 seeds ---
for seed in [0, 1, 2]:
    for method in ["fedavg", "fedprox", "krum", "trimmedmean"]:
        name = f"mvp_{method}_seed{seed}"
        out = os.path.join(RESULTS_ROOT, "mvp", f"{method}_seed{seed}.csv")
        CONFIGS[name] = {
            "script": "baseline_run.py",
            "args": f"--method {method} {MVP_COMMON} --seed {seed} --out {out}",
            "out": out,
        }
    name = f"mvp_tact_seed{seed}"
    out = os.path.join(RESULTS_ROOT, "mvp", f"tact_seed{seed}.csv")
    CONFIGS[name] = {"script": "tact_prototype.py", "args": f"{MVP_COMMON} --seed {seed} --out {out}", "out": out}

# --- Adaptive attacker: high heterogeneity, 20% malicious, onset round 21 ---
out = os.path.join(RESULTS_ROOT, "extra", "adaptive_alpha0.1_mal20pct_onset21.csv")
CONFIGS["adaptive_alpha0.1"] = {
    "script": "tact_prototype.py",
    "args": f"--num_agents 10 --num_malicious 2 --alpha 0.1 --rounds 50 --attack_onset 21 --seed 0 --out {out}",
    "out": out,
}

# --- Ablations: same config as MVP TACT ---
ABLATIONS = {
    "ablation_no_uq": "--no_uncertainty",
    "ablation_no_history": "--trust_momentum 0",
    "ablation_no_adaptation": "--no_adaptive_policy",
    "ablation_no_trust": "--no_trust",
}
for name, flag in ABLATIONS.items():
    out = os.path.join(RESULTS_ROOT, "extra", f"{name}.csv")
    CONFIGS[name] = {"script": "tact_prototype.py", "args": f"{ABLATION_COMMON} {flag} --out {out}", "out": out}

# --- Bounded (norm-capped) attack: TACT-full vs TACT-no-trust ---
BOUNDED_COMMON = "--num_agents 10 --num_malicious 2 --alpha 0.5 --rounds 50 --seed 0 --poison_bounded"
CONFIGS["bounded_tact_full"] = {
    "script": "tact_prototype.py",
    "args": f"{BOUNDED_COMMON} --out {os.path.join(RESULTS_ROOT, 'bounded', 'bounded_tact_full.csv')}",
    "out": os.path.join(RESULTS_ROOT, "bounded", "bounded_tact_full.csv"),
}
CONFIGS["bounded_tact_no_trust"] = {
    "script": "tact_prototype.py",
    "args": f"{BOUNDED_COMMON} --no_trust --out {os.path.join(RESULTS_ROOT, 'bounded', 'bounded_tact_no_trust.csv')}",
    "out": os.path.join(RESULTS_ROOT, "bounded", "bounded_tact_no_trust.csv"),
}

# --- Momentum retune (2026-10-02): ablation showed trust_momentum=0 scores 80.1%
# vs default 0.8's 53.9% under the (now realistic, escalating-magnitude) unbounded
# attack -- momentum smooths/delays the trust response, giving the attacker a
# window before being downweighted. Sweep a few values x 3 seeds to find a good
# setting, same MVP config otherwise. ---
MOMENTUM_VALUES = [0.0, 0.3, 0.5]
for momentum in MOMENTUM_VALUES:
    for seed in [0, 1, 2]:
        name = f"momentum_m{momentum}_seed{seed}"
        out = os.path.join(RESULTS_ROOT, "momentum_sweep", f"tact_m{momentum}_seed{seed}.csv")
        CONFIGS[name] = {
            "script": "tact_prototype.py",
            "args": f"{MVP_COMMON} --seed {seed} --trust_momentum {momentum} --out {out}",
            "out": out,
        }

# --- Assignment across 3 workers, roughly proportional to measured
# per-worker speed (two fast workers, one slower one). ---
GROUPS = {
    "worker_fast_0": [
        "mvp_tact_seed0", "mvp_tact_seed1",
        "mvp_fedavg_seed0", "mvp_fedavg_seed1",
        "mvp_fedprox_seed0", "mvp_fedprox_seed1",
        "ablation_no_uq", "ablation_no_history",
        "bounded_tact_full",
    ],
    "worker_fast_1": [
        "mvp_tact_seed2",
        "mvp_krum_seed0", "mvp_krum_seed1", "mvp_krum_seed2",
        "mvp_trimmedmean_seed0", "mvp_trimmedmean_seed1", "mvp_trimmedmean_seed2",
        "ablation_no_adaptation", "ablation_no_trust",
    ],
    "worker_slow": [
        "mvp_fedavg_seed2", "mvp_fedprox_seed2",
        "adaptive_alpha0.1",
        "bounded_tact_no_trust",
    ],
    "momentum_worker_fast_0": [
        "momentum_m0.0_seed0", "momentum_m0.0_seed1",
        "momentum_m0.3_seed0", "momentum_m0.3_seed1",
    ],
    "momentum_worker_fast_1": [
        "momentum_m0.0_seed2", "momentum_m0.3_seed2",
        "momentum_m0.5_seed0", "momentum_m0.5_seed1",
    ],
    "momentum_worker_slow": [
        "momentum_m0.5_seed2",
    ],
}

assert sorted(sum(GROUPS.values(), [])) == sorted(CONFIGS.keys()), "GROUPS must partition all CONFIGS exactly once"
