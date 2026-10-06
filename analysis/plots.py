"""Figures for the paper. Output as PDF, no titles (captions carry that
role in the paper), clean/minimal style. Run from the repo root.
"""
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# Colorblind-safe, print-safe palette (Okabe-Ito subset).
COLORS = {
    "fedavg": "#999999",
    "fedprox": "#56B4E9",
    "krum": "#E69F00",
    "trimmedmean": "#009E73",
    "tact": "#D55E00",
    "honest": "#0072B2",
    "malicious": "#D55E00",
    "m0.3": "#D55E00",
    "m0.8": "#999999",
}


def load(path):
    with open(path) as f:
        return list(csv.reader(f))[1:]


def col(rows, idx):
    return np.array([float(r[idx]) for r in rows])


def mean_std_across_seeds(paths, col_idx):
    series = [col(load(p), col_idx) for p in paths]
    n = min(len(s) for s in series)
    series = np.stack([s[:n] for s in series])
    return series.mean(axis=0), series.std(axis=0)


def fig_learning_and_trust():
    """Figure: (left) accuracy vs round for all methods; (right) TACT's
    honest vs malicious trust vs round, same experiment. mean +/- std over
    3 seeds. Main evaluation condition: alpha=0.5, 20% malicious, unbounded
    model poisoning, TACT at the selected momentum=0.3."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    methods = [
        ("fedavg", "FedAvg", os.path.join(RESULTS, "mvp", "fedavg_seed{}.csv")),
        ("fedprox", "FedProx", os.path.join(RESULTS, "mvp", "fedprox_seed{}.csv")),
        ("krum", "Krum", os.path.join(RESULTS, "mvp", "krum_seed{}.csv")),
        ("trimmedmean", "Trimmed Mean", os.path.join(RESULTS, "mvp", "trimmedmean_seed{}.csv")),
        ("tact", "TACT", os.path.join(RESULTS, "momentum_sweep", "tact_m0.3_seed{}.csv")),
    ]
    ax = axes[0]
    for key, label, pattern in methods:
        paths = [pattern.format(s) for s in (0, 1, 2)]
        mean, std = mean_std_across_seeds(paths, 1)
        rounds = np.arange(1, len(mean) + 1)
        ax.plot(rounds, mean * 100, label=label, color=COLORS[key], linewidth=1.4)
        ax.fill_between(rounds, (mean - std) * 100, (mean + std) * 100, color=COLORS[key], alpha=0.15, linewidth=0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=7, ncol=1)

    ax = axes[1]
    tact_paths = [os.path.join(RESULTS, "momentum_sweep", f"tact_m0.3_seed{s}.csv") for s in (0, 1, 2)]
    h_mean, h_std = mean_std_across_seeds(tact_paths, 2)
    m_mean, m_std = mean_std_across_seeds(tact_paths, 3)
    rounds = np.arange(1, len(h_mean) + 1)
    ax.plot(rounds, h_mean, label="Honest trust", color=COLORS["honest"], linewidth=1.4)
    ax.fill_between(rounds, h_mean - h_std, h_mean + h_std, color=COLORS["honest"], alpha=0.15, linewidth=0)
    ax.plot(rounds, m_mean, label="Malicious trust", color=COLORS["malicious"], linewidth=1.4)
    ax.fill_between(rounds, m_mean - m_std, m_mean + m_std, color=COLORS["malicious"], alpha=0.15, linewidth=0)
    ax.axhline(0.1, color="black", linestyle=":", linewidth=1.0, label="Reject threshold")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Trust")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    path = os.path.join(OUT, "fig_learning_and_trust.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_heterogeneity():
    """Figure: false isolation and honest trust vs. heterogeneity (alpha),
    zero-malicious condition. Demonstrates heterogeneity alone does not
    cause false isolation except at the most extreme level tested.

    Uses the hetero_alpha{a}_m0.3.csv files (re-run 2026-10-05 at the
    paper's selected momentum=0.3), not the original hetero_alpha{a}.csv
    files, which were silently computed at the pre-selection default
    momentum=0.8 -- inconsistent with the paper's claim that mu=0.3 is
    used throughout. The two differ only slightly (no active attacker to
    react to), but this keeps Table 4 and this figure genuinely matched."""
    alphas = [10, 1, 0.5, 0.1]
    labels = ["10", "1", "0.5", "0.1"]
    fiso, trust = [], []
    for a in alphas:
        rows = load(os.path.join(RESULTS, "extra", f"hetero_alpha{a}_m0.3.csv"))
        fiso.append(np.mean([float(r[4]) for r in rows[-5:]]) * 100)
        trust.append(np.mean([float(r[2]) for r in rows[-5:]]))

    fig, ax1 = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(alphas))
    bars = ax1.bar(x, fiso, color=COLORS["malicious"], alpha=0.75, width=0.5, label="False isolation")
    ax1.set_ylabel("False isolation (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(r"Heterogeneity ($\alpha$, lower = more heterogeneous)")
    ax1.set_ylim(0, max(fiso) * 1.3 + 1)

    ax2 = ax1.twinx()
    ax2.plot(x, trust, color=COLORS["honest"], marker="o", markersize=4, linewidth=1.4, label="Honest trust")
    ax2.set_ylabel("Honest trust")
    ax2.set_ylim(0, 1.0)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

    fig.tight_layout()
    path = os.path.join(OUT, "fig_heterogeneity.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_adaptive_tradeoff():
    """Figure: malicious trust (down) and false isolation (up) over rounds,
    mu=0.3 vs mu=0.8, adaptive attacker (alpha=0.1, onset round 21)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharex=True)

    configs = [
        ("m0.3", r"$\mu{=}0.3$ (selected)", [
            os.path.join(RESULTS, "extra", "adaptive_alpha0.1_momentum0.3.csv"),
            os.path.join(RESULTS, "extra", "adaptive_alpha0.1_momentum0.3_seed1.csv"),
            os.path.join(RESULTS, "extra", "adaptive_alpha0.1_momentum0.3_seed2.csv"),
        ]),
        ("m0.8", r"$\mu{=}0.8$ (reference)", [
            os.path.join(RESULTS, "extra", "adaptive_alpha0.1_mal20pct_onset21.csv"),
            os.path.join(RESULTS, "extra", "adaptive_alpha0.1_momentum0.8_seed1.csv"),
            os.path.join(RESULTS, "extra", "adaptive_alpha0.1_momentum0.8_seed2.csv"),
        ]),
    ]

    ax = axes[0]
    for key, label, paths in configs:
        mean, std = mean_std_across_seeds(paths, 3)
        rounds = np.arange(1, len(mean) + 1)
        ax.plot(rounds, mean, label=label, color=COLORS[key], linewidth=1.4)
        ax.fill_between(rounds, mean - std, mean + std, color=COLORS[key], alpha=0.15, linewidth=0)
    ax.axvline(21, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(0.1, color="black", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Malicious trust")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(loc="center right", fontsize=7)

    ax = axes[1]
    for key, label, paths in configs:
        mean, std = mean_std_across_seeds(paths, 4)
        mean, std = mean * 100, std * 100
        rounds = np.arange(1, len(mean) + 1)
        ax.plot(rounds, mean, label=label, color=COLORS[key], linewidth=1.4)
        ax.fill_between(rounds, mean - std, mean + std, color=COLORS[key], alpha=0.15, linewidth=0)
    ax.axvline(21, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("False isolation (%)")
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    path = os.path.join(OUT, "fig_adaptive_tradeoff.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    fig_learning_and_trust()
    fig_heterogeneity()
    fig_adaptive_tradeoff()
