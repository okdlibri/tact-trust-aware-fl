# TACT: Trust-Aware Coordination for Federated Multi-Agent Learning

Reference implementation for **TACT**, a trust mechanism for federated
learning that formulates the problem as multi-agent coordination: each
participant maintains a dynamic, per-peer trust belief derived from
behavioral evidence (confident-but-wrong predictions on the judging
agent's own held-out data) and predictive uncertainty, and uses that
belief to accept, down-weight, or reject each peer's contribution every
round.

This repository accompanies an anonymous submission and is provided for
review and reproducibility. It contains the full experimental codebase:
the TACT mechanism itself, all baselines and attacks evaluated against
it, and the experiment drivers used to produce every table and figure in
the paper.

## Repository structure

```
agents/          Agent abstraction: local training, held-out evaluation
                 of a peer's broadcast model (accuracy, confident-wrong
                 rate, predictive entropy).
trust/           The core trust update (trust_estimator.py): maps one
                 round of behavioral evidence to a target trust value,
                 combining an uncertainty-aware forgiveness term with a
                 confident-wrong penalty, then smooths it with momentum.
collaboration/   The discrete collaboration policy (policy.py): maps a
                 trust value to accept / downweight / reject and the
                 corresponding aggregation weight.
attacks/         Model-poisoning attacks: update negation (plain and
                 norm-bounded) and an ALIE-style ("a little is enough",
                 Baruch et al. 2019) attack parameterized by a magnitude
                 multiplier z, applied in parameter space.
baselines/       FLTrust, Krum, coordinate-wise median, and trimmed mean.
models/          Compact ResNet-9 for CIFAR-10.
data/            CIFAR-10 loading (auto-downloads via torchvision) and
                 Dirichlet non-IID partitioning across agents.
analysis/        Figure generation from experiment output CSVs.
experiments/     Experiment drivers (see below).
```

## Setup

```
pip install -r requirements.txt
```

Requires PyTorch with CUDA for practical runtimes; CPU works but is slow
for the full 50-round, 10-agent experiments. CIFAR-10 downloads
automatically on first run via `torchvision`.

## Running an experiment

The core script is `experiments/tact_prototype.py`. It runs `n` fully
decentralized, personalized agents (no central server or global model)
for a fixed number of communication rounds, with a configurable fraction
of adversarial agents.

```
python experiments/tact_prototype.py \
    --num_agents 10 --num_malicious 2 --alpha 0.5 --rounds 50 \
    --seed 0 --trust_momentum 0.3 --out results/example.csv
```

Key flags:
- `--alpha`: Dirichlet concentration controlling non-IID heterogeneity
  (lower = more heterogeneous; the paper sweeps `{10, 1, 0.5, 0.1}`).
- `--trust_momentum`: the trust update's temporal-memory parameter `mu`.
  **The code default is 0.8 (the original/reference value); the paper's
  main evaluation uses `mu=0.3`, selected via the sensitivity sweep in
  the Results section and applied identically throughout.** Pass
  `--trust_momentum 0.3` explicitly to reproduce the headline numbers.
- `--attack_type {reversal,alie}` and `--poison_bounded`: select the
  attack family; `--alie_z` sets the ALIE magnitude multiplier swept in
  the paper's magnitude-dependence experiment.
- `--no_uncertainty`, `--no_adaptive_policy`, `--no_trust`: the three
  ablations reported in the paper (remove uncertainty-awareness, remove
  the discrete action tiers, remove trust-gating entirely).
- `--attack_onset`: delay poisoning onset to test detection of a
  behavior change rather than a persistently malicious agent.

`experiments/personalized_median.py` and `experiments/baseline_run.py`
run the personalized-median/FedAvg controls and the single-global-model
baselines (FedAvg, FedProx, Krum, Trimmed Mean) respectively, under the
same CLI conventions.

## Reproducing the paper's tables

`experiments/all_configs.py` is the single source of truth mapping every
named configuration in the paper to its exact CLI invocation, and
`experiments/run_group.py --group <name>` runs a named subset
sequentially, skipping any output file that already has the expected
number of rows (safe to re-invoke or split across machines). The other
`experiments/run_*.py` scripts are drivers for specific follow-up
experiments (heterogeneity sweep, adaptive attacker, ablations).

Each run writes a single CSV with one row per communication round
(accuracy, honest/malicious trust, false isolation rate). `analysis/plots.py`
regenerates the paper's figures from these CSVs.

## Notes on scope

This repository contains code only. It does not include raw experiment
output, the paper source, or any machine-specific configuration —
only what is needed to run and inspect the method itself.
