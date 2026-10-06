""""A Little Is Enough" (Baruch, Baruch & Goldberg, NeurIPS 2019), simplified.

Unlike attacks/poisoning.py's update-reversal (which repeatedly reverses the
ATTACKER'S OWN compounding trained delta, round over round -- the mechanism
behind the BatchNorm-variance-explosion bug this project hit earlier), ALIE
is computed FRESH each round directly from the (bounded, always-finite)
HONEST agents' current trained states: malicious_j = mean_honest_j -
z * std_honest_j, per coordinate. It cannot escalate or explode numerically
round over round, because it never depends on its own output from a prior
round -- only on this round's honest values, which are always finite since
honest agents simply train normally.

This is the attack class robust aggregators (Krum, trimmed mean, coordinate
median) are NOT designed for: by staying within a small, calibrated multiple
of the honest standard deviation, a malicious value does not look like an
extreme outlier at any single coordinate, unlike full update-reversal (whose
magnitude is unconstrained and trivially filtered by any distance- or
range-based rule). We use z=1.0 (one honest standard deviation) as a simple,
representative instantiation of the paper's "a little is enough" principle,
rather than solving for the aggregation-rule-specific optimal z from the
paper's Section 3.2 -- there is no such closed form for TACT's behavioral
(accuracy/confidence) evidence, which is not a coordinate-wise statistic.

All colluding malicious agents broadcast the IDENTICAL crafted state (the
paper's own threat model: an omniscient attacker controlling all malicious
clients, which already observes every other agent's broadcast in our fully
connected communication model, Section~\ref{sec:formulation}).
"""
import torch

from attacks.poisoning import _is_bn_running_stat


def alie_attack(honest_states, z=1.0):
    if len(honest_states) < 2:
        raise ValueError("ALIE needs at least 2 honest states to compute a standard deviation")
    out = {}
    for k in honest_states[0]:
        if not torch.is_floating_point(honest_states[0][k]) or _is_bn_running_stat(k):
            out[k] = honest_states[0][k]
            continue
        stacked = torch.stack([s[k].float() for s in honest_states], dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        out[k] = mean - z * std
    return out
