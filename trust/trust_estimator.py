"""Behavioral, uncertainty-aware trust update.

Deliberately simple for the first implementation, but with uncertainty
(UQ) as a real, ablatable term (needed for the TACT-no-UQ ablation), not
just a logged-but-unused statistic.

Core idea, made concrete:
  - A peer that is WRONG but UNCERTAIN (high entropy) on my local data is
    plausibly just honestly heterogeneous -> its accuracy shortfall is
    forgiven.
  - A peer that is WRONG and CONFIDENT (low entropy / high conf_wrong) is
    anomalous -> its accuracy shortfall counts in full, plus an extra
    penalty proportional to how confident it was while wrong.
"""
import math

DEFAULT_NUM_CLASSES = 10


def evidence_to_target_trust(evidence, beta=1.0, gamma=0.7, num_classes=DEFAULT_NUM_CLASSES, use_uncertainty=True):
    """Map one round of behavioral evidence (from Agent.judge_peer) to a
    target trust value in [0, 1].

    shortfall          = 1 - acc
    forgiveness        = gamma * normalized_entropy   (0 if use_uncertainty=False; this is the UQ ablation switch)
    adjusted_shortfall = shortfall * (1 - forgiveness)
    target             = clip(1 - adjusted_shortfall - beta * conf_wrong, 0, 1)
    """
    acc = evidence["acc"]
    conf_wrong = evidence["conf_wrong"]
    entropy = evidence.get("entropy", 0.0)

    max_entropy = math.log(num_classes)
    norm_entropy = max(0.0, min(1.0, entropy / max_entropy)) if max_entropy > 0 else 0.0

    forgiveness = gamma * norm_entropy if use_uncertainty else 0.0
    shortfall = 1.0 - acc
    adjusted_shortfall = shortfall * (1.0 - forgiveness)

    target = 1.0 - adjusted_shortfall - beta * conf_wrong
    if not math.isfinite(target):
        # NaN/Inf evidence must never silently clamp to "fully trustworthy"
        # (Python's min/max with NaN is order-dependent and can do exactly
        # that) -- treat it as the worst possible target instead.
        return 0.0
    return max(0.0, min(1.0, target))


def update_trust(prev_trust, evidence, momentum=0.8, beta=1.0, gamma=0.7, num_classes=DEFAULT_NUM_CLASSES, use_uncertainty=True):
    """T_ij^{t+1} = momentum * T_ij^t + (1 - momentum) * target(evidence).

    momentum encodes historical reliability H_ij: higher momentum means trust
    changes more slowly and is harder to swing with a single round of evidence
    (the TACT-no-history ablation sets momentum=0, i.e. no memory).
    """
    target = evidence_to_target_trust(
        evidence, beta=beta, gamma=gamma, num_classes=num_classes, use_uncertainty=use_uncertainty
    )
    return momentum * prev_trust + (1 - momentum) * target
