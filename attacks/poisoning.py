"""Model poisoning via update negation (gradient/update reversal).

poisoned_update = -lambda * honest_update, applied on top of the agent's own
pre-round state, so a malicious agent still LOOKS like it trained (parameters
moved), just in the wrong direction.

BatchNorm's running_mean/running_var are excluded from the reversal: they are
not learned weights, they're a running estimate of activation statistics, and
variance must stay non-negative. Reversing their delta like an ordinary
parameter can trivially push running_var negative, which makes the very next
forward pass (sqrt of a negative variance) produce NaN -- regardless of how
strong or bounded the attack on the real weights is. That would make the
attack trivially catchable by basic finite-value filtering alone, rather than
by actually-adversarial (but numerically valid) behavior. These buffers are
passed through as the attacker's own genuinely-observed statistics instead.
"""
import torch

_BN_STATS_SUBSTRINGS = ("running_mean", "running_var")


def _is_bn_running_stat(key):
    return any(s in key for s in _BN_STATS_SUBSTRINGS)


def poison_update(prev_state, trained_state, lam=1.0):
    poisoned = {}
    for k in trained_state:
        if not torch.is_floating_point(trained_state[k]) or _is_bn_running_stat(k):
            poisoned[k] = trained_state[k]
            continue
        honest_delta = trained_state[k] - prev_state[k]
        poisoned[k] = prev_state[k] - lam * honest_delta
    return poisoned


def poison_update_bounded(prev_state, trained_state, lam=1.0, max_norm=800.0):
    """Norm-bounded variant: the reversed update's overall L2 norm (over the
    poisoned weight tensors only, excluding BN running stats) is capped at
    max_norm (a typical honest round-1 update has norm ~700-800 for ResNet9
    here), so it always stays a plausible-looking training step in magnitude
    -- the realistic/stealthy threat model, where an attacker would not want
    its update to look anomalously large either.
    """
    flat = torch.cat(
        [
            (trained_state[k] - prev_state[k]).float().flatten()
            for k in trained_state
            if torch.is_floating_point(trained_state[k]) and not _is_bn_running_stat(k)
        ]
    )
    honest_norm = flat.norm().item()
    reversed_norm = lam * honest_norm
    scale = lam
    if reversed_norm > max_norm and reversed_norm > 0:
        scale = lam * (max_norm / reversed_norm)

    poisoned = {}
    for k in trained_state:
        if not torch.is_floating_point(trained_state[k]) or _is_bn_running_stat(k):
            poisoned[k] = trained_state[k]
            continue
        honest_delta = trained_state[k] - prev_state[k]
        poisoned[k] = prev_state[k] - scale * honest_delta
    return poisoned
