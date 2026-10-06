"""Adaptive collaboration policy: trust changes behavior, not just a
diagnostic score.

Thresholds are deliberately simple and rule-based (no learned/RL policy).
"""
import torch

ACCEPT_T = 0.7
VERIFY_T = 0.4
DOWNWEIGHT_T = 0.1


def decide(trust):
    """Return (action, weight_multiplier) for a peer at this trust level.

    accept:      full weight
    verify:      full weight for now (a "verify" event is logged; the
                 penalty for failing verification shows up next round via
                 the trust update itself, since judge_peer() already
                 re-evaluates the peer's behavior every round)
    downweight:  reduced weight
    reject:      isolated (zero weight)
    """
    if trust >= ACCEPT_T:
        return "accept", trust
    elif trust >= VERIFY_T:
        return "verify", trust
    elif trust >= DOWNWEIGHT_T:
        return "downweight", trust * 0.3
    else:
        return "reject", 0.0


def weighted_aggregate(own_state, peer_states_and_weights):
    """Combine this agent's own (fully-trusted) state with peer states,
    weighted by collaboration weights, normalized to sum to 1 across
    {self} U {accepted/verified/downweighted peers}. Rejected peers (weight
    0) contribute nothing. Non-floating buffers keep the agent's own value.
    """
    total_w = 1.0 + sum(w for _, w in peer_states_and_weights)
    new_state = {}
    for k, v in own_state.items():
        if not torch.is_floating_point(v):
            new_state[k] = v
            continue
        acc = v * (1.0 / total_w)
        for state, w in peer_states_and_weights:
            if w > 0:
                acc = acc + state[k] * (w / total_w)
        new_state[k] = acc
    return new_state
