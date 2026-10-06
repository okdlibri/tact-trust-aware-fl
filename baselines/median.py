"""Coordinate-wise median aggregation (Byzantine-robust baseline, Section 18;
also used as TACT's trust-free bootstrap aggregator, see Section 9.1 note in
experiments/tact_prototype.py)."""
import torch


def median_aggregate(states):
    out = {}
    for k in states[0]:
        if not torch.is_floating_point(states[0][k]):
            out[k] = states[0][k]
            continue
        stacked = torch.stack([s[k].float() for s in states], dim=0)
        out[k] = stacked.median(dim=0).values
    return out
