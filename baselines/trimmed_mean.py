"""Coordinate-wise trimmed mean (Yin et al., 2018): per parameter, drop the
top and bottom `trim_k` values across clients, average the rest.
"""
import torch


def trimmed_mean_aggregate(states, trim_k):
    n = len(states)
    trim_k = min(trim_k, (n - 1) // 2)  # must leave at least 1 value after trimming both ends
    out = {}
    for k in states[0]:
        if not torch.is_floating_point(states[0][k]):
            out[k] = states[0][k]
            continue
        stacked = torch.stack([s[k].float() for s in states], dim=0)  # [n, ...]
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[trim_k : n - trim_k] if trim_k > 0 else sorted_vals
        out[k] = trimmed.mean(dim=0)
    return out
