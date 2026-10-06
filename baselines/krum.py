"""Krum (Blanchard et al., 2017): selects the single client update that is
"most central" (smallest sum of squared distances to its closest neighbors),
discarding the rest entirely. Tolerates up to f Byzantine clients out of n
when n >= 2f + 3.
"""
import torch


def _flatten(state):
    return torch.cat([v.float().flatten() for v in state.values() if torch.is_floating_point(v)])


def krum_select(states, num_malicious_expected):
    n = len(states)
    f = num_malicious_expected
    k = max(1, n - f - 2)  # number of closest neighbors to sum over

    flat = [_flatten(s) for s in states]
    dists = torch.zeros(n, n)
    for a in range(n):
        for b in range(n):
            if a != b:
                dists[a, b] = torch.sum((flat[a] - flat[b]) ** 2)

    scores = []
    for a in range(n):
        nearest = torch.topk(dists[a], k=min(k, n - 1), largest=False).values
        scores.append(nearest.sum().item())

    best = int(torch.tensor(scores).argmin().item())
    return states[best], best
