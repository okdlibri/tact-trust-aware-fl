"""FLTrust (Cao et al., NDSS 2021): a server holding a small, clean root
dataset computes its own reference update each round; each client's update
is trusted proportionally to the ReLU-clipped cosine similarity between its
update and the server's reference, and rescaled to the server's update
norm before averaging (so a malicious client cannot buy influence purely
by scaling its update's magnitude).

    g_0        = server's own update (root-dataset training)
    g_i        = client i's update
    TS_i       = max(0, cos_similarity(g_i, g_0))
    g_i'       = (||g_0|| / ||g_i||) * g_i
    G_{t+1}    = G_t + sum_i(TS_i * g_i') / sum_i(TS_i)

A client whose update is non-finite (numerically broken, e.g. from an
exploding attack) gets a hard TS=0 rather than risking a NaN cosine
similarity propagating through the sum -- the same hard-safety-rail
principle used throughout this codebase (see attacks/poisoning.py,
agents/agent.py).
"""
import torch


def _delta(trained_state, prev_state):
    return {k: trained_state[k] - prev_state[k] for k in trained_state if torch.is_floating_point(trained_state[k])}


def _flatten(d):
    return torch.cat([v.float().flatten() for v in d.values()])


def fltrust_aggregate(prev_state, server_trained_state, client_trained_states):
    """Returns (new_global_state, trust_scores, fallback_used).
    trust_scores[i] is client i's TS_i for this round (useful for
    logging/analysis, mirroring TACT's trust trajectory). fallback_used
    indicates every client scored <=0 this round, triggering a one-round
    equal-weight fallback (see below) instead of freezing the model."""
    g0 = _delta(server_trained_state, prev_state)
    g0_flat = _flatten(g0)
    g0_norm = g0_flat.norm().item()

    trust_scores = []
    client_deltas = []
    valid = []  # finite-output flag, independent of trust score -- a non-finite
    # client must NEVER be blended in, even in the zero-trust fallback below,
    # since NaN is "sticky" under averaging (same principle as elsewhere in
    # this codebase: attacks/poisoning.py, agents/agent.py).
    for cs in client_trained_states:
        gi = _delta(cs, prev_state)
        gi_flat = _flatten(gi)
        if not torch.isfinite(gi_flat).all() or not torch.isfinite(torch.tensor(g0_norm)):
            trust_scores.append(0.0)
            client_deltas.append((gi, 0.0))
            valid.append(False)
            continue
        gi_norm = gi_flat.norm().item()
        if gi_norm < 1e-12 or g0_norm < 1e-12:
            cos = 0.0
        else:
            cos = (gi_flat @ g0_flat).item() / (gi_norm * g0_norm)
        trust_scores.append(max(0.0, cos))
        client_deltas.append((gi, gi_norm))
        valid.append(True)

    total_ts = sum(trust_scores)
    fallback_used = total_ts < 1e-12
    if fallback_used:
        # Every client scored <=0 this round (plausible noise with a small
        # root set / few local epochs). Freezing the global model here would
        # be a self-reinforcing trap: next round, every client and the
        # server train from the SAME unchanged state and likely reproduce
        # the same zero-trust outcome, permanently stalling training
        # (confirmed empirically). Fall back to equal-weight averaging for
        # THIS ROUND ONLY so the state keeps moving and next round's trust
        # computation starts from fresh, different gradients.
        weights = [1.0 if v else 0.0 for v in valid]
        total_ts = float(sum(weights)) or 1.0  # guard: if literally everyone is non-finite, avoid div-by-zero (agg stays all-zero -> global unchanged, the correct behavior in that worse-case scenario)
    else:
        weights = trust_scores

    out = {}
    for k in prev_state:
        if not torch.is_floating_point(prev_state[k]):
            out[k] = server_trained_state[k]
            continue
        agg = torch.zeros_like(prev_state[k], dtype=torch.float32)
        for (gi, gi_norm), w in zip(client_deltas, weights):
            if w <= 0.0 or gi_norm < 1e-12:
                continue
            scale = (g0_norm / gi_norm) if not fallback_used else 1.0
            agg = agg + w * scale * gi[k]
        agg = agg / total_ts
        out[k] = prev_state[k] + agg.to(prev_state[k].dtype)

    return out, trust_scores, fallback_used


def carve_root_set(labels, root_size, num_classes=10, seed=0):
    """Hold out a small, class-balanced root set BEFORE client partitioning,
    so it never overlaps with any client's training data or the test set.
    Returns (root_indices, remaining_indices), both as plain index arrays
    into the original dataset -- remaining_indices should be what gets
    passed to dirichlet_partition (with its own index-mapping handled by
    the caller, since dirichlet_partition returns positions within
    whatever label array it's given, not original dataset indices).
    """
    import numpy as np

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed + 99991)  # offset so this draw is independent of client partitioning's own RNG stream
    per_class = max(1, root_size // num_classes)
    root_indices = []
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        root_indices.extend(idx_c[:per_class].tolist())
    root_indices = np.array(sorted(root_indices))
    mask = np.ones(len(labels), dtype=bool)
    mask[root_indices] = False
    remaining_indices = np.where(mask)[0]
    return root_indices, remaining_indices
