"""Agent abstraction: an autonomous learning participant.

Each agent owns local data, a local model copy, and (later) trust state /
interaction history / a collaboration policy. This module implements only
the learning core (Phase 1/2); trust state is added on top in trust/.
"""
import torch
from torch.utils.data import DataLoader, Subset


class Agent:
    def __init__(
        self,
        agent_id,
        dataset,
        indices,
        model_fn,
        device,
        batch_size=64,
        lr=0.01,
        is_malicious=False,
        val_frac=0.15,
        seed=0,
    ):
        self.id = agent_id
        self.indices = list(indices)
        self.device = device
        self.model = model_fn().to(device)
        self._peer_probe_model = model_fn().to(device)  # reused scratch model for judge_peer()
        self.lr = lr
        self.is_malicious = is_malicious

        # Hold out a slice of this agent's own local data as "reference
        # evidence" used only to judge peers (never trained on). This keeps
        # trust fully decentralized: no shared/public validation set needed.
        g = torch.Generator().manual_seed(seed * 1000 + agent_id)
        perm = torch.randperm(len(self.indices), generator=g).tolist()
        n_val = max(1, int(len(self.indices) * val_frac))
        val_idx = [self.indices[i] for i in perm[:n_val]]
        train_idx = [self.indices[i] for i in perm[n_val:]]

        self.loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, drop_last=len(train_idx) > batch_size
        )
        self.val_loader = DataLoader(Subset(dataset, val_idx), batch_size=256, shuffle=False)

        # Trust layer state.
        self.trust = {}  # peer_id -> trust score in [0, 1]
        self.history = []  # interaction/behavior log (per-round evidence, for analysis)

    def load_global(self, global_state):
        self.model.load_state_dict(global_state)

    def local_train(self, epochs=1, prox_mu=0.0, global_state=None):
        """prox_mu > 0 adds the FedProx proximal term (mu/2)*||theta - global_theta||^2,
        penalizing drift from global_state (required when prox_mu > 0)."""
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        if prox_mu > 0:
            global_params = {k: v.detach().to(self.device) for k, v in global_state.items()}
        total_loss, n_batches = 0.0, 0
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = criterion(self.model(x), y)
                if prox_mu > 0:
                    prox_term = sum(
                        (p - global_params[name]).pow(2).sum()
                        for name, p in self.model.named_parameters()
                    )
                    loss = loss + (prox_mu / 2) * prox_term
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def get_delta(self, global_state):
        """Return (state_dict delta, num_samples) relative to the global model
        this agent trained from. Used by the centralized FedAvg baseline only."""
        local_state = self.model.state_dict()
        delta = {k: (local_state[k] - global_state[k].to(self.device)) for k in local_state}
        return delta, len(self.indices)

    def get_state(self):
        """Snapshot of this agent's own model, e.g. to broadcast to peers."""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_state(self, state):
        self.model.load_state_dict(state)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / total

    @torch.no_grad()
    def judge_peer(self, peer_state, temperature=1.0):
        """Evaluate a peer's model on THIS agent's own held-out validation
        split (never trained on). Returns behavioral evidence used for trust:
          acc:  peer accuracy on my local ground truth (consistency evidence)
          conf_wrong: mean confidence of the peer on samples it got WRONG
                      (high => confidently wrong => suspicious;
                       low  => unsure-but-wrong => plausibly honest heterogeneity)
          entropy: mean predictive entropy of the peer over my validation inputs

        temperature > 1 softens the peer's softmax before conf_wrong/entropy
        are computed (does not affect acc, since argmax is scale-invariant).
        Classifiers are typically overconfident even when wrong on
        out-of-distribution-like inputs (a well-known calibration failure);
        under non-IID heterogeneity, another honest agent's data looks
        OOD-ish to me, so raw confidence would misread legitimate
        heterogeneous disagreement as malicious. Temperature scaling is the
        standard fix (Guo et al. 2017).
        """
        temp_model = self._peer_probe_model
        temp_model.load_state_dict(peer_state)
        temp_model.eval()

        correct, total = 0, 0
        wrong_conf_sum, wrong_count = 0.0, 0
        entropy_sum = 0.0
        eps = 1e-12
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = temp_model(x)
            if not torch.isfinite(logits).all():
                # A peer producing NaN/Inf output is numerically broken (e.g. an
                # attack that exploded parameter magnitudes). That is maximally
                # suspicious, not neutral -- never let NaN silently become
                # "trust this peer" via downstream clamping/averaging. `valid`
                # additionally triggers a hard weight=0 override in the caller:
                # blending even a small weight of a NaN state poisons the
                # entire aggregate (NaN is "sticky" under averaging), so this
                # can't be left to the soft trust score alone.
                return {"acc": 0.0, "conf_wrong": 1.0, "entropy": 0.0, "valid": False}
            pred = logits.argmax(dim=1)  # argmax is temperature-invariant
            probs = torch.softmax(logits / temperature, dim=1)
            conf = probs.max(dim=1).values
            correct_mask = pred == y
            correct += correct_mask.sum().item()
            total += y.size(0)
            if (~correct_mask).any():
                wrong_conf_sum += conf[~correct_mask].sum().item()
                wrong_count += (~correct_mask).sum().item()
            entropy = -(probs * (probs + eps).log()).sum(dim=1)
            entropy_sum += entropy.sum().item()

        acc = correct / max(total, 1)
        conf_wrong = wrong_conf_sum / wrong_count if wrong_count > 0 else 0.0
        mean_entropy = entropy_sum / max(total, 1)
        return {"acc": acc, "conf_wrong": conf_wrong, "entropy": mean_entropy, "valid": True}
