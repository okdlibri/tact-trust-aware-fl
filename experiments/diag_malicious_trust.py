"""One-off diagnostic for feedback.txt item 4: is the malicious trust in the
heterogeneity x attack grid (Table 3, bounded-attack cells) exactly 0.0000
because of the hard non-finite safety rail, or because evidence_to_target_trust
genuinely computes target=0.0 from finite (but very bad) accuracy/confidence
evidence? Reproduces alpha0.5_bounded's exact config (seed 0, momentum 0.3,
num_malicious 2) through the warmup + first few post-warmup rounds, and prints
the RAW evidence dict (acc, conf_wrong, entropy, valid) for one honest agent's
judgment of one malicious peer every round, rather than just the EMA trust
value that the main run logs.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent
from attacks.poisoning import poison_update_bounded
from baselines.median import median_aggregate
from data.cifar10 import load_cifar10
from data.partition import dirichlet_partition
from models.resnet9 import ResNet9
from trust.trust_estimator import update_trust

SEED = 0
NUM_AGENTS = 10
NUM_MALICIOUS = 2
ALPHA = 0.5
MOMENTUM = 0.3
WARMUP = 3
ROUNDS_TO_CHECK = 8

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

train_set, test_set = load_cifar10(root=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
labels = np.array(train_set.targets)
client_idxs = dirichlet_partition(labels, NUM_AGENTS, ALPHA, seed=SEED)
malicious_ids = set(range(NUM_MALICIOUS))
honest_ids = [i for i in range(NUM_AGENTS) if i not in malicious_ids]
judge, target_peer = honest_ids[0], 0  # agent judging; malicious peer 0

model_fn = lambda: ResNet9(num_classes=10)
agents = [
    Agent(i, train_set, client_idxs[i], model_fn, device, is_malicious=(i in malicious_ids), seed=SEED)
    for i in range(NUM_AGENTS)
]
shared_init = agents[0].get_state()
for a in agents[1:]:
    a.set_state(shared_init)

trust_j0 = 0.5
for rnd in range(1, ROUNDS_TO_CHECK + 1):
    prev_states = {i: agents[i].get_state() for i in range(NUM_AGENTS)}
    for a in agents:
        a.local_train(epochs=1)

    broadcast = {}
    for i in range(NUM_AGENTS):
        st = agents[i].get_state()
        if i in malicious_ids:
            st = poison_update_bounded(prev_states[i], st, lam=1.0, max_norm=800.0)
            agents[i].set_state(st)
        broadcast[i] = st

    if rnd <= WARMUP:
        boot = median_aggregate([broadcast[i] for i in range(NUM_AGENTS)])
        for i in range(NUM_AGENTS):
            agents[i].set_state(boot)
        print(f"round {rnd}: warmup (median bootstrap), trust not yet evaluated")
        continue

    evidence = agents[judge].judge_peer(broadcast[target_peer], temperature=3.0)
    trust_j0 = update_trust(trust_j0, evidence, momentum=MOMENTUM, beta=1.0, gamma=0.7, use_uncertainty=True)
    print(
        f"round {rnd}: agent{judge} judging malicious peer{target_peer} -> "
        f"acc={evidence['acc']:.4f} conf_wrong={evidence['conf_wrong']:.4f} "
        f"entropy={evidence['entropy']:.4f} valid={evidence['valid']} "
        f"-> T_{{{judge},{target_peer}}}={trust_j0:.6e}"
    )
