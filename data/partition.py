"""Dirichlet-based non-IID partitioning across federated agents.

alpha controls heterogeneity: alpha=10 ~ near-IID, alpha=0.1 ~ extreme non-IID.
"""
import numpy as np


def dirichlet_partition(labels, num_clients, alpha, num_classes=10, seed=0, min_size=10):
    """Partition sample indices into num_clients shards via a Dirichlet distribution
    over classes, following the standard non-IID FL partitioning scheme.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    while True:
        client_idxs = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, split_points)
            for i, split in enumerate(splits):
                client_idxs[i].extend(split.tolist())
        sizes = [len(x) for x in client_idxs]
        if min(sizes) >= min_size:
            break
        # retry with a new draw if some client ended up (almost) empty

    for i in range(num_clients):
        rng.shuffle(client_idxs[i])
    return client_idxs


if __name__ == "__main__":
    labels = np.random.randint(0, 10, size=50000)
    for alpha in [10, 1, 0.5, 0.1]:
        parts = dirichlet_partition(labels, 10, alpha, seed=0)
        sizes = [len(p) for p in parts]
        print(f"alpha={alpha}: sizes={sizes}")
