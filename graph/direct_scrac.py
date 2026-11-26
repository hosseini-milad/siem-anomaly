"""Label-blind representative-priority SCRAC, without transport optimization."""
import numpy as np
from graph.sc_fagc import mixed_similarity, size_penalty, _validate_features
from preprocessing.reduction import select_representatives


def direct_scrac(x, b, c, mass, *, threshold=.85, capacity=200,
                 lambda_size=.20, acceptance_threshold=.75, anchor_ratio=.05,
                 topology_labels=None, seed=42):
    x, b, c, mass = _validate_features(x, b, c, mass)
    n = len(x)
    if capacity < 1 or not 0 < anchor_ratio <= 1 or not -1 <= threshold <= 1 or lambda_size < 0 or not 0 <= acceptance_threshold <= 1:
        raise ValueError("invalid SCRAC parameters")
    topology = None if topology_labels is None else np.asarray(topology_labels)
    if topology is not None and topology.shape != (n,):
        raise ValueError("unaligned topology")
    budget = max(1, int(np.ceil(n * anchor_ratio)))
    # Leave room for coverage, rarity and diversity selection, not just random seeds.
    selected = select_representatives(x, b, c, ratio=anchor_ratio, seed=seed,
                                      anchors=max(1, min(256, budget // 4)))
    priority = selected["indices"]
    mask = np.ones(n, dtype=bool)
    mask[priority] = False
    order = np.r_[priority, np.flatnonzero(mask)]
    groups, seeds = [], []
    assignment = np.full(n, -1, dtype=np.int64)
    minimums = []
    for i in order:
        best = None
        if seeds:
            ids = np.asarray(seeds, dtype=np.int64)
            sims = mixed_similarity(x[i:i+1], b[i:i+1], c[i:i+1],
                                    x[ids], b[ids], c[ids]).ravel()
            eligible = sims > threshold
            if topology is not None:
                eligible &= topology[ids] == topology[i]
            for gid in np.flatnonzero(eligible):
                members = groups[gid]
                size = len(members) + 1
                if size > capacity:
                    continue
                s = mixed_similarity(x[i:i+1], b[i:i+1], c[i:i+1],
                                     x[members], b[members], c[members]).ravel()
                low = float(s.min())
                if low <= threshold:
                    continue
                value = .4 * sims[gid] + .3 * s.mean() + .3 * low - size_penalty(size, capacity, lambda_size)
                if value >= acceptance_threshold and (best is None or value > best[0]):
                    best = (float(value), int(gid), low)
        if best is None:
            gid = len(groups)
            groups.append([int(i)])
            seeds.append(int(i))
            minimums.append(None)
        else:
            _, gid, low = best
            groups[gid].append(int(i))
            minimums[gid] = low if minimums[gid] is None else min(minimums[gid], low)
        assignment[i] = gid
    sizes = [len(g) for g in groups]
    masses = [float(mass[g].sum()) for g in groups]
    if np.any(assignment < 0) or sum(sizes) != n or not np.isclose(sum(masses), mass.sum()):
        raise RuntimeError("SCRAC conservation failure")
    return assignment, {
        "method": "direct_scrac", "groups": len(groups),
        "priority_representatives": len(priority), "anchor_indices": seeds,
        "group_sizes": sizes, "group_mass_raw_rows": masses,
        "max_group_size": max(sizes), "singleton_groups": sizes.count(1),
        "capacity_violations": sum(s > capacity for s in sizes),
        "minimum_pair_similarity": min((v for v in minimums if v is not None), default=None),
        "group_minimum_pair_similarity": minimums,
        "represented_raw_rows": float(mass.sum()),
        "topology_constraint_enabled": topology is not None,
        "lambda_size": lambda_size, "acceptance_threshold": acceptance_threshold,
        "complete_link_threshold": threshold,
    }
