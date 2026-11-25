"""Label-blind structural quality audit for node-level SCRAC assignments.

Computes exact within-group pair similarities in bounded group-sized matrices.
Optional adjacency.npz adds pre/post grouping graph-structure diagnostics.
No mapping source paths or attack/normal labels are read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from graph.sc_fagc import mixed_similarity


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "min": None, "p05": None, "median": None,
                "mean": None, "p95": None, "max": None, "std": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def _pair_metrics(x: np.ndarray, b: np.ndarray, c: np.ndarray,
                  idx: np.ndarray, threshold: float) -> dict:
    """Exact all-pairs metrics for one group; group size is bounded by SCRAC cap."""
    m = len(idx)
    if m == 1:
        return {
            "members": 1, "pairs": 0, "similarity": _summary(np.array([])),
            "continuous_cosine": _summary(np.array([])),
            "discrete_agreement": _summary(np.array([])),
            "complete_link_valid": True, "complete_link_min": None,
            "violating_pairs": 0, "medoid_node": int(idx[0]),
            "medoid_mean_similarity": 1.0, "boundary_members": [],
            "member_mean_similarity": [1.0],
        }

    gx, gb, gc = x[idx], b[idx], c[idx]
    sim = mixed_similarity(gx, gb, gc, gx, gb, gc)
    norms = np.linalg.norm(gx, axis=1)
    den = norms[:, None] * norms[None, :]
    cosine = np.divide(gx @ gx.T, den, out=np.zeros((m, m), dtype=np.float64),
                       where=den > 0)
    discrete = np.mean(
        np.column_stack((gb, gc))[:, None, :] ==
        np.column_stack((gb, gc))[None, :, :], axis=2
    )
    upper = np.triu_indices(m, k=1)
    pair_sim = sim[upper]
    pair_cos = cosine[upper]
    pair_discrete = discrete[upper]

    # Per-member weakest link and mean similarity to other members.
    off = sim.copy()
    np.fill_diagonal(off, np.inf)
    member_min = off.min(axis=1)
    member_mean = (sim.sum(axis=1) - 1.0) / (m - 1)
    medoid_local = int(np.argmax(member_mean))
    boundary_order = np.argsort(member_min, kind="stable")[:min(5, m)]
    violating = np.flatnonzero(pair_sim <= threshold)

    return {
        "members": int(m),
        "pairs": int(len(pair_sim)),
        "similarity": _summary(pair_sim),
        "continuous_cosine": _summary(pair_cos),
        "discrete_agreement": _summary(pair_discrete),
        "complete_link_valid": bool(pair_sim.min() > threshold),
        "complete_link_min": float(pair_sim.min()),
        "violating_pairs": int(len(violating)),
        "medoid_node": int(idx[medoid_local]),
        "medoid_mean_similarity": float(member_mean[medoid_local]),
        "boundary_members": [
            {"node": int(idx[j]), "minimum_similarity": float(member_min[j]),
             "mean_similarity": float(member_mean[j])}
            for j in boundary_order
        ],
        "member_mean_similarity": member_mean.tolist(),
    }


def _graph_metrics(graph_path: Path, labels: np.ndarray) -> dict:
    """Measure how original sparse edges fall within and between final groups."""
    g = load_npz(graph_path).tocsr().astype(np.float64)
    g.sum_duplicates()
    if g.shape != (len(labels), len(labels)):
        raise ValueError(f"graph shape {g.shape} does not match assignment length {len(labels)}")
    if (g != g.T).nnz:
        raise ValueError("input adjacency must be symmetric")
    coo = g.tocoo()
    mask = coo.row < coo.col
    rows, cols, weights = coo.row[mask], coo.col[mask], coo.data[mask]
    same = labels[rows] == labels[cols]
    intra_w = weights[same]
    inter_w = weights[~same]
    n_groups = int(labels.max()) + 1
    # Sparse quotient graph: aggregate original edge weights between group pairs.
    pair_a = np.minimum(labels[rows], labels[cols])
    pair_b = np.maximum(labels[rows], labels[cols])
    inter = ~same
    if np.any(inter):
        keys = pair_a[inter].astype(np.int64) * n_groups + pair_b[inter].astype(np.int64)
        uniq, inv = np.unique(keys, return_inverse=True)
        qweights = np.bincount(inv, weights=weights[inter])
        quotient_edges = [
            {"group_a": int(k // n_groups), "group_b": int(k % n_groups),
             "original_edge_count": int(np.count_nonzero(inv == j)),
             "weight_sum": float(qweights[j])}
            for j, k in enumerate(uniq)
        ]
    else:
        quotient_edges = []
    return {
        "original_graph_edges": int(len(weights)),
        "original_graph_weight_sum": float(weights.sum()),
        "intra_group_edges": int(same.sum()),
        "inter_group_edges": int((~same).sum()),
        "intra_group_weight_sum": float(intra_w.sum()),
        "inter_group_weight_sum": float(inter_w.sum()),
        "intra_edge_fraction": float(same.mean()) if len(same) else None,
        "inter_edge_fraction": float((~same).mean()) if len(same) else None,
        "intra_weight_fraction": float(intra_w.sum() / weights.sum()) if weights.sum() else None,
        "inter_weight_fraction": float(inter_w.sum() / weights.sum()) if weights.sum() else None,
        "quotient_graph_edges": quotient_edges,
        "quotient_graph_edge_count": len(quotient_edges),
    }


def evaluate(nodes_path: Path, assignment_report: Path, threshold: float = 0.85,
             capacity: int = 200, graph_path: Path | None = None) -> dict:
    started = time.perf_counter()
    with np.load(nodes_path, allow_pickle=False) as f:
        x, b, c = (np.asarray(f[k]) for k in ("x", "b", "c"))
    report = json.loads(Path(assignment_report).read_text(encoding="utf-8"))
    labels = np.asarray(report.get("assignment", []), dtype=np.int64)

    if x.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("nodes.npz must contain 2-D x, b and c arrays")
    if not (len(x) == len(b) == len(c) == len(labels)):
        raise ValueError("node features and assignment lengths are not aligned")
    if len(labels) == 0 or np.any(labels < 0):
        raise ValueError("assignment is empty or contains unassigned nodes")
    if not np.isfinite(x).all():
        raise ValueError("continuous node features contain non-finite values")
    if capacity < 1 or not -1 <= threshold <= 1:
        raise ValueError("invalid threshold or capacity")

    groups = []
    all_pair_sims = []
    all_pair_cos = []
    all_pair_disc = []
    group_sizes = []
    violations = []
    singleton_count = 0
    for gid in np.unique(labels):
        idx = np.flatnonzero(labels == gid)
        metrics = _pair_metrics(x, b, c, idx, threshold)
        metrics["group_id"] = int(gid)
        groups.append(metrics)
        group_sizes.append(len(idx))
        singleton_count += int(len(idx) == 1)
        if metrics["pairs"]:
            # Recompute arrays for compact global summaries; max group <= capacity.
            gx, gb, gc = x[idx], b[idx], c[idx]
            sm = mixed_similarity(gx, gb, gc, gx, gb, gc)
            norms = np.linalg.norm(gx, axis=1)
            den = norms[:, None] * norms[None, :]
            co = np.divide(gx @ gx.T, den, out=np.zeros_like(sm), where=den > 0)
            disc = np.mean(np.column_stack((gb, gc))[:, None, :] ==
                            np.column_stack((gb, gc))[None, :, :], axis=2)
            up = np.triu_indices(len(idx), 1)
            all_pair_sims.extend(sm[up].tolist())
            all_pair_cos.extend(co[up].tolist())
            all_pair_disc.extend(disc[up].tolist())
        if len(idx) > capacity:
            violations.append({"group_id": int(gid), "members": int(len(idx))})
        if not metrics["complete_link_valid"]:
            violations.append({"group_id": int(gid), "members": int(len(idx)),
                               "complete_link_violating_pairs": metrics["violating_pairs"],
                               "minimum_similarity": metrics["complete_link_min"]})

    sizes = np.asarray(group_sizes, dtype=np.int64)
    result = {
        "status": "completed",
        "label_blind": True,
        "labels_read": False,
        "input_nodes": int(len(labels)),
        "groups": int(len(groups)),
        "assigned_nodes": int(len(labels)),
        "unassigned_nodes": 0,
        "group_member_count": {
            "min": int(sizes.min()), "median": float(np.median(sizes)),
            "mean": float(sizes.mean()), "p95": float(np.percentile(sizes, 95)),
            "max": int(sizes.max()),
        },
        "singleton_groups": int(singleton_count),
        "capacity": int(capacity),
        "capacity_valid": bool(sizes.max() <= capacity),
        "capacity_violations": int(np.count_nonzero(sizes > capacity)),
        "complete_link_threshold": float(threshold),
        "complete_link_valid": all(g["complete_link_valid"] for g in groups),
        "complete_link_violating_groups": int(sum(not g["complete_link_valid"] for g in groups)),
        "complete_link_violating_pairs": int(sum(g["violating_pairs"] for g in groups)),
        "within_group_pair_similarity": _summary(np.asarray(all_pair_sims)),
        "within_group_continuous_cosine": _summary(np.asarray(all_pair_cos)),
        "within_group_discrete_agreement": _summary(np.asarray(all_pair_disc)),
        "group_profiles": groups,
        "graph_structure": _graph_metrics(graph_path, labels) if graph_path else None,
        "source_assignment_parameters": {
            "acceptance_threshold": report.get("acceptance_threshold"),
            "scrac_lambda": report.get("scrac_lambda"),
            "complete_link_threshold": report.get("complete_link_threshold"),
            "scrac_capacity": report.get("scrac_capacity"),
            "input_unit": report.get("input_unit"),
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=Path, required=True)
    p.add_argument("--assignment-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--graph", type=Path, default=None,
                   help="optional original adjacency.npz for graph-preservation metrics")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--capacity", type=int, default=200)
    args = p.parse_args()
    result = evaluate(args.nodes, args.assignment_report, args.threshold, args.capacity, args.graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "group_profiles"}, indent=2))


if __name__ == "__main__":
    main()
