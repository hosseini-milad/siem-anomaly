"""Topology-preserving multilevel contraction of a weighted SCRAC quotient graph."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import connected_components


@dataclass
class HierarchyResult:
    levels: list[csr_matrix]
    child_to_parent: list[np.ndarray]
    component_ids: list[np.ndarray]
    base_group_sizes: list[np.ndarray]
    original_node_sizes: list[np.ndarray]
    manifest: dict[str, Any]


def _validate_graph(adjacency: csr_matrix) -> csr_matrix:
    g = csr_matrix(adjacency, dtype=np.float64)
    g.sum_duplicates()
    g.eliminate_zeros()
    if not g.shape[0] or g.shape[0] != g.shape[1]:
        raise ValueError("adjacency must be non-empty and square")
    if not np.isfinite(g.data).all() or np.any(g.data < 0):
        raise ValueError("adjacency weights must be finite and nonnegative")
    delta = g - g.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-10:
        raise ValueError("adjacency must be symmetric")
    if np.any(np.abs(g.diagonal()) > 1e-12):
        raise ValueError("original adjacency must not contain self-loops")
    return g


def _labels(values: np.ndarray, n: int, name: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.int64)
    if a.shape != (n,) or np.any(a < 0):
        raise ValueError(f"{name} must contain one nonnegative ID per node")
    return np.unique(a, return_inverse=True)[1].astype(np.int64)


def _quotient(g: csr_matrix, labels: np.ndarray) -> csr_matrix:
    n = int(labels.max()) + 1
    coo = g.tocoo()
    q = csr_matrix((coo.data, (labels[coo.row], labels[coo.col])),
                   shape=(n, n), dtype=np.float64)
    q.sum_duplicates()
    q.eliminate_zeros()
    return q


def _weight(g: csr_matrix) -> float:
    # Symmetric adjacency stores each undirected edge twice; quotient
    # diagonals store twice the internal undirected weight as well.
    return float(g.sum() / 2.0)


def _metrics(g: csr_matrix, cc: np.ndarray, base_size: np.ndarray,
             raw_size: np.ndarray) -> dict:
    off = g.copy()
    off.setdiag(0.0)
    off.eliminate_zeros()
    coo = off.tocoo()
    return {
        "nodes": int(g.shape[0]),
        "inter_cluster_edges": int(np.count_nonzero(coo.row < coo.col)),
        "self_loop_internal_weight": float(g.diagonal().sum() / 2.0),
        "inter_cluster_weight": float(_weight(g) - g.diagonal().sum() / 2.0),
        "total_undirected_weight": _weight(g),
        "degree_volume": float(g.sum()),
        "connected_components": int(connected_components(g, directed=False)[0]),
        "isolated_nodes": int(np.count_nonzero(np.diff(off.indptr) == 0)),
        "max_base_groups_per_node": int(base_size.max(initial=0)),
        "max_original_nodes_per_node": int(raw_size.max(initial=0)),
    }


def _matching(g: csr_matrix, cc: np.ndarray, min_affinity: float) -> np.ndarray:
    coo = g.tocoo()
    mask = (coo.row < coo.col) & (coo.data > 0)
    u, v, w = coo.row[mask], coo.col[mask], coo.data[mask]
    n = g.shape[0]
    if not len(w):
        return np.arange(n, dtype=np.int64)
    volume = np.asarray(g.sum(axis=1)).ravel()
    denom = np.sqrt(volume[u] * volume[v])
    affinity = np.divide(w, denom, out=np.zeros_like(w), where=denom > 0)
    ok = (cc[u] == cc[v]) & (affinity >= min_affinity)
    u, v, affinity = u[ok], v[ok], affinity[ok]
    order = np.lexsort((v, u, -affinity))
    partner = np.full(n, -1, dtype=np.int64)
    for k in order:
        a, b = int(u[k]), int(v[k])
        if partner[a] < 0 and partner[b] < 0:
            partner[a], partner[b] = b, a
    mapping = np.full(n, -1, dtype=np.int64)
    pid = 0
    for i in range(n):
        if mapping[i] >= 0:
            continue
        mapping[i] = pid
        if partner[i] >= 0:
            mapping[partner[i]] = pid
        pid += 1
    return mapping


def build_hierarchy(adjacency: csr_matrix, base_assignment: np.ndarray,
                    original_component_ids: np.ndarray | None = None, *,
                    max_levels: int = 6, min_affinity: float = 0.0) -> HierarchyResult:
    """Create SCRAC quotient level 0 and deterministic binary contraction levels."""
    started = time.perf_counter()
    g = _validate_graph(adjacency)
    if max_levels < 1 or not 0 <= min_affinity <= 1:
        raise ValueError("max_levels must be >=1 and min_affinity in [0,1]")
    labels = _labels(base_assignment, g.shape[0], "base_assignment")
    if original_component_ids is None:
        _, cc = connected_components(g, directed=False)
    else:
        cc = _labels(original_component_ids, g.shape[0], "original_component_ids")
    n_base = int(labels.max()) + 1
    base_cc = np.empty(n_base, dtype=np.int64)
    base_sizes = np.bincount(labels, minlength=n_base).astype(np.int64)
    for gid in range(n_base):
        ids = np.unique(cc[labels == gid])
        if len(ids) != 1:
            raise ValueError(f"base SCRAC group {gid} spans multiple original connected components")
        base_cc[gid] = ids[0]

    q0 = _quotient(g, labels)
    levels, transitions = [q0], []
    comp_levels = [base_cc]
    base_sizes_levels = [np.ones(n_base, dtype=np.int64)]
    raw_sizes_levels = [base_sizes]
    original_weight, original_volume = _weight(g), float(g.sum())
    expected_cc = int(np.unique(cc).size)
    reports = [{"level": 0, "kind": "scrac_weighted_quotient",
                **_metrics(q0, base_cc, base_sizes_levels[0], raw_sizes_levels[0])}]
    stop = "max_levels_reached"

    for level in range(1, max_levels):
        current = levels[-1]
        mapping = _matching(current, comp_levels[-1], min_affinity)
        n_parent = int(mapping.max()) + 1
        if n_parent == current.shape[0]:
            stop = "no_eligible_adjacent_pairs"
            break
        parent_cc = np.empty(n_parent, dtype=np.int64)
        for pid in range(n_parent):
            ids = np.unique(comp_levels[-1][mapping == pid])
            if len(ids) != 1:
                raise AssertionError("cross-component contraction attempted")
            parent_cc[pid] = ids[0]
        base_sizes = np.bincount(mapping, weights=base_sizes_levels[-1]).astype(np.int64)
        raw_sizes = np.bincount(mapping, weights=raw_sizes_levels[-1]).astype(np.int64)
        coo = current.tocoo()
        nxt = csr_matrix((coo.data, (mapping[coo.row], mapping[coo.col])),
                         shape=(n_parent, n_parent), dtype=np.float64)
        nxt.sum_duplicates()
        nxt.eliminate_zeros()
        if not np.isclose(_weight(nxt), original_weight, rtol=1e-9, atol=1e-9):
            raise AssertionError("undirected edge-weight conservation failed")
        if not np.isclose(float(nxt.sum()), original_volume, rtol=1e-9, atol=1e-9):
            raise AssertionError("degree-volume conservation failed")
        if connected_components(nxt, directed=False)[0] != expected_cc:
            raise AssertionError("connected-component count changed")
        transitions.append(mapping)
        levels.append(nxt)
        comp_levels.append(parent_cc)
        base_sizes_levels.append(base_sizes)
        raw_sizes_levels.append(raw_sizes)
        reports.append({"level": level, "kind": "greedy_maximal_matching_contraction",
                        "merged_pairs": int(current.shape[0] - n_parent),
                        "max_children_per_parent": 2,
                        **_metrics(nxt, parent_cc, base_sizes, raw_sizes)})
    else:
        stop = "max_levels_reached"

    manifest = {
        "schema_version": 1, "status": "completed", "label_blind": True,
        "labels_read": False, "algorithm": "deterministic_weighted_maximal_matching",
        "affinity": "inter_edge_weight / sqrt(child_volume_a * child_volume_b)",
        "min_affinity": float(min_affinity), "max_levels_requested": int(max_levels),
        "levels_built": len(levels), "stop_reason": stop,
        "original_nodes": int(g.shape[0]), "base_groups": int(q0.shape[0]),
        "original_connected_components": expected_cc,
        "original_undirected_weight": original_weight,
        "original_degree_volume": original_volume,
        "topology_preserved": True, "weight_conserved": True,
        "degree_volume_conserved": True, "level_metrics": reports,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "limitations": [
            "Hierarchy merges adjacent SCRAC groups; it does not re-evaluate feature Complete-Link similarity.",
            "Normalized edge affinity is a deterministic structural priority, not learned semantic similarity.",
            "Spectral distortion and downstream detection impact require separate evaluation."
        ]
    }
    return HierarchyResult(levels, transitions, comp_levels, base_sizes_levels,
                           raw_sizes_levels, manifest)


def save_hierarchy(result: HierarchyResult, output_prefix: Path) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = prefix.with_suffix(".npz"), prefix.with_suffix(".json")
    arrays = {}
    for i, g in enumerate(result.levels):
        arrays.update({
            f"level_{i}_data": g.data, f"level_{i}_indices": g.indices,
            f"level_{i}_indptr": g.indptr, f"level_{i}_shape": np.asarray(g.shape),
            f"level_{i}_component_id": result.component_ids[i],
            f"level_{i}_base_group_size": result.base_group_sizes[i],
            f"level_{i}_original_node_size": result.original_node_sizes[i],
        })
    for i, mapping in enumerate(result.child_to_parent):
        counts = np.bincount(mapping, minlength=int(mapping.max()) + 1)
        arrays[f"transition_{i}_child_to_parent"] = mapping
        arrays[f"transition_{i}_parent_child_offsets"] = np.r_[0, np.cumsum(counts)]
        arrays[f"transition_{i}_parent_children"] = np.argsort(mapping, kind="stable")
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(result.manifest, indent=2, allow_nan=False), encoding="utf-8")
    return npz_path, json_path


def load_hierarchy(npz_path: Path, manifest_path: Path | None = None) -> HierarchyResult:
    p = Path(npz_path)
    manifest = json.loads(Path(manifest_path or p.with_suffix(".json")).read_text(encoding="utf-8"))
    levels, transitions, comps, base_sizes, raw_sizes = [], [], [], [], []
    with np.load(p, allow_pickle=False) as f:
        i = 0
        while f"level_{i}_data" in f.files:
            shape = tuple(int(x) for x in f[f"level_{i}_shape"])
            levels.append(csr_matrix((f[f"level_{i}_data"], f[f"level_{i}_indices"],
                                      f[f"level_{i}_indptr"]), shape=shape))
            comps.append(f[f"level_{i}_component_id"].astype(np.int64))
            base_sizes.append(f[f"level_{i}_base_group_size"].astype(np.int64))
            raw_sizes.append(f[f"level_{i}_original_node_size"].astype(np.int64))
            i += 1
        for j in range(max(0, len(levels) - 1)):
            transitions.append(f[f"transition_{j}_child_to_parent"].astype(np.int64))
    if not levels:
        raise ValueError("hierarchy artifact contains no levels")
    return HierarchyResult(levels, transitions, comps, base_sizes, raw_sizes, manifest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--assignment", type=Path, required=True, help="SCRAC .npz with assignment array")
    p.add_argument("--components", type=Path, default=None, help="optional .npz with component_id")
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--max-levels", type=int, default=6)
    p.add_argument("--min-affinity", type=float, default=0.0)
    a = p.parse_args()
    with np.load(a.assignment, allow_pickle=False) as f:
        assignment = f["assignment"].astype(np.int64)
    component_ids = None
    if a.components:
        with np.load(a.components, allow_pickle=False) as f:
            key = "component_id" if "component_id" in f.files else f.files[0]
            component_ids = f[key].astype(np.int64)
    result = build_hierarchy(load_npz(a.graph), assignment, component_ids,
                             max_levels=a.max_levels, min_affinity=a.min_affinity)
    npz, report = save_hierarchy(result, a.output_prefix)
    print(json.dumps({"status": result.manifest["status"],
        "levels_built": result.manifest["levels_built"],
        "groups_per_level": [int(g.shape[0]) for g in result.levels],
        "topology_preserved": result.manifest["topology_preserved"],
        "weight_conserved": result.manifest["weight_conserved"],
        "degree_volume_conserved": result.manifest["degree_volume_conserved"],
        "stop_reason": result.manifest["stop_reason"], "npz": str(npz),
        "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
