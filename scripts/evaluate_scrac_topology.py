"""Label-blind topology-preservation audit for SCRAC graph partitions.

Builds the weighted quotient graph from the original sparse adjacency and a
node-level SCRAC assignment. Reports edge/weight conservation, degree-volume
conservation, partition modularity/conductance, connectivity distortion and
low-end normalized-Laplacian spectra. Does not read labels or modify inputs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "min": None, "median": None, "mean": None,
                "p95": None, "max": None}
    return {
        "count": int(values.size), "min": float(values.min()),
        "median": float(np.median(values)), "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)), "max": float(values.max()),
    }


def _load_assignment(path: Path, n: int) -> tuple[np.ndarray, dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if "assignment" not in report:
        raise ValueError("assignment report has no 'assignment' field")
    labels = np.asarray(report["assignment"], dtype=np.int64)
    if labels.shape != (n,) or np.any(labels < 0):
        raise ValueError("assignment must contain exactly one nonnegative group id per node")
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64), report


def _validate_graph(graph: csr_matrix) -> csr_matrix:
    g = graph.tocsr().astype(np.float64)
    g.sum_duplicates()
    g.eliminate_zeros()
    if g.shape[0] != g.shape[1] or not g.shape[0]:
        raise ValueError("adjacency must be non-empty and square")
    if not np.isfinite(g.data).all() or np.any(g.data < 0):
        raise ValueError("adjacency weights must be finite and nonnegative")
    diff = g - g.T
    if diff.nnz and float(np.max(np.abs(diff.data))) > 1e-10:
        raise ValueError("adjacency must be symmetric")
    if np.any(g.diagonal() != 0):
        raise ValueError("original adjacency must not contain self-loops")
    return g


def _spectral(g: csr_matrix, k: int) -> dict:
    n = g.shape[0]
    if n < 2 or k < 1:
        return {"status": "skipped", "reason": "graph has fewer than two nodes or spectral_k < 1",
                "eigenvalues": [], "zero_eigenvalue_count": None}
    ln = laplacian(g, normed=True).tocsr()
    try:
        if n <= 512:
            # Dense symmetric eigensolver supports k == n, including a
            # two-node quotient graph. Do not silently request zero values
            # for n=2 or n=3.
            kk = min(int(k), n)
            vals = np.linalg.eigvalsh(ln.toarray())[:kk]
        else:
            # ARPACK requires k < n for a sparse matrix.
            kk = min(int(k), n - 1)
            vals = eigsh(ln, k=kk, which="SM", return_eigenvectors=False,
                         tol=1e-5, maxiter=max(5000, n * 2))
            vals = np.sort(np.real(vals))
        vals = np.clip(np.real(vals), 0.0, 2.0)
        return {
            "status": "completed", "requested_k": int(k), "computed_k": int(len(vals)),
            "eigenvalues": vals.tolist(),
            "zero_eigenvalue_count": int(np.count_nonzero(vals < 1e-7)),
            "lambda_2": float(vals[1]) if len(vals) > 1 else None,
            "lambda_max_in_reported_range": float(vals[-1]) if len(vals) else None,
        }
    except ArpackNoConvergence as exc:
        vals = np.sort(np.real(exc.eigenvalues)) if exc.eigenvalues is not None else np.array([])
        vals = np.clip(vals, 0.0, 2.0)
        return {
            "status": "partial_convergence", "requested_k": int(k),
            "computed_k": int(len(vals)), "eigenvalues": vals.tolist(),
            "zero_eigenvalue_count": int(np.count_nonzero(vals < 1e-7)),
            "lambda_2": float(vals[1]) if len(vals) > 1 else None,
            "warning": "ARPACK did not converge fully; treat spectral comparison as incomplete.",
        }


def evaluate_topology(graph_path: Path, assignment_report: Path,
                      spectral_k: int = 20) -> dict:
    started = time.perf_counter()
    g = _validate_graph(load_npz(graph_path))
    labels, source_report = _load_assignment(assignment_report, g.shape[0])
    n = g.shape[0]
    n_groups = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=n_groups).astype(np.int64)

    coo = g.tocoo()
    upper = coo.row < coo.col
    u, v, w = coo.row[upper], coo.col[upper], coo.data[upper]
    total_weight = float(w.sum())
    total_edges = int(len(w))
    same = labels[u] == labels[v]
    internal_weight = float(w[same].sum())
    external_weight = float(w[~same].sum())
    internal_edges = int(same.sum())
    external_edges = int((~same).sum())

    # Weighted quotient Q = H.T A H. Its diagonal stores twice the
    # internal undirected edge weight; row sums exactly conserve group volume.
    q = csr_matrix((coo.data, (labels[coo.row], labels[coo.col])),
                   shape=(n_groups, n_groups), dtype=np.float64)
    q.sum_duplicates()
    q.eliminate_zeros()
    qdiag = q.diagonal()
    q_off = q.copy()
    q_off.setdiag(0.0)
    q_off.eliminate_zeros()
    q_upper = q_off.tocoo()
    q_edge_mask = q_upper.row < q_upper.col
    quotient_edges = int(np.count_nonzero(q_edge_mask))
    quotient_weight = float(q_upper.data[q_edge_mask].sum())

    original_degree = np.asarray(g.sum(axis=1)).ravel()
    original_volume = np.bincount(labels, weights=original_degree, minlength=n_groups)
    quotient_volume = np.asarray(q.sum(axis=1)).ravel()
    volume_error = np.abs(original_volume - quotient_volume)
    group_internal = np.bincount(labels[u[same]], weights=w[same], minlength=n_groups)
    group_external = np.bincount(labels[u[~same]], weights=w[~same], minlength=n_groups)
    # Count each cut edge once at each endpoint's group.
    np.add.at(group_external, labels[v[~same]], w[~same])
    conductance = np.divide(group_external, original_volume,
                            out=np.zeros_like(group_external), where=original_volume > 0)

    m2 = 2.0 * total_weight
    if m2 > 0:
        modularity = float(np.sum(group_internal / total_weight -
                                  (original_volume / m2) ** 2))
    else:
        modularity = None

    original_cc_count, original_cc = connected_components(g, directed=False)
    group_cc_counts = np.zeros(n_groups, dtype=np.int64)
    for gid in range(n_groups):
        group_cc_counts[gid] = np.unique(original_cc[labels == gid]).size
    mixed_cc_groups = int(np.count_nonzero(group_cc_counts > 1))
    q_cc_count, _ = connected_components(q, directed=False)

    original_spec = _spectral(g, spectral_k)
    quotient_spec = _spectral(q, spectral_k)
    spectral_delta = None
    if original_spec["eigenvalues"] and quotient_spec["eigenvalues"]:
        count = min(len(original_spec["eigenvalues"]), len(quotient_spec["eigenvalues"]))
        a = np.asarray(original_spec["eigenvalues"][:count])
        b = np.asarray(quotient_spec["eigenvalues"][:count])
        spectral_delta = {
            "compared_eigenvalues": int(count),
            "mean_absolute_error": float(np.mean(np.abs(a - b))),
            "max_absolute_error": float(np.max(np.abs(a - b))),
            "original_minus_quotient": (a - b).tolist(),
            "interpretation": "Diagnostic only: original and quotient operators have different dimensions and bases; this is not an isospectrality claim.",
        }

    result = {
        "status": "completed",
        "label_blind": True,
        "labels_read": False,
        "input_nodes": int(n),
        "groups": int(n_groups),
        "group_size": {
            "min": int(sizes.min()), "median": float(np.median(sizes)),
            "mean": float(sizes.mean()), "p95": float(np.percentile(sizes, 95)),
            "max": int(sizes.max()),
        },
        "original_graph": {
            "undirected_edges": total_edges, "weight_sum": total_weight,
            "connected_components": int(original_cc_count),
        },
        "quotient_graph": {
            "groups": int(n_groups), "undirected_inter_group_edges": quotient_edges,
            "inter_group_weight_sum": quotient_weight,
            "nonzero_diagonal_groups": int(np.count_nonzero(qdiag)),
            "diagonal_internal_weight_sum": float(qdiag.sum() / 2.0),
            "connected_components": int(q_cc_count),
        },
        "edge_and_weight_conservation": {
            "internal_edges": internal_edges, "external_edges": external_edges,
            "internal_edge_fraction": float(internal_edges / total_edges) if total_edges else None,
            "external_edge_fraction": float(external_edges / total_edges) if total_edges else None,
            "internal_weight": internal_weight, "external_weight": external_weight,
            "internal_weight_fraction": float(internal_weight / total_weight) if total_weight else None,
            "external_weight_fraction": float(external_weight / total_weight) if total_weight else None,
            "quotient_external_weight_matches_original_cut": bool(
                np.isclose(quotient_weight, external_weight, rtol=1e-9, atol=1e-9)),
        },
        "degree_volume_preservation": {
            "max_absolute_group_volume_error": float(volume_error.max(initial=0.0)),
            "mean_absolute_group_volume_error": float(volume_error.mean()),
            "exact_within_tolerance": bool(np.allclose(original_volume, quotient_volume,
                                                       rtol=1e-9, atol=1e-9)),
            "original_group_volume": original_volume.tolist(),
            "quotient_group_volume": quotient_volume.tolist(),
        },
        "partition_quality": {
            "weighted_modularity": modularity,
            "group_conductance": _summary(conductance),
            "groups_with_zero_volume": int(np.count_nonzero(original_volume == 0)),
            "groups_spanning_multiple_original_connected_components": mixed_cc_groups,
            "max_original_components_per_group": int(group_cc_counts.max(initial=0)),
            "original_component_count": int(original_cc_count),
            "quotient_component_count": int(q_cc_count),
            "warning": "A SCRAC group containing nodes from multiple original connected components can create artificial connectivity after contraction.",
        },
        "spectral_preservation": {
            "original_normalized_laplacian": original_spec,
            "quotient_normalized_laplacian": quotient_spec,
            "low_spectrum_comparison": spectral_delta,
        },
        "source_assignment_parameters": {
            "acceptance_threshold": source_report.get("acceptance_threshold"),
            "scrac_lambda": source_report.get("scrac_lambda"),
            "complete_link_threshold": source_report.get("complete_link_threshold"),
            "scrac_capacity": source_report.get("scrac_capacity"),
            "input_unit": source_report.get("input_unit"),
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "Spectral comparison is a diagnostic between different-size operators, not a proof of spectral equivalence.",
            "This report evaluates a fixed partition; it does not modify SCRAC or produce a final hierarchy.",
            "No attack/normal labels, mapping source paths, or raw flow records are read.",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--assignment-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spectral-k", type=int, default=20)
    args = parser.parse_args()
    result = evaluate_topology(args.graph, args.assignment_report, args.spectral_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    printable = {k: v for k, v in result.items()
                 if k not in {"degree_volume_preservation", "spectral_preservation"}}
    print(json.dumps(printable, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
