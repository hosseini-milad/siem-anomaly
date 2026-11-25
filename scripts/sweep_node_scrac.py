"""One-pass SC-FAGC + multi-configuration SCRAC sweep (strictly label-blind).

SC-FAGC anchors/transport are computed once. SCRAC is then rerun with a small
predeclared grid of acceptance thresholds and cubic size penalties. Mapping
files are read only through their integer 'node' column to count raw-row mass.
No source paths, partition labels, or class labels are accessed here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from graph.node_scrac import load_node_mass
from graph.sc_fagc import (
    build_bipartite,
    expand_anchors,
    sc_fagc,
    scrac_assign,
    select_anchor_indices,
)


def parse_floats(value: str, name: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from exc
    if not values or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError(f"{name} must contain finite values")
    return values


def run_sweep(args: argparse.Namespace) -> list[dict]:
    started = time.perf_counter()
    with np.load(args.nodes, allow_pickle=False) as f:
        x, b, c = (np.asarray(f[k]) for k in ("x", "b", "c"))
    if x.ndim != 2 or b.ndim != 2 or c.ndim != 2 or not (len(x) == len(b) == len(c)):
        raise ValueError("node feature arrays are not aligned")

    mass, mapping_files = load_node_mass(args.mapping_dir, len(x))
    n = len(x)
    n_anchors = min(n, max(args.anchors or int(np.ceil(.05 * n)), args.clusters))
    n_clusters = min(n, args.clusters)

    # One shared, deterministic SC-FAGC pass for all SCRAC configurations.
    initial = select_anchor_indices(x, b, c, n_anchors, seed=42)
    anchors, expansion = expand_anchors(x, b, c, initial, threshold=args.anchor_threshold)
    B = build_bipartite(x, b, c, anchors, k=args.anchor_k, threshold=args.anchor_threshold)
    anchor_mass = np.asarray(B.T @ mass).ravel()
    active = anchor_mass > 0
    B = B[:, active].tocsr()
    anchors = anchors[active]
    anchor_mass = anchor_mass[active]
    if not np.allclose(np.asarray(B.sum(axis=1)).ravel(), 1.0):
        raise ValueError("bipartite rows do not conserve unit assignment mass")
    if not np.isclose(anchor_mass.sum(), mass.sum()):
        raise ValueError("anchor transport does not conserve raw-row mass")

    n_clusters = min(n_clusters, len(anchors))
    target = float(mass.sum() / n_clusters)
    lower = .4 * target
    upper = max(target, float(args.capacity))
    U, opt = sc_fagc(
        B, anchor_mass, n_clusters, lower, upper,
        epsilon=args.epsilon, max_iter=args.max_iter,
        sinkhorn_iter=args.sinkhorn_iter,
    )
    if not opt.get("converged", False):
        raise RuntimeError("SC-FAGC did not converge; SCRAC sweep aborted")

    probabilities = np.asarray(B @ U)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_components = args.output_dir / "input_components_identity.npz"
    np.savez_compressed(
        input_components,
        component_id=np.arange(n, dtype=np.int32),
        raw_rows=mass.astype(np.int64),
    )

    outputs = []
    for threshold in args.acceptance_thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("acceptance thresholds must be in [0, 1]")
        for lam in args.lambdas:
            if lam < 0:
                raise ValueError("lambda values must be nonnegative")
            run_started = time.perf_counter()
            labels, scrac = scrac_assign(
                x, b, c, mass, probabilities, anchors, U,
                threshold=args.complete_link_threshold,
                capacity=args.capacity,
                lambda_size=lam,
                acceptance_threshold=threshold,
            )
            name = f"accept_{threshold:.3f}_lambda_{lam:.3f}".replace(".", "p")
            report = {
                "status": "completed",
                "components": n,
                "represented_raw_rows": int(mass.sum()),
                "anchors": int(len(anchors)),
                "clusters_effective": int(n_clusters),
                "anchor_threshold": float(args.anchor_threshold),
                "anchor_expansion": expansion,
                "optimization": {
                    "iterations": opt.get("iterations"),
                    "converged": opt.get("converged"),
                    "sinkhorn_iterations": opt.get("sinkhorn_iterations"),
                    "sinkhorn_converged": opt.get("sinkhorn_converged"),
                    "row_relative_error": opt.get("row_relative_error"),
                    "column_relative_error": opt.get("column_relative_error"),
                },
                "scrac_assignment": scrac,
                "assignment": labels.astype(int).tolist(),
                "input_unit": "one_graph_representative_node",
                "label_blind": True,
                "mapping_files_used_for_mass_only": mapping_files,
                "mapping_source_labels_read": False,
                "input_components_artifact": str(input_components),
                "complete_link_threshold": float(args.complete_link_threshold),
                "acceptance_threshold": float(threshold),
                "scrac_lambda": float(lam),
                "scrac_capacity": int(args.capacity),
                "runtime_seconds_scrac": round(time.perf_counter() - run_started, 3),
                "purpose": "Label-blind SCRAC sensitivity experiment; labels are reserved for a separate post-hoc audit.",
            }
            out = args.output_dir / f"{name}.json"
            out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            outputs.append({
                "report": str(out),
                "groups": scrac["groups"],
                "singletons": scrac["singleton_groups"],
                "max_group_size": scrac["max_group_size"],
                "capacity_violations": scrac["capacity_violations"],
                "scrac_runtime_seconds": report["runtime_seconds_scrac"],
                "acceptance_threshold": threshold,
                "lambda_size": lam,
            })

    summary = {
        "status": "completed",
        "label_blind": True,
        "mapping_source_labels_read": False,
        "shared_sc_fagc_runtime_included": True,
        "total_runtime_seconds": round(time.perf_counter() - started, 3),
        "sc_fagc_iterations": opt.get("iterations"),
        "sc_fagc_converged": opt.get("converged"),
        "runs": outputs,
    }
    (args.output_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return outputs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=Path, required=True)
    p.add_argument("--mapping-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--anchors", type=int, default=None)
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--capacity", type=int, default=200)
    p.add_argument("--anchor-k", type=int, default=5)
    p.add_argument("--anchor-threshold", type=float, default=.85)
    p.add_argument("--complete-link-threshold", type=float, default=.85)
    p.add_argument("--acceptance-thresholds", type=lambda s: parse_floats(s, "acceptance thresholds"), default=[.75, .78, .80])
    p.add_argument("--lambdas", type=lambda s: parse_floats(s, "lambdas"), default=[.20, .30])
    p.add_argument("--epsilon", type=float, default=.01)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--sinkhorn-iter", type=int, default=2000)
    args = p.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
