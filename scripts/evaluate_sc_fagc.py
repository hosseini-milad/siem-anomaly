"""Independent audit of SCRAC member-count capacity and Complete-Link."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from graph.component_sc_fagc import aggregate_components
from graph.sc_fagc import mixed_similarity


def evaluate(nodes, components, report_path, threshold=0.85, capacity=200):
    started = time.perf_counter()
    x, b, c, mass = aggregate_components(nodes, components)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    labels = np.asarray(report.get("assignment", []), dtype=np.int64)
    if labels.shape != (len(x),) or np.any(labels < 0):
        raise ValueError("report assignment is missing, invalid, or not aligned with component data")
    if capacity < 1:
        raise ValueError("capacity must be positive")

    group_ids = np.unique(labels)
    sizes, masses, violations = [], [], []
    for gid in group_ids:
        idx = np.flatnonzero(labels == gid)
        sizes.append(int(len(idx)))
        masses.append(int(mass[idx].sum()))
        min_sim = 1.0
        for pos in range(len(idx)):
            if pos + 1 == len(idx):
                continue
            sims = mixed_similarity(
                x[idx[pos:pos + 1]], b[idx[pos:pos + 1]], c[idx[pos:pos + 1]],
                x[idx[pos + 1:]], b[idx[pos + 1:]], c[idx[pos + 1:]]
            ).ravel()
            if sims.size:
                min_sim = min(min_sim, float(sims.min()))
        if len(idx) > 1 and min_sim <= threshold:
            violations.append({"group": int(gid), "size": int(len(idx)), "minimum_similarity": min_sim})

    assigned_once = bool(len(labels) == len(x) and np.unique(labels).size == len(group_ids))
    max_size = max(sizes, default=0)
    out = {
        "components": int(len(x)),
        "raw_rows": int(mass.sum()),
        "groups": int(len(group_ids)),
        "component_group_size": {
            "min": int(min(sizes)), "median": float(np.median(sizes)),
            "mean": float(np.mean(sizes)), "p95": float(np.percentile(sizes, 95)),
            "max": int(max_size)
        },
        "raw_mass_group_size": {
            "min": int(min(masses)), "median": float(np.median(masses)),
            "mean": float(np.mean(masses)), "p95": float(np.percentile(masses, 95)),
            "max": int(max(masses))
        },
        "capacity": int(capacity),
        "groups_over_capacity": int(sum(v > capacity for v in sizes)),
        "groups_at_capacity": int(sum(v == capacity for v in sizes)),
        "capacity_violations": int(sum(v > capacity for v in sizes)),
        "capacity_valid": bool(max_size <= capacity),
        "groups_over_100_components": int(sum(v > 100 for v in sizes)),
        "groups_over_100_raw_rows": int(sum(v > 100 for v in masses)),
        "singleton_groups": int(sum(v == 1 for v in sizes)),
        "complete_link_threshold": float(threshold),
        "complete_link_violations": violations,
        "complete_link_valid": len(violations) == 0,
        "all_components_assigned_once": assigned_once,
        "anchor_expansion": report.get("anchor_expansion", {}),
        "anchors_initial": report.get("initial_anchors"),
        "anchors_after_expansion": report.get("anchors_after_expansion", report.get("anchors")),
        "zero_mass_anchors_removed": report.get("zero_mass_anchors_removed"),
        "runtime_audit_seconds": round(time.perf_counter() - started, 3)
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--capacity", type=int, default=200)
    args = parser.parse_args()
    result = evaluate(args.nodes, args.components, args.report, args.threshold, args.capacity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
