"""Post-hoc attack/normal composition audit. Labels are never passed to clustering."""
import argparse
import json
import re
from pathlib import Path
import numpy as np


def _partition_labels(source):
    paths = np.char.replace(np.asarray(source).astype(str), chr(92), "/")
    attacks = np.char.find(paths, "/type=attacks/") >= 0
    normal = np.char.find(paths, "/type=normal/") >= 0
    valid = attacks ^ normal
    if not np.all(valid):
        bad = paths[~valid][:5].tolist()
        raise ValueError(f"unrecognized source partition paths: {bad}")
    return attacks.astype(np.int8)


def audit(mapping_dir, components_path, assignment_report, output_path=None):
    mapping_dir = Path(mapping_dir)
    with np.load(components_path, allow_pickle=False) as f:
        node_to_component = np.asarray(f["component_id"], dtype=np.int64)
        component_raw_rows = np.asarray(f["raw_rows"], dtype=np.int64)
        stored_attack_rows = np.asarray(f["attack_rows"], dtype=np.int64) if "attack_rows" in f.files else None

    report = json.loads(Path(assignment_report).read_text(encoding="utf-8"))
    assignment = np.asarray(report.get("assignment", []), dtype=np.int64)
    n_nodes = len(node_to_component)
    n_components = len(component_raw_rows)
    if node_to_component.shape != (n_nodes,) or component_raw_rows.shape != (n_components,):
        raise ValueError("invalid component artifact dimensions")
    if len(assignment) != n_components:
        raise ValueError(f"assignment length {len(assignment)} != initial components {n_components}")
    if np.any(node_to_component < 0) or np.any(node_to_component >= n_components):
        raise ValueError("component_id contains out-of-range component indices")
    if np.any(assignment < 0):
        raise ValueError("negative/unassigned final group labels")
    if np.any(component_raw_rows < 0):
        raise ValueError("negative raw_rows in components artifact")

    expected_per_component = np.bincount(
        node_to_component, weights=np.ones(n_nodes, dtype=np.int64), minlength=n_components
    )
    # Raw rows represented by each graph node are not stored in nodes.npz.
    # The components artifact stores the sum per connected component, so compare
    # mapping-derived raw-row totals to these values after aggregation below.
    group_count = int(assignment.max()) + 1 if assignment.size else 0
    total_attack = np.zeros(group_count, dtype=np.int64)
    total_normal = np.zeros(group_count, dtype=np.int64)
    mapped_rows_per_node = np.zeros(n_nodes, dtype=np.int64)
    source_files = set()\n    source_partition_counts = {"attacks": 0, "normal": 0}\n    source_days = set()
    mapping_files = sorted(mapping_dir.glob("mapping_*.npz"))
    if not mapping_files:
        raise FileNotFoundError(f"no mapping_*.npz files found in {mapping_dir}")

    mapped_total = 0
    for path in mapping_files:
        with np.load(path, allow_pickle=False) as f:
            node = np.asarray(f["node"], dtype=np.int64)
            labels = _partition_labels(f["source"])
            if node.shape != labels.shape:
                raise ValueError(f"{path.name}: node/source length mismatch")
            if np.any(node < 0) or np.any(node >= n_nodes):
                raise ValueError(f"{path.name}: node index out of range")
            if len(labels) and (assignment[node_to_component[node]].max(initial=-1) >= group_count):
                raise ValueError(f"{path.name}: invalid final group mapping")
            comp = node_to_component[node]
            group = assignment[comp]
            np.add.at(total_attack, group, labels)
            np.add.at(total_normal, group, 1 - labels)
            np.add.at(mapped_rows_per_node, node, 1)
            mapped_total += len(node)
            source_values = np.asarray(f["source"]).astype(str)\n            source_files.update(source_values.tolist())\n            for value in source_values:\n                match = re.search(r"date=(\\d{4}-\\d{2}-\\d{2})", value.replace(chr(92), "/"))\n                if match:\n                    source_days.add(match.group(1))

    mapped_per_component = np.bincount(
        node_to_component, weights=mapped_rows_per_node, minlength=n_components
    ).astype(np.int64)
    expected_raw = component_raw_rows.astype(np.int64)
    if not np.array_equal(mapped_per_component, expected_raw):
        bad = np.flatnonzero(mapped_per_component != expected_raw)[:10]
        raise ValueError(
            "mapping/raw_rows mismatch for initial components; "
            f"first component ids: {bad.tolist()}, "
            f"mapping totals: {mapped_per_component[bad].tolist()}, "
            f"artifact raw_rows: {expected_raw[bad].tolist()}"
        )

    if stored_attack_rows is not None:
        stored_attack_rows = stored_attack_rows.astype(np.int64)
        mapped_attack_by_component = np.zeros(n_components, dtype=np.int64)
        # Re-read mappings only for the independent consistency cross-check.
        for path in mapping_files:
            with np.load(path, allow_pickle=False) as f:
                node = np.asarray(f["node"], dtype=np.int64)
                labels = _partition_labels(f["source"])
                np.add.at(mapped_attack_by_component, node_to_component[node], labels)
        if not np.array_equal(mapped_attack_by_component, stored_attack_rows):
            bad = np.flatnonzero(mapped_attack_by_component != stored_attack_rows)[:10]
            raise ValueError(
                "stored attack_rows disagree with source partition mapping; "
                f"first component ids: {bad.tolist()}, "
                f"mapping labels: {mapped_attack_by_component[bad].tolist()}, "
                f"artifact attack_rows: {stored_attack_rows[bad].tolist()}"
            )

    total = total_attack + total_normal
    if np.any(total == 0):
        raise ValueError("one or more final groups have no mapped raw records")
    rows = []
    for gid in range(group_count):
        a, n = int(total_attack[gid]), int(total_normal[gid])
        t = a + n
        minority = min(a, n)
        rows.append({
            "group_id": gid, "attack_rows": a, "normal_rows": n, "total_rows": t,
            "attack_pct": round(100 * a / t, 4),
            "normal_pct": round(100 * n / t, 4),
            "impurity_pct": round(100 * minority / t, 4),
            "purity_pct": round(100 * max(a, n) / t, 4),
            "class": "pure_attack" if n == 0 else "pure_normal" if a == 0 else "mixed"
        })

    mixed = [r for r in rows if r["class"] == "mixed"]
    pure_attack = [r for r in rows if r["class"] == "pure_attack"]
    pure_normal = [r for r in rows if r["class"] == "pure_normal"]
    imp = np.asarray([r["impurity_pct"] for r in rows], dtype=float)
    total_rows = int(total.sum())
    weighted_impurity = float(np.minimum(total_attack, total_normal).sum() / total_rows * 100)
    summary = {
        "label_usage": "post-hoc evaluation only; not used in graph construction, anchor selection, or SCRAC assignment",
        "mapping_files": len(mapping_files),
        "source_paths": len(source_files),\n        "source_days": sorted(source_days),\n        "source_partition_raw_rows": source_partition_counts,
        "graph_nodes": n_nodes,
        "initial_components": n_components,
        "final_groups": group_count,
        "mapped_raw_rows": mapped_total,
        "attack_rows": int(total_attack.sum()),
        "normal_rows": int(total_normal.sum()),
        "attack_pct_overall": round(100 * int(total_attack.sum()) / total_rows, 4),
        "normal_pct_overall": round(100 * int(total_normal.sum()) / total_rows, 4),
        "pure_attack_groups": len(pure_attack),
        "pure_normal_groups": len(pure_normal),
        "mixed_groups": len(mixed),
        "mixed_group_pct": round(100 * len(mixed) / group_count, 4),
        "pure_group_pct": round(100 * (len(pure_attack) + len(pure_normal)) / group_count, 4),
        "macro_mean_impurity_pct": round(float(imp.mean()), 4),
        "raw_row_weighted_impurity_pct": round(weighted_impurity, 4),
        "max_group_impurity_pct": round(float(imp.max()), 4),
        "median_group_impurity_pct": round(float(np.median(imp)), 4),
        "all_attack_data_warning": bool(total_normal.sum() == 0),
        "all_normal_data_warning": bool(total_attack.sum() == 0),
        "groups": rows
    }
    result = {"summary": summary}
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping-dir", type=Path, required=True)
    p.add_argument("--components", type=Path, required=True)
    p.add_argument("--assignment-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.mapping_dir, a.components, a.assignment_report, a.output)
    print(json.dumps(result["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
