"""Convert independently processed chunk-level SCRAC outputs into temporal atoms.

Manifest format:
{"chunks":[{"chunk_id":"2026-06-01-c000","period":"2026-06-01",
"nodes":".../nodes.npz","scrac":".../scrac.npz"}]}
Each nodes.npz contains x,b,c; each scrac.npz contains assignment,component_mass.
Labels and source partition metadata are deliberately not accepted or read.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def _mode(values):
    out = []
    for col in values.T:
        unique, counts = np.unique(col, return_counts=True)
        out.append(unique[np.flatnonzero(counts == counts.max())[0]])
    return np.asarray(out, dtype=values.dtype)


def _chunk_atoms(nodes_path, scrac_path):
    with np.load(nodes_path, allow_pickle=False) as nf, np.load(scrac_path, allow_pickle=False) as sf:
        x = np.asarray(nf["x"], dtype=np.float64)
        b = np.asarray(nf["b"])
        c = np.asarray(nf["c"])
        assignment = np.asarray(sf["assignment"], dtype=np.int64)
        mass = np.asarray(sf["component_mass"], dtype=np.float64)
    n = len(x)
    if x.ndim != 2 or b.ndim != 2 or c.ndim != 2 or len(b) != n or len(c) != n:
        raise ValueError("node feature arrays must be aligned 2D matrices")
    if assignment.shape != (n,) or mass.shape != (n,):
        raise ValueError("SCRAC assignment and component_mass must align one-to-one with nodes")
    if n == 0 or np.any(assignment < 0) or not np.isfinite(x).all():
        raise ValueError("empty nodes, negative assignments, or non-finite features")
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("component_mass must be finite and positive")
    group_ids = np.unique(assignment)
    if not np.array_equal(group_ids, np.arange(len(group_ids))):
        raise ValueError("SCRAC group ids must be contiguous from zero")
    out_x, out_b, out_c, out_mass = [], [], [], []
    for gid in group_ids:
        idx = np.flatnonzero(assignment == gid)
        weights = mass[idx]
        centroid = np.average(x[idx], axis=0, weights=weights)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        out_x.append(centroid)
        out_b.append(_mode(b[idx]))
        out_c.append(_mode(c[idx]))
        out_mass.append(float(weights.sum()))
    return (np.asarray(out_x, dtype=np.float64),
            np.asarray(out_b, dtype=b.dtype).reshape(len(group_ids), b.shape[1]),
            np.asarray(out_c, dtype=c.dtype).reshape(len(group_ids), c.shape[1]),
            np.asarray(out_mass, dtype=np.float64),
            {"input_nodes": int(n), "scrac_groups": int(len(group_ids)),
             "represented_raw_rows": float(mass.sum()),
             "assignment_complete": bool(np.all(np.bincount(assignment) > 0))})


def build_atoms(manifest_path, output_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("manifest must contain a non-empty chunks list")
    base = manifest_path.parent
    xs, bs, cs, masses, periods, chunk_ids = [], [], [], [], [], []
    reports, seen = [], set()
    for item in chunks:
        chunk_id = str(item["chunk_id"])
        period = str(item["period"])
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        seen.add(chunk_id)
        if len(period) != 10 or period[4] != "-" or period[7] != "-":
            raise ValueError(f"period must be YYYY-MM-DD: {period}")
        nodes_path = Path(item["nodes"])
        scrac_path = Path(item["scrac"])
        if not nodes_path.is_absolute():
            nodes_path = base / nodes_path
        if not scrac_path.is_absolute():
            scrac_path = base / scrac_path
        x, b, c, mass, stats = _chunk_atoms(nodes_path, scrac_path)
        xs.append(x); bs.append(b); cs.append(c); masses.append(mass)
        periods.extend([period] * len(mass))
        chunk_ids.extend([chunk_id] * len(mass))
        reports.append({"chunk_id": chunk_id, "period": period, **stats})
    if len({a.shape[1] for a in bs}) != 1 or len({a.shape[1] for a in cs}) != 1 or len({a.shape[1] for a in xs}) != 1:
        raise ValueError("feature dimensions must be identical across chunks")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, x=np.concatenate(xs), b=np.concatenate(bs),
                        c=np.concatenate(cs), mass=np.concatenate(masses),
                        period=np.asarray(periods, dtype="U10"),
                        chunk_id=np.asarray(chunk_ids, dtype=str))
    result = {"status": "completed", "label_blind": True, "labels_read": False,
              "chunk_count": len(chunks), "temporal_atom_count": int(sum(len(x) for x in xs)),
              "represented_raw_rows": float(sum(m.sum() for m in masses)),
              "chunks": reports, "output": str(output_path)}
    output_path.with_suffix(".report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_atoms(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
