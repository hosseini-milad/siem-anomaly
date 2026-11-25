"""Temporal component hierarchy: chunk -> day -> month -> year.

This is deliberately different from graph contraction. Each input row is an
already-formed component descriptor emitted by one independent time chunk.
Components are re-compared within their parent calendar period and grouped
using label-blind complete-link similarity. Parent descriptors are aggregated
from their children and become the atoms of the next temporal level.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from graph.sc_fagc import mixed_similarity


LEVELS = ("day", "month", "year")


def _validate_atoms(x, b, c, mass, period, chunk_id):
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(b)
    c = np.asarray(c)
    mass = np.asarray(mass, dtype=np.float64)
    period = np.asarray(period).astype(str)
    chunk_id = np.asarray(chunk_id).astype(str)
    n = len(x)
    if x.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("x, b and c must be two-dimensional feature matrices")
    if not (len(b) == len(c) == len(mass) == len(period) == len(chunk_id) == n):
        raise ValueError("all atom arrays must have aligned rows")
    if n == 0 or not np.isfinite(x).all() or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("atoms must be non-empty with finite features and positive mass")
    if not all(re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}", value) for value in period):
        raise ValueError("period values must be ISO dates YYYY-MM-DD")
    if not all(value.strip() for value in chunk_id):
        raise ValueError("chunk_id cannot be empty")
    return x, b, c, mass, period, chunk_id


def _aggregate_descriptors(x, b, c, mass, groups):
    """Mass-weighted continuous centroid and deterministic categorical modes."""
    out_x, out_b, out_c, out_mass = [], [], [], []
    for members in groups:
        w = mass[members]
        centroid = np.average(x[members], axis=0, weights=w)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        out_x.append(centroid)
        def mode_matrix(a):
            vals = []
            for col in a[members].T:
                values, counts = np.unique(col, return_counts=True)
                vals.append(values[np.flatnonzero(counts == counts.max())[0]])
            return vals
        out_b.append(mode_matrix(b))
        out_c.append(mode_matrix(c))
        out_mass.append(float(w.sum()))
    return (np.asarray(out_x, dtype=np.float64),
            np.asarray(out_b, dtype=b.dtype).reshape(len(groups), b.shape[1]),
            np.asarray(out_c, dtype=c.dtype).reshape(len(groups), c.shape[1]),
            np.asarray(out_mass, dtype=np.float64))


def _complete_link_groups(x, b, c, mass, threshold, capacity, block_size):
    """Deterministic greedy complete-link grouping; no labels are accepted."""
    n = len(x)
    if n == 1:
        return [np.array([0], dtype=np.int64)], {"candidate_pairs": 0, "similarity_comparisons": 0}
    # Build threshold-qualified candidates in bounded blocks, not an n*n dense matrix.
    candidates = []
    comparisons = 0
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        sim = mixed_similarity(x[start:end], b[start:end], c[start:end], x, b, c)
        comparisons += sim.size
        for local, row in enumerate(sim):
            i = start + local
            eligible = np.flatnonzero(row[i + 1:] > threshold) + i + 1
            for j in eligible:
                candidates.append((float(row[j]), i, int(j)))
    candidates.sort(key=lambda z: (-z[0], z[1], z[2]))
    parent = np.arange(n, dtype=np.int64)
    members = {i: [i] for i in range(n)}
    totals = mass.astype(np.float64).copy()
    # Complete-link is checked against all cross-cluster pairs. Candidate edges
    # only propose merges; they never bypass the all-pairs constraint.
    accepted = 0
    for score, i, j in candidates:
        ri, rj = int(parent[i]), int(parent[j])
        if ri == rj or totals[ri] + totals[rj] > capacity:
            continue
        left, right = members[ri], members[rj]
        cross_min = 1.0
        for a0 in range(0, len(left), block_size):
            aa = np.asarray(left[a0:a0 + block_size], dtype=np.int64)
            ss = mixed_similarity(x[aa], b[aa], c[aa], x[right], b[right], c[right])
            cross_min = min(cross_min, float(ss.min()))
            if cross_min <= threshold:
                break
        if cross_min <= threshold:
            continue
        new_root = min(ri, rj)
        old_root = max(ri, rj)
        parent[parent == old_root] = new_root
        members[new_root] = left + right
        del members[old_root]
        totals[new_root] += totals[old_root]
        accepted += 1
    groups = [np.asarray(v, dtype=np.int64) for _, v in sorted(members.items())]
    return groups, {"candidate_pairs": len(candidates), "similarity_comparisons": comparisons,
                    "accepted_merges": accepted}


def build_temporal_hierarchy(x, b, c, mass, period, chunk_id, *,
                             similarity_threshold=0.85, max_component_mass=200,
                             block_size=128):
    """Build daily, monthly and yearly super-components from chunk components."""
    started = time.perf_counter()
    x, b, c, mass, period, chunk_id = _validate_atoms(x, b, c, mass, period, chunk_id)
    if not -1 <= similarity_threshold <= 1 or max_component_mass <= 0 or block_size <= 0:
        raise ValueError("invalid threshold, capacity or block size")

    current = {"x": x, "b": b, "c": c, "mass": mass,
               "period": period, "origin_count": np.ones(len(x), dtype=np.int64)}
    outputs = {}
    summaries = {}
    for level in LEVELS:
        parent_keys = (current["period"] if level == "day"
                       else np.asarray([str(p)[:7] for p in current["period"]]) if level == "month"
                       else np.asarray([str(p)[:4] for p in current["period"]]))
        next_rows = {k: [] for k in ("x", "b", "c", "mass", "period", "origin_count")}
        records = []
        for key in sorted(np.unique(parent_keys)):
            idx = np.flatnonzero(parent_keys == key)
            groups, stats = _complete_link_groups(
                current["x"][idx], current["b"][idx], current["c"][idx],
                current["mass"][idx], similarity_threshold, max_component_mass, block_size)
            gx, gb, gc, gm = _aggregate_descriptors(
                current["x"][idx], current["b"][idx], current["c"][idx], current["mass"][idx], groups)
            for local_id, members in enumerate(groups):
                global_members = idx[members]
                next_rows["x"].append(gx[local_id])
                next_rows["b"].append(gb[local_id])
                next_rows["c"].append(gc[local_id])
                next_rows["mass"].append(gm[local_id])
                next_rows["period"].append(str(key))
                next_rows["origin_count"].append(int(current["origin_count"][global_members].sum()))
                records.append({"period": str(key), "supercomponent_id": local_id,
                                "child_component_indices": global_members.astype(int).tolist(),
                                "child_components": int(len(members)),
                                "represented_raw_rows": float(gm[local_id]),
                                "origin_chunk_components": int(current["origin_count"][global_members].sum())})
            summaries[str(key)] = {"input_components": int(len(idx)),
                                   "output_supercomponents": int(len(groups)), **stats}
        outputs[level] = records
        current = {k: np.asarray(v) for k, v in next_rows.items()}
        if level == "day":
            current["period"] = np.asarray([p[:7] for p in current["period"]])
        elif level == "month":
            current["period"] = np.asarray([p[:4] for p in current["period"]])
        else:
            current["period"] = np.asarray(current["period"])
        summaries = {"periods": summaries, "total_input": int(len(x)),
                     "total_output": int(len(current["mass"]))}
        # Reset per-level period summary for next iteration.
        outputs[level + "_summary"] = summaries
        summaries = {}
    manifest = {
        "schema_version": 1, "status": "completed", "architecture": "temporal_component_super_hierarchy",
        "label_blind": True, "labels_read": False,
        "input_unit": "one component descriptor from one independently processed chunk",
        "levels": ["chunk_components_input", "daily_supercomponents", "monthly_supercomponents", "yearly_supercomponents"],
        "similarity": "mixed_similarity with strict complete-link across every pair of child components",
        "similarity_threshold_strictly_greater_than": float(similarity_threshold),
        "max_represented_raw_rows_per_supercomponent": int(max_component_mass),
        "daily_chunk_count": int(len(np.unique(chunk_id))),
        "input_component_count": int(len(x)),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "caveats": [
            "The input must contain component descriptors already computed independently per 10,000-row chunk.",
            "This module does not itself read Parquet or run the chunk-level graph/SCRAC stage.",
            "Capacity is measured in represented raw rows; indivisible child components above capacity remain singleton supercomponents.",
            "Centroid/mode descriptors are summaries; the Complete-Link gate uses the original child descriptors during each period merge."
        ]
    }
    return {"levels": outputs, "manifest": manifest}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=Path, required=True, help="NPZ: x,b,c,mass,period,chunk_id")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--capacity", type=int, default=200)
    p.add_argument("--block-size", type=int, default=128)
    a = p.parse_args()
    with np.load(a.atoms, allow_pickle=False) as f:
        result = build_temporal_hierarchy(f["x"], f["b"], f["c"], f["mass"],
                                          f["period"], f["chunk_id"],
                                          similarity_threshold=a.threshold,
                                          max_component_mass=a.capacity,
                                          block_size=a.block_size)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["manifest"]["status"],
                      "daily": result["levels"]["day_summary"],
                      "monthly": result["levels"]["month_summary"],
                      "yearly": result["levels"]["year_summary"],
                      "report": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
