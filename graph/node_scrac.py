"""Run SC-FAGC + SCRAC directly on graph representative nodes.

This node-level mode is intended to prevent a pre-existing giant connected
component from becoming an indivisible input atom. Each representative node is
one SCRAC member. Mapping files are used only to count represented raw rows;
the source/partition labels are never read by this module.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components

from graph.sc_fagc import run


def load_node_mass(mapping_dir: Path, n_nodes: int) -> tuple[np.ndarray, list[str]]:
    """Count raw rows per representative node without reading source labels."""
    mapping_dir = Path(mapping_dir)
    files = sorted(mapping_dir.glob("mapping_*.npz"))
    if not files:
        raise FileNotFoundError(f"no mapping_*.npz files found in {mapping_dir}")
    mass = np.zeros(n_nodes, dtype=np.int64)
    for path in files:
        with np.load(path, allow_pickle=False) as f:
            if "node" not in f.files:
                raise ValueError(f"{path.name}: missing node column")
            node = np.asarray(f["node"], dtype=np.int64)
        if node.ndim != 1 or np.any(node < 0) or np.any(node >= n_nodes):
            raise ValueError(f"{path.name}: invalid node indices")
        np.add.at(mass, node, 1)
    if np.any(mass == 0):
        missing = np.flatnonzero(mass == 0)[:20].tolist()
        raise ValueError(f"some representative nodes have no mapped rows: {missing}")
    return mass, [p.name for p in files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--adjacency", type=Path, default=None,
                        help="Original sparse adjacency; enables hard same-original-component assignment.")
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", type=int, default=None)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--lambda-size", type=float, default=0.20)
    parser.add_argument("--acceptance-threshold", type=float, default=0.75)
    parser.add_argument("--anchor-k", type=int, default=5)
    parser.add_argument("--anchor-threshold", type=float, default=0.85)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--sinkhorn-iter", type=int, default=2000)
    args = parser.parse_args()

    started = time.perf_counter()
    with np.load(args.nodes, allow_pickle=False) as f:
        x = np.asarray(f["x"])
        b = np.asarray(f["b"])
        c = np.asarray(f["c"])
    if x.ndim != 2 or b.ndim != 2 or c.ndim != 2 or not (len(x) == len(b) == len(c)):
        raise ValueError("node feature arrays are not aligned")

    mass, mapping_files = load_node_mass(args.mapping_dir, len(x))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    topology_labels = None
    topology_components_artifact = None
    if args.adjacency is not None:
        adjacency = load_npz(args.adjacency).tocsr()
        if adjacency.shape != (len(x), len(x)):
            raise ValueError("adjacency dimensions must match nodes.npz")
        n_topology_components, topology_labels = connected_components(adjacency, directed=False)
        topology_labels = topology_labels.astype(np.int64, copy=False)
        topology_components_artifact = args.output.with_name(args.output.stem + "_topology_components.npz")
        np.savez_compressed(topology_components_artifact, component_id=topology_labels)
    result = run(
        x, b, c, mass,
        n_anchors=args.anchors,
        n_clusters=args.clusters,
        max_component_group_mass=args.capacity,
        anchor_k=args.anchor_k,
        anchor_threshold=args.anchor_threshold,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        sinkhorn_iter=args.sinkhorn_iter,
        scrac_capacity=args.capacity,
        scrac_lambda=args.lambda_size,
        scrac_acceptance=args.acceptance_threshold,
        topology_labels=topology_labels,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_components = args.output.with_name(args.output.stem + "_input_components.npz")
    np.savez_compressed(
        input_components,
        component_id=np.arange(len(x), dtype=np.int32),
        raw_rows=mass.astype(np.int64),
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        assignment=np.asarray(result["assignment"], dtype=np.int64),
        probabilities=np.asarray(result["probabilities"], dtype=np.float32),
        component_mass=mass,
        anchor_indices=np.asarray(result["anchor_indices"], dtype=np.int64),
    )
    result["input_unit"] = "one_graph_representative_node"
    result["label_blind"] = True
    result["mapping_files_used_for_mass_only"] = mapping_files
    result["mapping_source_labels_read"] = False
    result["input_components_artifact"] = str(input_components)
    result["topology_constraint"] = {
        "enabled": topology_labels is not None,
        "original_connected_components": int(n_topology_components) if topology_labels is not None else None,
        "component_ids_artifact": str(topology_components_artifact) if topology_components_artifact else None,
        "policy": "hard same-original-connected-component group eligibility" if topology_labels is not None else "disabled",
    }
    result["complete_link_threshold"] = 0.85
    result["runtime_seconds"] = round(time.perf_counter() - started, 3)
    result["purpose"] = (
        "Node-level SCRAC experiment: Complete-Link is applied across graph "
        "representative nodes, so a giant connected component is not an "
        "indivisible input atom."
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": result.get("status", "completed"),
        "components": result.get("components"),
        "groups": result.get("scrac_assignment", {}).get("groups"),
        "max_group_size": result.get("scrac_assignment", {}).get("max_group_size"),
        "capacity_violations": result.get("scrac_assignment", {}).get("capacity_violations"),
        "topology_constraint": result.get("topology_constraint"),
        "runtime_seconds": result["runtime_seconds"],
        "report": str(args.output),
        "input_components": str(input_components),
    }, indent=2))


if __name__ == "__main__":
    main()
