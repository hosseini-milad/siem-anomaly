import json
import numpy as np
import pytest
from scripts.build_temporal_atoms import build_atoms


def test_build_atoms_aggregates_scrac_groups_without_labels(tmp_path):
    nodes = tmp_path / "nodes.npz"
    scrac = tmp_path / "scrac.npz"
    manifest = tmp_path / "chunks.json"
    output = tmp_path / "atoms.npz"
    np.savez_compressed(nodes, x=np.array([[1., 0.], [.9, .1], [0., 1.]]),
                        b=np.array([[1], [1], [0]]), c=np.array([[2], [2], [3]]))
    np.savez_compressed(scrac, assignment=np.array([0, 0, 1]),
                        component_mass=np.array([10., 20., 30.]))
    manifest.write_text(json.dumps({"chunks": [{
        "chunk_id": "2026-06-01-c000", "period": "2026-06-01",
        "nodes": "nodes.npz", "scrac": "scrac.npz"}]}), encoding="utf-8")
    result = build_atoms(manifest, output)
    with np.load(output, allow_pickle=False) as data:
        assert data["x"].shape == (2, 2)
        assert data["mass"].tolist() == [30., 30.]
        assert data["period"].tolist() == ["2026-06-01"] * 2
        assert data["chunk_id"].tolist() == ["2026-06-01-c000"] * 2
    assert result["label_blind"] is True
    assert result["labels_read"] is False
    assert result["represented_raw_rows"] == 60.


def test_build_atoms_rejects_unaligned_scrac_output(tmp_path):
    nodes = tmp_path / "nodes.npz"
    scrac = tmp_path / "scrac.npz"
    manifest = tmp_path / "chunks.json"
    np.savez_compressed(nodes, x=np.ones((2, 2)), b=np.ones((2, 1)), c=np.ones((2, 1)))
    np.savez_compressed(scrac, assignment=np.array([0]), component_mass=np.array([10.]))
    manifest.write_text(json.dumps({"chunks": [{
        "chunk_id": "c0", "period": "2026-06-01",
        "nodes": "nodes.npz", "scrac": "scrac.npz"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="align"):
        build_atoms(manifest, tmp_path / "atoms.npz")
