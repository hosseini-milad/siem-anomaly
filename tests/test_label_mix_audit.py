import json
import numpy as np
import pytest

from scripts.audit_scrac_label_mix import audit


def test_posthoc_attack_normal_mix_uses_raw_mapping_rows(tmp_path):
    mapping = tmp_path / "mapping"
    mapping.mkdir()
    components = tmp_path / "components.npz"
    report = tmp_path / "report.json"
    np.savez_compressed(
        components,
        component_id=np.array([0, 0, 1], dtype=np.int32),
        raw_rows=np.array([3, 2], dtype=np.int64),
        attack_rows=np.array([2, 0], dtype=np.int64),
    )
    np.savez_compressed(
        mapping / "mapping_000.npz",
        node=np.array([0, 0, 0, 1, 1], dtype=np.int64),
        source=np.array([
            "date=2026-06-21/type=attacks/a.parquet",
            "date=2026-06-21/type=attacks/a.parquet",
            "date=2026-06-21/type=normal/b.parquet",
            "date=2026-06-21/type=normal/b.parquet",
            "date=2026-06-21/type=normal/b.parquet",
        ]),
    )
    report.write_text(json.dumps({"assignment": [0, 1]}), encoding="utf-8")
    result = audit(mapping, components, report)["summary"]
    assert result["attack_rows"] == 2
    assert result["normal_rows"] == 3
    assert result["mixed_groups"] == 1
    assert result["pure_normal_groups"] == 1
    assert result["groups"][0]["impurity_pct"] == pytest.approx(33.3333)
    assert result["raw_row_weighted_impurity_pct"] == pytest.approx(20.0)


def test_audit_detects_attack_only_graph(tmp_path):
    mapping = tmp_path / "mapping"
    mapping.mkdir()
    components = tmp_path / "components.npz"
    report = tmp_path / "report.json"
    np.savez_compressed(
        components,
        component_id=np.array([0], dtype=np.int32),
        raw_rows=np.array([2], dtype=np.int64),
        attack_rows=np.array([2], dtype=np.int64),
    )
    np.savez_compressed(
        mapping / "mapping_000.npz",
        node=np.array([0, 0], dtype=np.int64),
        source=np.array([
            "date=2026-06-21/type=attacks/a.parquet",
            "date=2026-06-21/type=attacks/a.parquet",
        ]),
    )
    report.write_text(json.dumps({"assignment": [0]}), encoding="utf-8")
    result = audit(mapping, components, report)["summary"]
    assert result["all_attack_data_warning"] is True
    assert result["normal_rows"] == 0
    assert result["pure_attack_groups"] == 1


def test_audit_fails_if_mapping_and_component_raw_rows_disagree(tmp_path):
    mapping = tmp_path / "mapping"
    mapping.mkdir()
    components = tmp_path / "components.npz"
    report = tmp_path / "report.json"
    np.savez_compressed(
        components,
        component_id=np.array([0], dtype=np.int32),
        raw_rows=np.array([3], dtype=np.int64),
        attack_rows=np.array([0], dtype=np.int64),
    )
    np.savez_compressed(
        mapping / "mapping_000.npz",
        node=np.array([0, 0], dtype=np.int64),
        source=np.array([
            "date=2026-06-21/type=normal/a.parquet",
            "date=2026-06-21/type=normal/a.parquet",
        ]),
    )
    report.write_text(json.dumps({"assignment": [0]}), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping/raw_rows mismatch"):
        audit(mapping, components, report)
