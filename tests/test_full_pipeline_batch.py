from pathlib import Path
import pytest
from scripts.run_full_pipeline_batch import discover_days, select_days

def make_day(root, day, normal=True, attacks=True):
    folder=Path(root)/f"date={day}"
    for kind, enabled in (("normal",normal),("attacks",attacks)):
        if enabled:
            target=folder/f"type={kind}"
            target.mkdir(parents=True,exist_ok=True)
            (target/"part.parquet").write_bytes(b"test")

def test_discovery_excludes_normal_only_training_days(tmp_path):
    make_day(tmp_path,"2026-06-20",normal=True,attacks=False)
    make_day(tmp_path,"2026-06-21")
    make_day(tmp_path,"2026-06-30")
    assert discover_days(tmp_path)==["2026-06-21","2026-06-30"]

def test_explicit_days_are_validated(tmp_path):
    make_day(tmp_path,"2026-06-21")
    make_day(tmp_path,"2026-06-30")
    assert select_days(tmp_path,["2026-06-30","2026-06-21"])==["2026-06-21","2026-06-30"]
    with pytest.raises(ValueError):
        select_days(tmp_path,["2026-06-20"])
    with pytest.raises(ValueError):
        select_days(tmp_path,["2026-07-01"])

def test_auto_discovery_empty_fails(tmp_path):
    with pytest.raises(ValueError):
        select_days(tmp_path,None)
