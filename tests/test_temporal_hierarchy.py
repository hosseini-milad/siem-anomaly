import numpy as np
import pytest

from hierarchy.temporal import build_temporal_hierarchy


def atoms():
    # Two daily chunks on day 1, one chunk on day 2, and a second month.
    x = np.array([[1., 0.], [.99, .01], [0., 1.], [.01, .99],
                  [1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    b = np.zeros((8, 1), dtype=np.int64)
    c = np.zeros((8, 1), dtype=np.int64)
    mass = np.full(8, 10., dtype=np.float64)
    period = np.array(["2026-06-01"] * 4 + ["2026-06-02"] * 2 +
                      ["2026-07-01"] * 2)
    chunk = np.array(["d1-c1"] * 2 + ["d1-c2"] * 2 + ["d2-c1"] * 2 +
                     ["m2-d1-c1"] * 2)
    return x, b, c, mass, period, chunk


def test_builds_day_month_year_from_chunk_components_label_blind():
    result = build_temporal_hierarchy(*atoms(), similarity_threshold=.85,
                                      max_component_mass=100)
    assert result["manifest"]["label_blind"] is True
    assert result["manifest"]["labels_read"] is False
    assert result["manifest"]["levels"] == [
        "chunk_components_input", "daily_supercomponents",
        "monthly_supercomponents", "yearly_supercomponents"]
    day = result["levels"]["day_summary"]["periods"]
    assert set(day) == {"2026-06-01", "2026-06-02", "2026-07-01"}
    month = result["levels"]["month_summary"]["periods"]
    assert set(month) == {"2026-06", "2026-07"}
    year = result["levels"]["year_summary"]["periods"]
    assert set(year) == {"2026"}
    assert result["levels"]["day_summary"]["total_input"] == 8


def test_complete_link_prevents_transitive_chain_merge():
    # A~B and B~C, but A is not similar to C. All three must not form one group.
    x = np.array([[1., 0.], [.8, .6], [.28, .96]])
    b = np.zeros((3, 1), dtype=np.int64)
    c = np.zeros((3, 1), dtype=np.int64)
    mass = np.ones(3)
    period = np.array(["2026-06-01"] * 3)
    chunk = np.array(["a", "b", "c"])
    result = build_temporal_hierarchy(x, b, c, mass, period, chunk,
                                      similarity_threshold=.85, max_component_mass=10)
    rows = result["levels"]["day"]
    assert len(rows) == 2


def test_capacity_is_respected_and_singletons_remain():
    x, b, c, mass, period, chunk = atoms()
    result = build_temporal_hierarchy(x, b, c, mass, period, chunk,
                                      similarity_threshold=.85, max_component_mass=15)
    for level in ("day", "month", "year"):
        for row in result["levels"][level]:
            assert row["represented_raw_rows"] <= 15


def test_rejects_invalid_period_and_unaligned_inputs():
    x, b, c, mass, period, chunk = atoms()
    period[0] = "not-a-date"
    with pytest.raises(ValueError, match="ISO dates"):
        build_temporal_hierarchy(x, b, c, mass, period, chunk)
