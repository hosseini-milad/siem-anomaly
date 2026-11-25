import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hierarchy.graph import build_hierarchy, save_hierarchy, load_hierarchy


def fixture_graph():
    # Two connected components: a triangle and a three-node chain.
    rows = np.array([0, 1, 0, 2, 1, 2, 3, 4, 4, 5])
    cols = np.array([1, 0, 2, 0, 2, 1, 4, 3, 5, 4])
    data = np.array([2., 2., 1., 1., 3., 3., 4., 4., 2., 2.])
    return csr_matrix((data, (rows, cols)), shape=(6, 6))


def test_hierarchy_conserves_weight_volume_topology_and_roundtrips(tmp_path):
    graph = fixture_graph()
    result = build_hierarchy(graph, np.arange(6), max_levels=5)
    assert len(result.levels) >= 2
    for level, quotient in enumerate(result.levels):
        assert np.isclose(quotient.sum() / 2, graph.sum() / 2)
        assert np.isclose(quotient.sum(), graph.sum())
        assert len(np.unique(result.component_ids[level])) == 2
    for mapping in result.child_to_parent:
        assert np.all(np.bincount(mapping) <= 2)
    npz, report = save_hierarchy(result, tmp_path / "hierarchy")
    restored = load_hierarchy(npz, report)
    assert len(restored.levels) == len(result.levels)
    for a, b in zip(result.levels, restored.levels):
        assert (a != b).nnz == 0
    for a, b in zip(result.child_to_parent, restored.child_to_parent):
        np.testing.assert_array_equal(a, b)


def test_rejects_base_scrac_group_spanning_original_components():
    with pytest.raises(ValueError, match="spans multiple"):
        build_hierarchy(fixture_graph(), np.array([0, 1, 2, 0, 3, 4]))


def test_affinity_gate_can_stop_merging():
    result = build_hierarchy(fixture_graph(), np.arange(6),
                             max_levels=4, min_affinity=1.0)
    assert len(result.levels) == 1
    assert result.manifest["stop_reason"] == "no_eligible_adjacent_pairs"


def test_internal_only_quotient_node_is_reported_as_isolated():
    graph = csr_matrix(np.array([[0., 1.], [1., 0.]]))
    result = build_hierarchy(graph, np.array([0, 0]))
    assert result.manifest["level_metrics"][0]["isolated_nodes"] == 1
