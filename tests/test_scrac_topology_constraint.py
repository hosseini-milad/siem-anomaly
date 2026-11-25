import numpy as np
import pytest
from graph.sc_fagc import scrac_assign


def test_topology_constraint_prevents_cross_component_group_merges():
    x = np.tile(np.array([[1.0, 0.0]]), (4, 1))
    b = np.ones((4, 1), dtype=np.int8)
    c = np.zeros((4, 1), dtype=np.int8)
    mass = np.array([2, 3, 4, 5], dtype=np.int64)
    probabilities = np.array([[0.9, 0.1]] * 4)
    anchors = np.array([0, 2], dtype=np.int64)
    anchor_assignments = np.array([[0.9, 0.1], [0.1, 0.9]])
    original_cc = np.array([0, 0, 1, 1], dtype=np.int64)
    labels, report = scrac_assign(
        x, b, c, mass, probabilities, anchors, anchor_assignments,
        threshold=0.85, capacity=10, lambda_size=0.0,
        acceptance_threshold=0.0, topology_labels=original_cc,
    )
    assert np.all(labels >= 0)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    for group_id in np.unique(labels):
        assert np.unique(original_cc[labels == group_id]).size == 1
    assert report["topology_constraint_enabled"] is True
    assert report["original_connected_component_count"] == 2
    assert report["groups_spanning_multiple_original_components"] == 0
    assert report["blocked_cross_component_group_checks"] > 0
    assert sum(report["group_sizes"]) == 4
    assert sum(report["group_mass_raw_rows"]) == int(mass.sum())


def test_topology_labels_must_align_with_nodes():
    x = np.array([[1.0, 0.0], [0.9, 0.1]])
    b = np.ones((2, 1), dtype=np.int8)
    c = np.zeros((2, 1), dtype=np.int8)
    mass = np.ones(2, dtype=np.int64)
    probabilities = np.array([[0.9], [0.9]])
    with pytest.raises(ValueError, match="topology_labels"):
        scrac_assign(x, b, c, mass, probabilities, np.array([0]), np.array([[1.0]]),
                     topology_labels=np.array([0]))
