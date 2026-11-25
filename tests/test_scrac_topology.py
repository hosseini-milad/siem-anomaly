import json

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from scripts.evaluate_scrac_topology import evaluate_topology


def test_topology_audit_conserves_weight_and_degree_volume(tmp_path):
    graph_path = tmp_path / "adjacency.npz"
    assignment_path = tmp_path / "assignment.json"
    # Triangle in group 0, one edge in group 1, and one cut edge.
    rows = np.array([0, 1, 0, 2, 1, 2, 2, 3, 3, 4, 4, 3])
    cols = np.array([1, 0, 2, 0, 2, 1, 3, 2, 4, 3, 3, 4])
    data = np.array([1., 1., 2., 2., 1., 1., 3., 3., 4., 4., 0., 0.])
    # Construct a clean symmetric graph with edges (0,1)=1, (0,2)=2,
    # (1,2)=1, (2,3)=3, (3,4)=4.
    r = np.array([0, 1, 0, 2, 1, 2, 2, 3, 3, 4])
    c = np.array([1, 0, 2, 0, 2, 1, 3, 2, 4, 3])
    w = np.array([1., 1., 2., 2., 1., 1., 3., 3., 4., 4.])
    g = csr_matrix((w, (r, c)), shape=(5, 5))
    save_npz(graph_path, g)
    assignment_path.write_text(json.dumps({"assignment": [0, 0, 0, 1, 1]}),
                               encoding="utf-8")

    result = evaluate_topology(graph_path, assignment_path, spectral_k=3)
    assert result["label_blind"] is True
    assert result["original_graph"]["undirected_edges"] == 5
    assert result["quotient_graph"]["undirected_inter_group_edges"] == 1
    assert result["edge_and_weight_conservation"]["internal_edges"] == 4
    assert result["edge_and_weight_conservation"]["external_edges"] == 1
    assert result["edge_and_weight_conservation"]["quotient_external_weight_matches_original_cut"]
    assert result["degree_volume_preservation"]["exact_within_tolerance"]
    assert result["partition_quality"]["groups_spanning_multiple_original_connected_components"] == 0
    assert result["spectral_preservation"]["low_spectrum_comparison"]["compared_eigenvalues"] > 0
