import json
import numpy as np
from scripts.evaluate_scrac_structure import evaluate


def test_structural_audit_checks_pairs_and_capacity(tmp_path):
    nodes = tmp_path / "nodes.npz"
    report = tmp_path / "assignment.json"
    output = tmp_path / "out.json"
    # Identical vectors and discrete features produce similarity 1.0.
    x = np.array([[1., 0.], [1., 0.], [0., 1.]])
    b = np.array([[1], [1], [0]])
    c = np.array([[0], [0], [1]])
    np.savez_compressed(nodes, x=x, b=b, c=c)
    report.write_text(json.dumps({
        "assignment": [0, 0, 1],
        "acceptance_threshold": .75,
        "scrac_lambda": .2,
        "complete_link_threshold": .85,
        "scrac_capacity": 2,
        "input_unit": "one_graph_representative_node",
    }))
    result = evaluate(nodes, report, threshold=.85, capacity=2)
    assert result["assigned_nodes"] == 3
    assert result["groups"] == 2
    assert result["capacity_valid"] is True
    assert result["complete_link_valid"] is True
    assert result["within_group_pair_similarity"]["min"] == 1.0
    assert result["within_group_pair_similarity"]["count"] == 1
