from scripts.sweep_node_scrac import parse_floats


def test_parse_floats_accepts_comma_separated_values():
    assert parse_floats("0.75, 0.78,0.80", "thresholds") == [0.75, 0.78, 0.80]


def test_parse_floats_rejects_empty_values():
    import argparse
    try:
        parse_floats(" , ", "thresholds")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("empty sweep values must be rejected")
