from scripts.run_superpnl_experiment import format_metric


def test_unrepresentable_metric_renders_as_unavailable():
    assert format_metric(None) == "N/A"
