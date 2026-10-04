import numpy as np
import pytest

from superpnl.metrics import compute_pnl_metrics


@pytest.mark.parametrize(
    ("gross_returns", "expected_drawdown"),
    [([0.9], -0.1), ([0.9, 0.8 / 0.9], -0.2), ([], 0.0)],
)
def test_max_drawdown_includes_initial_capital(gross_returns, expected_drawdown):
    returns = np.log(np.asarray(gross_returns, dtype="float64"))
    positions = np.ones((1, len(returns)))

    metrics = compute_pnl_metrics(returns, positions)

    assert metrics.max_drawdown == pytest.approx(expected_drawdown)
    if not gross_returns:
        assert metrics.turnover == 0.0


def test_calmar_uses_drawdown_from_initial_capital():
    metrics = compute_pnl_metrics(np.log(np.array([0.9])), np.ones((1, 1)))

    assert metrics.calmar == pytest.approx(-1.0)
