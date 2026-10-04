import json
import warnings

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


@pytest.mark.parametrize("gross_returns", [[1.01, 1.01], [0.99, 1.04]])
def test_unrepresentable_annualized_return_is_null_without_losing_other_metrics(gross_returns):
    returns = np.log(np.asarray(gross_returns))
    metrics = compute_pnl_metrics(returns, np.ones((1, len(returns))))

    assert metrics.annualized_return is None
    assert metrics.calmar is None
    assert metrics.total_return == pytest.approx(np.prod(gross_returns) - 1.0)
    assert metrics.max_drawdown == pytest.approx(min(gross_returns[0] - 1.0, 0.0))
    serialized = json.dumps(metrics.as_dict(), allow_nan=False)
    assert json.loads(serialized)["annualized_return"] is None


def test_calmar_overflow_is_null_even_when_annualized_return_is_finite():
    returns = np.array([-1e-5, 2 * 709.0 / (365 * 24 * 60) + 1e-5])

    metrics = compute_pnl_metrics(returns, np.ones((1, 2)))

    assert metrics.annualized_return is not None
    assert np.isfinite(metrics.annualized_return)
    assert metrics.calmar is None


def test_single_downside_observation_does_not_emit_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = compute_pnl_metrics(np.log(np.array([0.999, 1.001])), np.ones((1, 2)))

    assert metrics.sortino == 0.0
    assert metrics.annualized_return is not None
