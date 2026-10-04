from types import SimpleNamespace

import numpy as np
import pytest

from superpnl.training import backtest_buy_and_hold, backtest_rule_momentum, backtest_scores


def make_dataset(gross_returns):
    gross_returns = np.asarray(gross_returns, dtype="float64")
    return SimpleNamespace(
        n_symbols=gross_returns.shape[0],
        train_range=(0, gross_returns.shape[1]),
        val_range=(0, gross_returns.shape[1]),
        test_range=(0, gross_returns.shape[1]),
        next_returns=np.log(gross_returns),
    )


def test_equal_weight_portfolio_uses_asset_values_not_mean_log_returns():
    dataset = make_dataset([[2.0], [0.5]])

    metrics, positions, returns = backtest_scores(dataset, np.ones((2, 1, 1)), "test", 0)

    assert metrics["total_return"] == pytest.approx(0.25)
    np.testing.assert_allclose(positions, [[0.5], [0.5]])
    assert np.expm1(returns.sum()) == pytest.approx(0.25)
    assert metrics["average_position"] == pytest.approx(1.0)


def test_buy_and_hold_keeps_initial_units_without_rebalancing():
    gross = np.ones((2, 1000))
    gross[:, :2] = [[2.0, 1.0], [0.5, 2.0]]

    metrics = backtest_buy_and_hold(make_dataset(gross), "test")

    assert metrics["total_return"] == pytest.approx(0.5)
    assert metrics["trade_count"] == 2


def test_entry_fee_is_funded_without_borrowing_and_no_terminal_sale():
    dataset = make_dataset([[1.0]])

    metrics = backtest_buy_and_hold(dataset, "test", fixed_fee_bps=100.0)

    assert metrics["total_return"] == pytest.approx(1.0 / 1.01 - 1.0)
    assert metrics["trade_count"] == 1
    assert metrics["turnover"] == pytest.approx(1.0 / 1.01)


def test_rebalance_trades_even_when_target_signals_do_not_change():
    from superpnl.portfolio import simulate_portfolio

    ledger = simulate_portfolio(np.log([[2.0, 1.0], [1.0, 1.0]]), np.ones((2, 2)))

    np.testing.assert_allclose(ledger.traded_notional, [[0.5, 0.25], [0.5, 0.25]])
    np.testing.assert_allclose(ledger.equity, [1.5, 1.5])
    np.testing.assert_allclose(ledger.positions, 0.5)


def test_ledger_reconciles_costs_cash_and_per_symbol_contributions():
    from superpnl.portfolio import simulate_portfolio

    ledger = simulate_portfolio(
        np.log([[1.1, 0.8, 1.0], [0.9, 1.2, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        cost_rate=0.01,
    )

    previous_equity = np.r_[1.0, ledger.equity[:-1]]
    np.testing.assert_allclose(ledger.pnl_by_symbol.sum(axis=0), ledger.equity - previous_equity)
    np.testing.assert_allclose(ledger.costs, ledger.traded_notional * 0.01)
    np.testing.assert_allclose(np.exp(ledger.portfolio_returns), ledger.equity / previous_equity)
    assert np.all(ledger.cash >= -1e-12)
    assert ledger.cash[-1] == pytest.approx(ledger.equity[-1])
    assert ledger.positions[:, -1].sum() == 0.0


def test_all_cash_stays_constant_even_when_assets_move_and_fees_apply():
    from superpnl.portfolio import simulate_portfolio

    ledger = simulate_portfolio(np.log([[2.0, 0.5], [0.5, 2.0]]), np.zeros((2, 2)), cost_rate=0.01)

    np.testing.assert_allclose(ledger.equity, 1.0)
    np.testing.assert_allclose(ledger.traded_notional, 0.0)
    np.testing.assert_allclose(ledger.pnl_by_symbol, 0.0)


@pytest.mark.parametrize("cost_rate", [-0.1, 1.0, np.nan])
def test_invalid_cost_rate_is_rejected(cost_rate):
    from superpnl.portfolio import simulate_portfolio

    with pytest.raises(ValueError, match="cost_rate"):
        simulate_portfolio(np.zeros((1, 1)), np.ones((1, 1)), cost_rate=cost_rate)


def momentum_dataset(raw_return, mean, std, feature_name="ret_30m"):
    dataset = make_dataset([[1.0]])
    dataset.feature_names = ["rsi_5m", feature_name]
    dataset.feature_mean = np.array([0.0, mean])
    dataset.feature_std = np.array([1.0, std])
    dataset.feature_inputs = np.array([[[1.0, (raw_return - mean) / std]]])
    return dataset


@pytest.mark.parametrize(
    ("raw_return", "mean", "std", "threshold_bps", "expected_position"),
    [(0.01, 0.02, 0.01, 0.0, 1.0), (0.0003, 0.0, 0.0001, 5.0, 0.0)],
)
def test_momentum_compares_raw_returns_with_bps_threshold(
    raw_return, mean, std, threshold_bps, expected_position
):
    dataset = momentum_dataset(raw_return, mean, std)

    metrics = backtest_rule_momentum(dataset, "test", 0, threshold_bps=threshold_bps)

    assert metrics["average_position"] == pytest.approx(expected_position)


def test_momentum_falls_back_to_an_available_return_feature():
    dataset = momentum_dataset(-0.01, 0.0, 0.01, feature_name="ret_5m")

    metrics = backtest_rule_momentum(dataset, "test", 0)

    assert metrics["average_position"] == 0.0


def test_momentum_rule_is_reused_for_every_model_horizon():
    dataset = momentum_dataset(0.01, 0.02, 0.01)

    metrics = [backtest_rule_momentum(dataset, "test", head) for head in (0, 1, 2)]

    assert metrics[0] == metrics[1] == metrics[2]
    assert metrics[2]["average_position"] == 1.0


@pytest.mark.parametrize("missing_stat", ["feature_mean", "feature_std"])
def test_momentum_rejects_missing_training_normalization(missing_stat):
    dataset = momentum_dataset(0.01, 0.0, 0.01)
    setattr(dataset, missing_stat, None)

    with pytest.raises(ValueError, match="normalization"):
        backtest_rule_momentum(dataset, "test", 0)


def test_momentum_rejects_schema_without_return_features():
    dataset = momentum_dataset(0.01, 0.0, 0.01, feature_name="vol_std_30m")

    with pytest.raises(ValueError, match="return feature"):
        backtest_rule_momentum(dataset, "test", 0)


@pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -np.inf])
def test_backtest_rejects_nonfinite_scores_before_creating_positions(invalid_score):
    dataset = make_dataset([[1.0, 1.0], [1.0, 1.0]])
    pred = np.ones((2, 2, 1))
    pred[1, 0, 0] = invalid_score

    with pytest.raises(ValueError, match="scores.*finite"):
        backtest_scores(dataset, pred, "test", 0)


@pytest.mark.parametrize("threshold_bps", [np.nan, np.inf, -np.inf])
def test_backtest_rejects_nonfinite_thresholds(threshold_bps):
    dataset = make_dataset([[1.0]])

    with pytest.raises(ValueError, match="threshold.*finite"):
        backtest_scores(dataset, np.ones((1, 1, 1)), "test", 0, threshold_bps=threshold_bps)
