from types import SimpleNamespace

import numpy as np
import pytest

from superpnl.training import backtest_buy_and_hold, backtest_scores


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
