from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_superpnl_experiment import by_month_summary, by_symbol_summary


def sample_dataset():
    return SimpleNamespace(symbols=["A-USDT", "B-USDT"], n_symbols=2,
                           next_returns=np.log([[2.0], [0.5]]),
                           train_range=(0, 1), val_range=(0, 1), test_range=(0, 1))


def test_symbol_contributions_reconcile_to_portfolio_return():
    rows = by_symbol_summary(sample_dataset(), np.full((2, 1), 0.5), "test")
    by_name = {row["symbol"]: row for row in rows}
    assert by_name["A-USDT"].get("pnl_contribution") == pytest.approx(0.5)
    assert by_name["B-USDT"].get("pnl_contribution") == pytest.approx(-0.25)
    assert sum(row["pnl_contribution"] for row in rows) == pytest.approx(0.25)


def test_symbol_contributions_include_actual_fees():
    rows = by_symbol_summary(sample_dataset(), np.full((2, 1), 0.5), "test",
                             fixed_fee_bps=100.0)
    assert sum(row["pnl_contribution"] for row in rows) == pytest.approx(1.25 / 1.01 - 1.0)


def test_monthly_attribution_uses_return_end_instead_of_decision_time():
    # Jan 31 23:58 UTC signal earns Feb 1 00:00 UTC closing mark.
    timestamp = np.datetime64("2026-01-31T23:58", "ms").astype("int64")
    data = SimpleNamespace(timestamps=timestamp + np.arange(3) * 60000,
                           train_range=(0, 1), val_range=(0, 1), test_range=(0, 1))
    rows = by_month_summary(data, np.array([0.01]), "test")
    assert rows[0]["month"] == "2026-02"
