from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import select_low_turnover_params as selection
from superpnl.data import DatasetConfig, prepare_dataset
from superpnl.provenance import prepared_data_contract
from superpnl.portfolio import simulate_weight_portfolio
from superpnl.tabular_training import (
    LightGBMTrainConfig, LightGBMWrapper, _lgb_params,
    predict_lightgbm_split, train_lightgbm_model,
)
from superpnl.training import TrainConfig
from test_data import make_raw_frame, write_raw_frame


@pytest.fixture
def tabular_dataset(tmp_path):
    raw = tmp_path / "raw"
    for i, symbol in enumerate(("BTC-USDT", "ETH-USDT", "SOL-USDT")):
        frame = make_raw_frame()
        multiplier = 1 + i * 0.0001 * np.arange(len(frame))
        for column in ("open", "high", "low", "close"):
            frame[column] *= multiplier
        frame["volume"] += i * 20
        write_raw_frame(raw, frame, symbol)
    return prepare_dataset(DatasetConfig(
        str(raw), str(tmp_path / "cache"), lookback=32, horizons=(5, 15), feature_windows=(5, 15, 30),
    ))


def train_tiny(dataset, out_dir, objective="regression_l2"):
    config = LightGBMTrainConfig(
        objective=objective, num_threads=1, num_boost_round=2,
        early_stopping_rounds=0, train_samples=30, validation_samples=18,
        min_data_in_leaf=1, num_leaves=3, max_depth=2,
        feature_fraction=1, bagging_fraction=1, bagging_freq=0,
    )
    return train_lightgbm_model(dataset, TrainConfig(device="cpu"), out_dir, "tiny_lgb", config)


def strict_json(path):
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=invalid_constant)


@pytest.mark.parametrize("objective", ["regression_l2", "rank_regression", "lambdarank"])
def test_lightgbm_saves_verified_tabular_contract_and_loads_identical_predictions(tabular_dataset, tmp_path, objective):
    out = tmp_path / "run"
    wrapper, _ = train_tiny(tabular_dataset, out, objective)
    metadata = strict_json(out / "tiny_lgb_metadata.json")
    assert metadata["model_type"] == "lightgbm"
    assert metadata["data_contract"] == prepared_data_contract(tabular_dataset)
    assert not list(out.glob("*.pt"))
    loaded = LightGBMWrapper.load(out, tabular_dataset.horizons, name="tiny_lgb")
    original, _, _ = predict_lightgbm_split(tabular_dataset, wrapper, "test")
    restored, _, _ = predict_lightgbm_split(tabular_dataset, loaded, "test")
    np.testing.assert_allclose(restored, original)
    for path in out.glob("*.json"):
        strict_json(path)


def test_lightgbm_rejects_legacy_dataset_before_training(tabular_dataset, tmp_path):
    tabular_dataset.cache_metadata = None
    with pytest.raises(ValueError, match="data contract"):
        train_tiny(tabular_dataset, tmp_path / "run")
    assert not list((tmp_path / "run").glob("*.txt"))


@pytest.mark.parametrize("change", ["features", "stats", "horizons"])
def test_lightgbm_inference_rejects_mismatched_dataset(tabular_dataset, tmp_path, change):
    model, _ = train_tiny(tabular_dataset, tmp_path / "run")
    if change == "features":
        tabular_dataset.feature_names = list(reversed(tabular_dataset.feature_names))
    elif change == "stats":
        tabular_dataset.feature_mean = tabular_dataset.feature_mean + 0.125
    else:
        tabular_dataset.horizons = tuple(reversed(tabular_dataset.horizons))
    with pytest.raises(ValueError, match="data contract"):
        predict_lightgbm_split(tabular_dataset, model, "test")


def test_lightgbm_load_rejects_reordered_horizons(tabular_dataset, tmp_path):
    out = tmp_path / "run"
    train_tiny(tabular_dataset, out)
    with pytest.raises(ValueError, match="horizon"):
        LightGBMWrapper.load(out, tuple(reversed(tabular_dataset.horizons)), name="tiny_lgb")


def test_lightgbm_honors_single_thread_environment(monkeypatch):
    monkeypatch.setenv("LGB_NUM_THREADS", "1")
    assert _lgb_params(LightGBMTrainConfig())["num_threads"] == 1
    assert _lgb_params(LightGBMTrainConfig(num_threads=2))["num_threads"] == 2


def test_selection_compact_metrics_preserves_baseline_net_return():
    result = selection.compact_metrics({"total_return": 0.25, "annualized_return": None, "calmar": None})
    assert result["net_total_return"] == 0.25
    assert result["annualized_return"] is None


def test_selection_symbol_attribution_uses_capital_contributions():
    dataset = Namespace(
        symbols=["up", "down"], test_range=(0, 1), train_range=(0, 1), val_range=(0, 1),
        next_returns=np.log([[2.0], [0.5]]),
    )
    rows = selection.by_symbol_summary(dataset, np.full((2, 1), 0.5), "test", 0, 0)
    assert sum(row["pnl_contribution"] for row in rows) == pytest.approx(0.25)


def test_selection_attribution_reproduces_weight_drift_and_paid_costs():
    returns = np.log([[2.0, 1.1, 0.8], [0.5, 1.2, 1.4]])
    ledger = simulate_weight_portfolio(
        returns, np.full((2, 3), 0.4), cost_rate=0.001, rebalance_mask=np.array([True, False, True]),
    )
    dataset = Namespace(symbols=["a", "b"], train_range=(0, 3), val_range=(0, 3), test_range=(0, 3), next_returns=returns)
    rows = selection.by_symbol_summary(dataset, ledger.positions, "test", 10, 0, rebalance_interval_bars=2)
    assert sum(row["pnl_contribution"] for row in rows) == pytest.approx(ledger.equity[-1] - 1)
    assert sum(row["cost_contribution"] for row in rows) == pytest.approx(ledger.costs.sum())


def test_selection_months_use_the_return_end_timestamp():
    timestamps = pd.date_range("2025-01-31 23:58", periods=4, freq="min", tz="UTC").as_unit("ms").asi8
    dataset = Namespace(timestamps=timestamps, train_range=(0, 2), val_range=(0, 2), test_range=(0, 2))
    rows = selection.by_month_summary(dataset, np.log([1.1, 1.2]), "test")
    assert rows == [{"month": "2025-02", "net_total_return": pytest.approx(0.32)}]


def write_predictions(path, dataset, split, *, contract=True, finite=True):
    start, end = getattr(dataset, f"{split}_range")
    pred = np.full((dataset.n_symbols, end - start, len(dataset.horizons)), 0.01, dtype="float32")
    if not finite:
        pred[0, 0, 0] = np.nan
    fields = {"pred": pred}
    if contract:
        fields["data_contract"] = np.array(json.dumps(prepared_data_contract(dataset), sort_keys=True))
    np.savez(path, **fields)


@pytest.mark.parametrize("problem", ["missing_contract", "horizon_order", "shape", "nonfinite"])
def test_selection_rejects_incompatible_prediction_artifacts(tabular_dataset, tmp_path, problem):
    path = tmp_path / "pred.npz"
    write_predictions(path, tabular_dataset, "val", contract=problem != "missing_contract", finite=problem != "nonfinite")
    if problem in {"horizon_order", "shape"}:
        with np.load(path) as saved:
            fields = dict(saved)
        if problem == "horizon_order":
            contract = json.loads(str(fields["data_contract"]))
            contract["horizons"] = list(reversed(contract["horizons"]))
            fields["data_contract"] = np.array(json.dumps(contract))
        else:
            fields["pred"] = fields["pred"][:, :-1]
        np.savez(path, **fields)
    with pytest.raises(ValueError, match="prediction"):
        selection.load_predictions(path, tabular_dataset, "val")


def test_selection_cli_writes_finite_report_and_reconciles_contributions(tabular_dataset, tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    for split in ("val", "test"):
        write_predictions(out / f"tiny_lgb_{split}_predictions.npz", tabular_dataset, split)
    monkeypatch.setattr(sys, "argv", [
        "select_low_turnover_params.py", "--cache-dir", str(tmp_path / "cache"), "--out-dir", str(out),
        "--models", "tiny_lgb", "--threshold-bps", "0", "--top-k", "1",
        "--rebalance-interval-bars", "15", "--min-holding-bars", "0", "--cooldown-bars", "0",
        "--liquidity-filter", "none", "--min-select-trades", "0", "--min-select-average-position", "0",
        "--fixed-fee-bps", "1", "--output-name", "selected.json",
    ])
    selection.main()
    report = strict_json(out / "selected.json")
    assert report["baselines"]["test"]["no_trade"]["net_total_return"] == 0
    assert report["baselines"]["test"]["buy_and_hold"]["net_total_return"] is not None
    model_report = report["models"]["tiny_lgb"]
    assert model_report["searched_combinations"] == 1
    for split in ("val", "test"):
        summary = model_report["attribution"][split]["by_symbol"]
        metrics = model_report["validation_metrics" if split == "val" else "test_metrics"]
        assert sum(row["pnl_contribution"] for row in summary) == pytest.approx(metrics["net_total_return"])
        assert sum(row["cost_contribution"] for row in summary) == pytest.approx(metrics["cost_return"])
