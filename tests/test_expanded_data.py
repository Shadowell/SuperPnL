import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superpnl.data import DatasetConfig, _rolling_beta, load_prepared_dataset, prepare_dataset
from superpnl.provenance import cache_data_contract, prepared_data_contract, validate_cache
from test_data import make_raw_frame, write_raw_frame


def expanded_frame(phase: float = 0.0) -> pd.DataFrame:
    frame = make_raw_frame(420)
    t = np.arange(len(frame))
    close = 100 * np.exp(np.cumsum(0.0001 + 0.002 * np.sin(t / 11 + phase)))
    frame["close"] = close
    frame["open"] = close * 0.9997
    frame["high"] = close * 1.001
    frame["low"] = close * 0.999
    frame["amount"] = frame.volume * close
    return frame


def expanded_config(tmp_path: Path) -> DatasetConfig:
    raw = tmp_path / "raw"
    write_raw_frame(raw, expanded_frame(), "BTC-USDT")
    write_raw_frame(raw, expanded_frame(0.7), "ETH-USDT")
    return DatasetConfig(
        str(raw), str(tmp_path / "cache"), lookback=32,
        horizons=(5, 15), feature_windows=(5, 15, 30), factor_set="expanded",
    )


def test_rolling_beta_against_itself_is_one():
    returns = pd.Series([0.01, 0.02, -0.03, 0.04, -0.02, 0.015, 0.03, -0.01, 0.005, 0.012])

    np.testing.assert_allclose(_rolling_beta(returns, returns, 5).dropna(), 1.0, rtol=1e-12)


def test_expanded_cache_roundtrip_binds_factor_set_and_feature_order(tmp_path: Path):
    config = expanded_config(tmp_path)
    dataset = prepare_dataset(config)
    restored = load_prepared_dataset(config.cache_dir)
    metadata = json.loads((Path(config.cache_dir) / "metadata.json").read_text())
    validate_cache(json.loads(config.to_json()), metadata)
    contract = prepared_data_contract(dataset)

    assert contract["preparation_config"]["factor_set"] == "expanded"
    assert contract == prepared_data_contract(restored) == cache_data_contract(config.cache_dir)
    assert dataset.feature_names == restored.feature_names
    assert {"amihud_5m", "eth_beta_15m", "btc_resid_ret_30m", "market_dispersion_15m"} <= set(dataset.feature_names)
    assert np.isfinite(dataset.feature_inputs).all()
    assert np.all(np.diff(dataset.timestamps) == 60_000)
    for earlier, later in [(dataset.train_range, dataset.val_range), (dataset.val_range, dataset.test_range)]:
        assert earlier[1] + max(dataset.horizons) < later[0]
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_cache(json.loads(replace(config, factor_set="base").to_json()), metadata)


@pytest.mark.parametrize("earlier_split,later_split", [("train_range", "val_range"), ("val_range", "test_range")])
def test_expanded_features_and_labels_do_not_read_future_partitions(tmp_path, earlier_split, later_split):
    config = expanded_config(tmp_path)
    before = prepare_dataset(config)
    later_start, _ = getattr(before, later_split)
    for symbol, phase in [("BTC-USDT", 0.0), ("ETH-USDT", 0.7)]:
        frame = expanded_frame(phase)
        frame.loc[later_start:, ["open", "high", "low", "close"]] *= 2
        frame.loc[later_start:, "volume"] *= 3
        frame.loc[later_start:, "amount"] *= 6
        write_raw_frame(Path(config.raw_dir), frame, symbol)
    after = prepare_dataset(config)
    start, end = getattr(before, earlier_split)

    np.testing.assert_array_equal(before.labels[:, start:end], after.labels[:, start:end])
    np.testing.assert_array_equal(
        before.feature_inputs[:, start - before.lookback + 1 : end],
        after.feature_inputs[:, start - before.lookback + 1 : end],
    )
    np.testing.assert_array_equal(before.feature_mean, after.feature_mean)
    np.testing.assert_array_equal(before.feature_std, after.feature_std)


def test_expanded_mode_rejects_missing_minutes(tmp_path):
    config = expanded_config(tmp_path)
    write_raw_frame(Path(config.raw_dir), expanded_frame().drop(index=100), "BTC-USDT")

    with pytest.raises(ValueError, match="one-minute grid"):
        prepare_dataset(config)
