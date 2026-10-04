import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superpnl.data import DatasetConfig, load_raw_bars, prepare_dataset


def make_raw_frame(n: int = 300) -> pd.DataFrame:
    """Deterministic, complete one-minute bars for dataset regressions."""
    price = 100.0 + np.arange(n, dtype=float) * 0.1
    return pd.DataFrame(
        {
            "timestamp": 1_700_000_040_000 + np.arange(n, dtype=np.int64) * 60_000,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": np.arange(n, dtype=float) + 100,
            "amount": (np.arange(n, dtype=float) + 100) * price,
        }
    )


def write_raw_frame(raw_dir: Path, frame: pd.DataFrame, symbol: str = "BTC-USDT") -> None:
    csv_dir = raw_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_dir / f"{symbol}.csv.gz", index=False, compression="gzip")


def test_next_returns_begin_at_next_open_after_decision(tmp_path: Path) -> None:
    frame = make_raw_frame()
    decision = 100
    frame.loc[:, ["open", "close"]] = 100.0
    frame.loc[:, "high"] = 101.0
    frame.loc[:, "low"] = 99.0
    # A jump from today's close to the next open happens before our entry.
    frame.loc[decision + 1 :, ["open", "close"]] = 200.0
    frame.loc[decision + 1 :, "high"] = 201.0
    frame.loc[decision + 1 :, "low"] = 199.0
    raw_dir = tmp_path / "raw"
    write_raw_frame(raw_dir, frame)
    dataset = prepare_dataset(
        DatasetConfig(str(raw_dir), str(tmp_path / "cache"), lookback=32, horizons=(1, 5))
    )

    assert dataset.next_returns[0, decision] == pytest.approx(0.0)
    assert dataset.next_returns[0, decision - 1] == pytest.approx(np.log(2.0))
    # The one-minute backtest return and one-minute supervised target agree.
    np.testing.assert_allclose(dataset.next_returns[0, :-2], dataset.labels[0, :-2, 0])


@pytest.mark.parametrize(
    ("earlier_split", "later_split"),
    [("train_range", "val_range"), ("val_range", "test_range")],
)
def test_future_split_prices_do_not_change_earlier_samples(
    tmp_path: Path, earlier_split: str, later_split: str
) -> None:
    raw_dir = tmp_path / "raw"
    frame = make_raw_frame()
    write_raw_frame(raw_dir, frame)
    config = DatasetConfig(str(raw_dir), str(tmp_path / "cache"), lookback=32, horizons=(5, 15))
    before = prepare_dataset(config)
    later_start, _ = getattr(before, later_split)
    frame.loc[later_start:, ["open", "high", "low", "close"]] *= 2
    write_raw_frame(raw_dir, frame)
    after = prepare_dataset(config)
    start, end = getattr(before, earlier_split)

    np.testing.assert_array_equal(before.labels[:, start:end], after.labels[:, start:end])
    np.testing.assert_array_equal(before.next_returns[:, start:end], after.next_returns[:, start:end])
    # Include each sample's entire historical input window and normalization.
    input_start = start - before.lookback + 1
    np.testing.assert_array_equal(
        before.bar_inputs[:, input_start:end], after.bar_inputs[:, input_start:end]
    )
    np.testing.assert_array_equal(
        before.feature_inputs[:, input_start:end], after.feature_inputs[:, input_start:end]
    )
    assert end - 1 + max(before.horizons) + 1 < later_start


def test_dataset_rejects_empty_split_after_label_purge(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_raw_frame(raw_dir, make_raw_frame(n=100))
    config = DatasetConfig(str(raw_dir), str(tmp_path / "cache"), lookback=32, horizons=(5, 15))

    with pytest.raises(ValueError, match="empty.*split.*purge"):
        prepare_dataset(config)


@pytest.mark.parametrize("both_symbols", [False, True])
def test_raw_bars_reject_missing_minutes_even_with_low_coverage_threshold(
    tmp_path: Path, both_symbols: bool
) -> None:
    frame = make_raw_frame()
    write_raw_frame(tmp_path, frame.iloc[::2] if both_symbols else frame)
    write_raw_frame(tmp_path, frame.iloc[::2], "ETH-USDT")

    with pytest.raises(ValueError, match="one-minute grid"):
        load_raw_bars(tmp_path, min_coverage=0.1)


def test_raw_bars_require_every_metadata_symbol(tmp_path: Path) -> None:
    write_raw_frame(tmp_path, make_raw_frame())
    (tmp_path / "metadata.json").write_text(json.dumps({"symbols": ["BTC-USDT", "ETH-USDT"]}))

    with pytest.raises(FileNotFoundError, match="ETH-USDT"):
        load_raw_bars(tmp_path)


@pytest.mark.parametrize("boundary", ["start_ms", "end_ms"])
def test_raw_bars_enforce_declared_time_boundaries(tmp_path: Path, boundary: str) -> None:
    frame = make_raw_frame()
    write_raw_frame(tmp_path, frame.iloc[1:-1])
    value = int(frame.timestamp.iloc[0 if boundary == "start_ms" else -1])
    (tmp_path / "metadata.json").write_text(json.dumps({boundary: value}))

    with pytest.raises(ValueError, match=boundary):
        load_raw_bars(tmp_path)


@pytest.mark.parametrize(
    ("column", "value"),
    [("close", np.nan), ("open", np.inf), ("close", 0.0), ("low", -1.0), ("volume", -1.0), ("amount", -1.0)],
)
def test_raw_bars_reject_invalid_values_without_dropping_rows(
    tmp_path: Path, column: str, value: float
) -> None:
    frame = make_raw_frame()
    frame.loc[50, column] = value
    write_raw_frame(tmp_path, frame)

    with pytest.raises(ValueError, match="invalid OHLCV"):
        load_raw_bars(tmp_path)


@pytest.mark.parametrize("corruption", ["duplicate", "off_minute", "different_extent", "bad_ohlc"])
def test_raw_bars_reject_ambiguous_timeline_or_candles(tmp_path: Path, corruption: str) -> None:
    frame = make_raw_frame()
    write_raw_frame(tmp_path, frame)
    bad = frame.copy()
    if corruption == "duplicate":
        bad = pd.concat([bad, bad.iloc[[50]]], ignore_index=True)
    elif corruption == "off_minute":
        bad.loc[50, "timestamp"] += 1
    elif corruption == "different_extent":
        bad = bad.iloc[1:]
    else:
        bad.loc[50, "high"] = bad.loc[50, "low"] - 1
    write_raw_frame(tmp_path, bad, "ETH-USDT")

    with pytest.raises(ValueError):
        load_raw_bars(tmp_path)


def test_raw_bars_keep_complete_minutes_and_symbol_order(tmp_path: Path) -> None:
    frame = make_raw_frame()
    write_raw_frame(tmp_path, frame)
    write_raw_frame(tmp_path, frame, "ETH-USDT")
    metadata = {
        "symbols": ["ETH-USDT", "BTC-USDT"],
        "start_ms": int(frame.timestamp.iloc[0]),
        "end_ms": int(frame.timestamp.iloc[-1]),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    symbols, frames = load_raw_bars(tmp_path)

    assert symbols == metadata["symbols"]
    for symbol in symbols:
        pd.testing.assert_frame_equal(frames[symbol], frame)
