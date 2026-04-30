from __future__ import annotations

import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BAR_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
DEFAULT_FEATURE_WINDOWS = (5, 15, 30)
DEFAULT_HORIZONS = (5, 15)


@dataclass(frozen=True)
class DatasetConfig:
    raw_dir: str
    cache_dir: str
    lookback: int = 256
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    feature_windows: tuple[int, ...] = DEFAULT_FEATURE_WINDOWS
    min_coverage: float = 0.995

    def to_json(self) -> str:
        data = asdict(self)
        data["horizons"] = list(self.horizons)
        data["feature_windows"] = list(self.feature_windows)
        return json.dumps(data, indent=2)


@dataclass
class PreparedDataset:
    symbols: list[str]
    timestamps: np.ndarray
    bar_inputs: np.ndarray
    feature_inputs: np.ndarray
    labels: np.ndarray
    next_returns: np.ndarray
    realized_horizon_returns: np.ndarray
    feature_names: list[str]
    horizons: tuple[int, ...]
    lookback: int
    train_range: tuple[int, int]
    val_range: tuple[int, int]
    test_range: tuple[int, int]
    bar_mean: np.ndarray | None = None
    bar_std: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_times(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def bar_dim(self) -> int:
        return int(self.bar_inputs.shape[-1])

    @property
    def feature_dim(self) -> int:
        return int(self.feature_inputs.shape[-1])


def read_metadata(raw_dir: Path) -> dict:
    path = raw_dir / "metadata.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _read_symbol_csv(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", newline="") as f:
        df = pd.read_csv(f)
    required = {"timestamp", "open", "high", "low", "close", "volume", "amount"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[["timestamp", "open", "high", "low", "close", "volume", "amount"]].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    for col in BAR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", *BAR_COLUMNS])
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return df


def load_raw_bars(raw_dir: str | Path, min_coverage: float = 0.995) -> tuple[list[str], dict[str, pd.DataFrame]]:
    raw = Path(raw_dir)
    csv_dir = raw / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"missing csv dir: {csv_dir}")
    metadata = read_metadata(raw)
    symbols = metadata.get("symbols") or [p.stem.replace(".csv", "") for p in sorted(csv_dir.glob("*.csv.gz"))]
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = csv_dir / f"{symbol}.csv.gz"
        if not path.exists():
            continue
        frames[symbol] = _read_symbol_csv(path)
    if not frames:
        raise ValueError(f"no symbol CSV files found under {csv_dir}")

    timestamp_sets = [set(frame["timestamp"].to_numpy()) for frame in frames.values()]
    common = sorted(set.intersection(*timestamp_sets))
    if not common:
        raise ValueError("symbols have no common timestamps")
    common_index = pd.Index(common, name="timestamp")
    aligned: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        reindexed = frame.set_index("timestamp").reindex(common_index)
        coverage = 1.0 - float(reindexed[BAR_COLUMNS].isna().any(axis=1).mean())
        if coverage < min_coverage:
            raise ValueError(f"{symbol} coverage {coverage:.4f} below min_coverage={min_coverage}")
        reindexed = reindexed.ffill().bfill()
        aligned[symbol] = reindexed.reset_index()
    return list(aligned.keys()), aligned


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    denom = gain + loss
    rsi = gain / denom.replace(0, np.nan)
    return (rsi.fillna(0.5) - 0.5).astype("float32")


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return ((series - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _symbol_features(df: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    close = df["close"]
    log_close = np.log(close)
    log_ret_1m = log_close.diff()
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"ret_{w}m"] = log_close - log_close.shift(w)
    for w in windows:
        out[f"rsi_{w}m"] = _rsi(close, w)
    for w in windows:
        out[f"vol_std_{w}m"] = log_ret_1m.rolling(w, min_periods=w).std(ddof=0)
    for w in windows:
        ma = close.rolling(w, min_periods=w).mean()
        out[f"ma_dev_{w}m"] = close / ma.replace(0, np.nan) - 1
    for w in windows:
        ma = close.rolling(w, min_periods=w).mean()
        std = close.rolling(w, min_periods=w).std(ddof=0)
        out[f"boll_z_{w}m"] = (close - ma) / (2 * std.replace(0, np.nan))
    if 5 in windows and 15 in windows:
        out["macd_5m_15m"] = (close.ewm(span=5, adjust=False).mean() - close.ewm(span=15, adjust=False).mean()) / close
    if 15 in windows and 30 in windows:
        out["macd_15m_30m"] = (close.ewm(span=15, adjust=False).mean() - close.ewm(span=30, adjust=False).mean()) / close
    return out


def _bar_inputs(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["close"].shift(1).fillna(df["close"])
    ret = np.log(df["close"] / prev_close.replace(0, np.nan))
    out = pd.DataFrame(index=df.index)
    out["open_rel"] = np.log(df["open"] / prev_close.replace(0, np.nan))
    out["high_rel"] = np.log(df["high"] / prev_close.replace(0, np.nan))
    out["low_rel"] = np.log(df["low"] / prev_close.replace(0, np.nan))
    out["close_rel"] = ret
    out["volume_z_30m"] = _zscore(np.log1p(df["volume"]), 30)
    out["amount_z_30m"] = _zscore(np.log1p(df["amount"]), 30)
    return out


def _time_features(timestamps: np.ndarray) -> pd.DataFrame:
    dt_index = pd.to_datetime(timestamps, unit="ms", utc=True)
    hour = dt_index.hour + dt_index.minute / 60.0
    day = dt_index.dayofweek.to_numpy(dtype="float64")
    data = {
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dayofweek_sin": np.sin(2 * np.pi * day / 7.0),
        "dayofweek_cos": np.cos(2 * np.pi * day / 7.0),
    }
    return pd.DataFrame(data)


def _standardize_train(data: np.ndarray, train_slice: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = data[:, train_slice, :].reshape(-1, data.shape[-1])
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    data = (data - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return data, mean.astype("float32"), std.astype("float32")


def prepare_dataset(config: DatasetConfig) -> PreparedDataset:
    symbols, frames = load_raw_bars(config.raw_dir, config.min_coverage)
    timestamps = frames[symbols[0]]["timestamp"].to_numpy(dtype="int64")
    n_symbols = len(symbols)
    n_times = len(timestamps)
    windows = tuple(config.feature_windows)

    bar_blocks = []
    feature_blocks = []
    label_blocks = []
    next_return_blocks = []
    realized_horizon_blocks = []
    symbol_feature_frames: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        df = frames[symbol]
        bars = _bar_inputs(df)
        symbol_features = _symbol_features(df, windows)
        symbol_feature_frames[symbol] = symbol_features
        close = df["close"]
        open_ = df["open"]
        labels = []
        for horizon in config.horizons:
            entry = open_.shift(-1)
            exit_ = open_.shift(-(horizon + 1))
            labels.append(np.log(exit_ / entry.replace(0, np.nan)).to_numpy(dtype="float32"))
        labels_arr = np.stack(labels, axis=-1)
        next_return = np.log(close.shift(-1) / close.replace(0, np.nan)).to_numpy(dtype="float32")
        realized_horizon_blocks.append(labels_arr)
        label_blocks.append(labels_arr)
        next_return_blocks.append(next_return)
        bar_blocks.append(bars)

    feature_names = list(next(iter(symbol_feature_frames.values())).columns)
    # Market context uses BTC as benchmark.
    btc_symbol = "BTC-USDT" if "BTC-USDT" in symbol_feature_frames else symbols[0]
    btc_features = symbol_feature_frames[btc_symbol]
    market = pd.DataFrame(index=btc_features.index)
    for w in windows:
        market[f"market_ret_{w}m"] = btc_features[f"ret_{w}m"]
        market[f"market_vol_{w}m"] = btc_features[f"vol_std_{w}m"]

    # Cross-section ranks for ret and volatility windows.
    for w in windows:
        ret_matrix = pd.DataFrame({sym: symbol_feature_frames[sym][f"ret_{w}m"] for sym in symbols})
        vol_matrix = pd.DataFrame({sym: symbol_feature_frames[sym][f"vol_std_{w}m"] for sym in symbols})
        ret_rank = ret_matrix.rank(axis=1, pct=True) - 0.5
        vol_rank = vol_matrix.rank(axis=1, pct=True) - 0.5
        for sym in symbols:
            symbol_feature_frames[sym][f"cross_section_ret_rank_{w}m"] = ret_rank[sym]
            symbol_feature_frames[sym][f"cross_section_vol_rank_{w}m"] = vol_rank[sym]

    time_frame = _time_features(timestamps)
    extra_names = list(market.columns) + [
        f"cross_section_ret_rank_{w}m" for w in windows
    ] + [
        f"cross_section_vol_rank_{w}m" for w in windows
    ] + list(time_frame.columns)

    combined_feature_blocks = []
    for symbol in symbols:
        features = symbol_feature_frames[symbol].copy()
        for col in market.columns:
            features[col] = market[col].to_numpy()
        for col in time_frame.columns:
            features[col] = time_frame[col].to_numpy()
        combined_feature_blocks.append(features)

    feature_names = list(combined_feature_blocks[0].columns)
    bar_names = list(bar_blocks[0].columns)

    bar_inputs = np.stack([block[bar_names].to_numpy(dtype="float32") for block in bar_blocks], axis=0)
    feature_inputs = np.stack(
        [block[feature_names].to_numpy(dtype="float32") for block in combined_feature_blocks], axis=0
    )
    labels = np.stack(label_blocks, axis=0).astype("float32")
    next_returns = np.stack(next_return_blocks, axis=0).astype("float32")
    realized_horizon_returns = np.stack(realized_horizon_blocks, axis=0).astype("float32")

    max_horizon = max(config.horizons)
    valid_start = max(config.lookback, max(windows) + 5)
    valid_end = n_times - max_horizon - 2
    if valid_end <= valid_start:
        raise ValueError("not enough rows after lookback/horizon filtering")

    train_end = valid_start + int((valid_end - valid_start) * 0.70)
    val_end = valid_start + int((valid_end - valid_start) * 0.85)
    train_range = (valid_start, train_end)
    val_range = (train_end, val_end)
    test_range = (val_end, valid_end)

    bar_inputs, bar_mean, bar_std = _standardize_train(bar_inputs, slice(*train_range))
    feature_inputs, feature_mean, feature_std = _standardize_train(feature_inputs, slice(*train_range))
    labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    next_returns = np.nan_to_num(next_returns, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    realized_horizon_returns = np.nan_to_num(
        realized_horizon_returns, nan=0.0, posinf=0.0, neginf=0.0
    ).astype("float32")

    dataset = PreparedDataset(
        symbols=symbols,
        timestamps=timestamps,
        bar_inputs=bar_inputs,
        feature_inputs=feature_inputs,
        labels=labels,
        next_returns=next_returns,
        realized_horizon_returns=realized_horizon_returns,
        feature_names=feature_names,
        horizons=tuple(config.horizons),
        lookback=config.lookback,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        bar_mean=bar_mean,
        bar_std=bar_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    save_prepared_dataset(dataset, config.cache_dir, config)
    return dataset


def save_prepared_dataset(dataset: PreparedDataset, cache_dir: str | Path, config: DatasetConfig | None = None) -> None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "timestamps.npy", dataset.timestamps)
    np.save(cache / "bar_inputs.npy", dataset.bar_inputs)
    np.save(cache / "feature_inputs.npy", dataset.feature_inputs)
    np.save(cache / "labels.npy", dataset.labels)
    np.save(cache / "next_returns.npy", dataset.next_returns)
    np.save(cache / "realized_horizon_returns.npy", dataset.realized_horizon_returns)
    if dataset.bar_mean is not None:
        np.save(cache / "bar_mean.npy", dataset.bar_mean)
    if dataset.bar_std is not None:
        np.save(cache / "bar_std.npy", dataset.bar_std)
    if dataset.feature_mean is not None:
        np.save(cache / "feature_mean.npy", dataset.feature_mean)
    if dataset.feature_std is not None:
        np.save(cache / "feature_std.npy", dataset.feature_std)
    metadata = {
        "symbols": dataset.symbols,
        "feature_names": dataset.feature_names,
        "horizons": list(dataset.horizons),
        "lookback": dataset.lookback,
        "train_range": list(dataset.train_range),
        "val_range": list(dataset.val_range),
        "test_range": list(dataset.test_range),
        "n_symbols": dataset.n_symbols,
        "n_times": dataset.n_times,
        "bar_dim": dataset.bar_dim,
        "feature_dim": dataset.feature_dim,
        "bar_stats_path": "bar_mean.npy/bar_std.npy",
        "feature_stats_path": "feature_mean.npy/feature_std.npy",
    }
    if config is not None:
        metadata["config"] = json.loads(config.to_json())
    (cache / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def load_prepared_dataset(cache_dir: str | Path, mmap: bool = True) -> PreparedDataset:
    cache = Path(cache_dir)
    mode = "r" if mmap else None
    metadata = json.loads((cache / "metadata.json").read_text())
    bar_mean = np.load(cache / "bar_mean.npy", mmap_mode=mode) if (cache / "bar_mean.npy").exists() else None
    bar_std = np.load(cache / "bar_std.npy", mmap_mode=mode) if (cache / "bar_std.npy").exists() else None
    feature_mean = (
        np.load(cache / "feature_mean.npy", mmap_mode=mode) if (cache / "feature_mean.npy").exists() else None
    )
    feature_std = (
        np.load(cache / "feature_std.npy", mmap_mode=mode) if (cache / "feature_std.npy").exists() else None
    )
    return PreparedDataset(
        symbols=list(metadata["symbols"]),
        timestamps=np.load(cache / "timestamps.npy", mmap_mode=mode),
        bar_inputs=np.load(cache / "bar_inputs.npy", mmap_mode=mode),
        feature_inputs=np.load(cache / "feature_inputs.npy", mmap_mode=mode),
        labels=np.load(cache / "labels.npy", mmap_mode=mode),
        next_returns=np.load(cache / "next_returns.npy", mmap_mode=mode),
        realized_horizon_returns=np.load(cache / "realized_horizon_returns.npy", mmap_mode=mode),
        feature_names=list(metadata["feature_names"]),
        horizons=tuple(int(x) for x in metadata["horizons"]),
        lookback=int(metadata["lookback"]),
        train_range=tuple(int(x) for x in metadata["train_range"]),
        val_range=tuple(int(x) for x in metadata["val_range"]),
        test_range=tuple(int(x) for x in metadata["test_range"]),
        bar_mean=bar_mean,
        bar_std=bar_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


class WindowBatcher:
    def __init__(
        self,
        dataset: PreparedDataset,
        split: str,
        use_features: bool,
        batch_size: int,
        samples_per_epoch: int | None,
        seed: int = 17,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.use_features = use_features
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)
        ranges = {
            "train": dataset.train_range,
            "val": dataset.val_range,
            "test": dataset.test_range,
        }
        if split not in ranges:
            raise ValueError(f"unknown split: {split}")
        self.start, self.end = ranges[split]
        self.indices = self._make_indices()

    def _make_indices(self) -> np.ndarray:
        times = np.arange(self.start, self.end, dtype=np.int64)
        symbols = np.arange(self.dataset.n_symbols, dtype=np.int64)
        sym_grid, time_grid = np.meshgrid(symbols, times, indexing="ij")
        return np.stack([sym_grid.ravel(), time_grid.ravel()], axis=1)

    def __len__(self) -> int:
        n = len(self.indices) if self.samples_per_epoch is None else min(self.samples_per_epoch, len(self.indices))
        return int(math.ceil(n / self.batch_size))

    def iter_batches(self, shuffle: bool = True):
        if self.samples_per_epoch is None or self.samples_per_epoch >= len(self.indices):
            selected = self.indices.copy()
            if shuffle:
                self.rng.shuffle(selected)
        else:
            choice = self.rng.choice(len(self.indices), size=self.samples_per_epoch, replace=False)
            selected = self.indices[choice]
            if shuffle:
                self.rng.shuffle(selected)
        lookback = self.dataset.lookback
        offsets = np.arange(-lookback + 1, 1, dtype=np.int64)
        for offset in range(0, len(selected), self.batch_size):
            batch_idx = selected[offset : offset + self.batch_size]
            sym_idx = batch_idx[:, 0].astype("int64")
            time_idx = batch_idx[:, 1].astype("int64")
            window_times = time_idx[:, None] + offsets[None, :]
            bar = self.dataset.bar_inputs[sym_idx[:, None], window_times, :].astype("float32", copy=False)
            if self.use_features:
                feat = self.dataset.feature_inputs[sym_idx[:, None], window_times, :].astype("float32", copy=False)
            else:
                feat = np.zeros((len(batch_idx), lookback, 0), dtype="float32")
            labels = self.dataset.labels[sym_idx, time_idx, :].astype("float32", copy=False)
            yield bar, feat, labels, sym_idx, time_idx
