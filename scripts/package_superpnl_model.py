#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BAR_FEATURE_NAMES = [
    "open_rel",
    "high_rel",
    "low_rel",
    "close_rel",
    "volume_z_30m",
    "amount_z_30m",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_metrics_summary(metrics: dict, model_name: str, live_horizon: str) -> dict:
    key = f"{model_name}_{live_horizon}"
    backtest = metrics.get("backtests", {}).get(key, {})
    evaluation = metrics.get("evaluation", {}).get(model_name, {}).get("horizon_metrics", {}).get(live_horizon, {})
    return {
        "model": model_name,
        "recommended_horizon": live_horizon,
        "zero_cost_test_backtest": backtest,
        "test_prediction_metrics": evaluation,
        "baseline_backtests": {
            name: value
            for name, value in metrics.get("backtests", {}).items()
            if name in {"no_trade", "buy_and_hold_equal_weight"}
        },
        "warning": (
            "Main result is zero-cost. Real-time downstream strategy must apply threshold, top-k, "
            "holding and cooldown constraints, then re-evaluate with realistic fees/slippage."
        ),
    }


def make_tarball(package_dir: Path) -> Path:
    tar_path = package_dir.with_suffix(".tar.gz")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(package_dir, arcname=package_dir.name)
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="outputs/superpnl_top20_365d_l256_h5_15_hd64_e3",
        help="Training output directory containing full_feature_tcn.pt and metrics.json.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache/okx_spot_1m_top20_365d_l256_h5_15",
        help="Prepared dataset cache containing feature schema and train normalization stats.",
    )
    parser.add_argument("--model-name", default="full_feature_tcn")
    parser.add_argument("--recommended-horizon", default="15m")
    parser.add_argument(
        "--package-dir",
        default="artifacts/superpnl_full_feature_tcn_15m_top20_20260430",
        help="Output model package directory. artifacts/ is intentionally gitignored.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = Path(args.cache_dir)
    package_dir = Path(args.package_dir)

    if package_dir.exists():
        if not args.force:
            raise FileExistsError(f"{package_dir} exists; pass --force to overwrite")
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    run_config = read_json(run_dir / "run_config.json")
    cache_metadata = read_json(cache_dir / "metadata.json")
    metrics = read_json(run_dir / "metrics.json")

    model_src = run_dir / f"{args.model_name}.pt"
    model_dst = package_dir / "model.pt"
    copy_required(model_src, model_dst)

    stats_dst = package_dir / "normalization_stats.npz"
    np.savez_compressed(
        stats_dst,
        bar_mean=np.load(cache_dir / "bar_mean.npy"),
        bar_std=np.load(cache_dir / "bar_std.npy"),
        feature_mean=np.load(cache_dir / "feature_mean.npy"),
        feature_std=np.load(cache_dir / "feature_std.npy"),
    )

    horizons = [int(h) for h in cache_metadata["horizons"]]
    recommended_horizon_minutes = int(args.recommended_horizon.removesuffix("m"))
    if recommended_horizon_minutes not in horizons:
        raise ValueError(f"recommended horizon {args.recommended_horizon} not in {horizons}")

    model_config = {
        "model_class": "superpnl.model.SuperPnLModel",
        "model_name": args.model_name,
        "use_features": True,
        "bar_dim": int(cache_metadata["bar_dim"]),
        "feature_dim": int(cache_metadata["feature_dim"]),
        "num_horizons": len(horizons),
        "hidden_dim": int(run_config["train_config"]["hidden_dim"]),
        "dropout": float(run_config["train_config"].get("dropout", 0.05)),
        "lookback": int(cache_metadata["lookback"]),
        "horizons": horizons,
        "horizon_index": {f"{h}m": i for i, h in enumerate(horizons)},
        "recommended_horizon": args.recommended_horizon,
        "recommended_horizon_index": horizons.index(recommended_horizon_minutes),
        "input_shapes": {
            "bar": ["batch", int(cache_metadata["lookback"]), int(cache_metadata["bar_dim"])],
            "features": ["batch", int(cache_metadata["lookback"]), int(cache_metadata["feature_dim"])],
        },
    }

    feature_schema = {
        "bar_feature_names": BAR_FEATURE_NAMES,
        "feature_names": cache_metadata["feature_names"],
        "feature_windows_minutes": cache_metadata["config"]["feature_windows"],
        "bar_size": "1m",
        "normalization": {
            "stats_file": "normalization_stats.npz",
            "bar": {"mean_key": "bar_mean", "std_key": "bar_std"},
            "features": {"mean_key": "feature_mean", "std_key": "feature_std"},
            "fit_scope": "train split only",
        },
        "leakage_constraints": [
            "Use only bars with timestamp <= decision timestamp t.",
            "All rolling, EMA and cross-section rank features must be computed causally.",
            "Do not refit normalization stats online or on validation/test/live data.",
            "Do not use future volume, future slippage, centered rolling windows or future universe membership.",
        ],
    }

    universe = {
        "symbols_superpnl": cache_metadata["symbols"],
        "symbols_bitpro": [symbol.replace("-", "/") for symbol in cache_metadata["symbols"]],
        "symbol_mapping": {symbol.replace("-", "/"): symbol for symbol in cache_metadata["symbols"]},
        "selection_note": (
            "OKX spot *-USDT non-stablecoin Top20 at download time, requiring full 365d 1m history coverage. "
            "Keep this universe fixed for reproducible inference/backtests."
        ),
    }

    data_contract = {
        "decision_time": "At confirmed 1m bar t, generate features from bars <= t.",
        "entry_exit_label_used_in_training": "label_h = log(open_{t+h+1} / open_{t+1})",
        "live_prediction_output": {
            "pred_ret": "model pred_ret[:, recommended_horizon_index]",
            "pos_score": "sigmoid(model pos_logit[:, recommended_horizon_index])",
            "score_bps": "pred_ret * 10000",
        },
        "minimum_live_history": {
            "lookback_bars": int(cache_metadata["lookback"]),
            "extra_for_rolling_features": max(int(x) for x in cache_metadata["config"]["feature_windows"]),
            "recommended_warmup_bars": int(cache_metadata["lookback"]) + max(int(x) for x in cache_metadata["config"]["feature_windows"]),
        },
        "not_included": [
            "raw market data",
            "historical test predictions",
            "checkpoint optimizer state",
            "orderbook/cost/liquidity features",
        ],
    }

    write_json(package_dir / "model_config.json", model_config)
    write_json(package_dir / "feature_schema.json", feature_schema)
    write_json(package_dir / "universe.json", universe)
    write_json(package_dir / "data_contract.json", data_contract)
    write_json(package_dir / "metrics_summary.json", build_metrics_summary(metrics, args.model_name, args.recommended_horizon))

    readme = f"""# SuperPnL Model Package

This package is for real-time SuperPnL inference, not for reading historical predictions.

Recommended model:

```text
model={args.model_name}
horizon={args.recommended_horizon}
lookback={cache_metadata["lookback"]}
bar_size=1m
```

Files:

- `model.pt`: PyTorch state dict package from training output.
- `model_config.json`: architecture and horizon mapping.
- `feature_schema.json`: exact bar/features order and normalization contract.
- `normalization_stats.npz`: train-split mean/std arrays.
- `universe.json`: fixed Top20 symbol list and BitPro symbol mapping.
- `data_contract.json`: real-time inference contract.
- `metrics_summary.json`: test metrics summary and zero-cost warning.
- `manifest.json`: hashes and source paths.

Live usage:

1. Keep a rolling 1m history window for every symbol in `universe.json`.
2. After a 1m bar is confirmed, compute features using only data `<= t`.
3. Standardize using `normalization_stats.npz`.
4. Load `model.pt` with `SuperPnLModel` from `model_config.json`.
5. Read `pred_ret` at `recommended_horizon_index`.
6. Strategy layer applies threshold/top-k/min-holding/cooldown/cost controls before order placement.

Historical prediction `.npz` files are intentionally not part of this package. They are only useful for offline regression tests, not for simulation/live trading.
"""
    (package_dir / "README.md").write_text(readme)

    files = sorted(path for path in package_dir.iterdir() if path.is_file())
    manifest = {
        "package_name": package_dir.name,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_run_dir": str(run_dir),
        "source_cache_dir": str(cache_dir),
        "model_source": str(model_src),
        "model_sha256": sha256_file(model_src),
        "package_files": {path.name: sha256_file(path) for path in files},
        "git_note": "This package is written under artifacts/ and must not be committed.",
    }
    write_json(package_dir / "manifest.json", manifest)

    tar_path = make_tarball(package_dir)
    print(json.dumps({"package_dir": str(package_dir), "tarball": str(tar_path)}, indent=2))


if __name__ == "__main__":
    main()
