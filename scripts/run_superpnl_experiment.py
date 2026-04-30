#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from superpnl.data import DatasetConfig, load_prepared_dataset, prepare_dataset
from superpnl.training import (
    TrainConfig,
    backtest_buy_and_hold,
    backtest_rule_momentum,
    backtest_scores,
    choose_device,
    evaluate_model,
    no_trade_metrics,
    train_model,
)


def ensure_dataset(args) -> object:
    cache_dir = Path(args.cache_dir)
    if (cache_dir / "metadata.json").exists() and not args.rebuild_cache:
        return load_prepared_dataset(cache_dir, mmap=True)
    config = DatasetConfig(
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        lookback=args.lookback,
        horizons=tuple(int(x) for x in args.horizons.split(",")),
        feature_windows=tuple(int(x) for x in args.feature_windows.split(",")),
    )
    return prepare_dataset(config)


def format_metric(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def split_datetime(dataset, split: str) -> tuple[str, str]:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    ts = pd.to_datetime(dataset.timestamps[[start, end - 1]], unit="ms", utc=True)
    return ts[0].strftime("%Y-%m-%d %H:%M UTC"), ts[1].strftime("%Y-%m-%d %H:%M UTC")


def by_symbol_summary(dataset, positions: np.ndarray, split: str) -> list[dict]:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    next_returns = dataset.next_returns[:, start:end].astype("float64")
    out = []
    for i, symbol in enumerate(dataset.symbols):
        log_sum = float(np.nansum(positions[i] * next_returns[i]))
        out.append(
            {
                "symbol": symbol,
                "total_return": float(np.exp(log_sum) - 1.0),
                "avg_position": float(np.nanmean(positions[i])),
            }
        )
    return sorted(out, key=lambda item: item["total_return"], reverse=True)


def by_month_summary(dataset, portfolio_returns: np.ndarray, split: str) -> list[dict]:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    ts = pd.to_datetime(dataset.timestamps[start:end], unit="ms", utc=True)
    frame = pd.DataFrame({"month": ts.strftime("%Y-%m"), "ret": portfolio_returns})
    rows = []
    for month, group in frame.groupby("month"):
        rows.append({"month": month, "total_return": float(np.exp(group["ret"].sum()) - 1.0)})
    return rows


def write_report(out_dir: Path, dataset, results: dict) -> None:
    report = []
    report.append("# SuperPnL Top20 12个月训练与回测报告")
    report.append("")
    report.append("## 数据")
    report.append("")
    report.append(f"- symbols: `{', '.join(dataset.symbols)}`")
    report.append(f"- bars: `{dataset.n_times}` 1min timestamps")
    report.append(f"- train: `{split_datetime(dataset, 'train')[0]}` -> `{split_datetime(dataset, 'train')[1]}`")
    report.append(f"- val: `{split_datetime(dataset, 'val')[0]}` -> `{split_datetime(dataset, 'val')[1]}`")
    report.append(f"- test: `{split_datetime(dataset, 'test')[0]}` -> `{split_datetime(dataset, 'test')[1]}`")
    report.append(f"- lookback: `{dataset.lookback}`")
    report.append(f"- horizons: `{', '.join(str(h) + 'm' for h in dataset.horizons)}`")
    report.append(f"- feature_dim: `{dataset.feature_dim}`")
    report.append(f"- cost: `fee={results['cost_config']['fixed_fee_bps']}bps, slippage={results['cost_config']['fixed_slippage_bps']}bps`")
    report.append(f"- threshold: `{results['cost_config']['threshold_bps']}bps`")
    report.append("")
    report.append("## 训练过程")
    report.append("")
    report.append("| model | epoch | train_loss | val_mae_mean | val_rank_ic_mean | elapsed_sec |")
    report.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model_name, history in results["training"].items():
        for row in history:
            report.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        str(row["epoch"]),
                        format_metric(row["train_loss"]),
                        format_metric(row["val_mae_mean"]),
                        format_metric(row["val_rank_ic_mean"]),
                        format_metric(row["elapsed_sec"]),
                    ]
                )
                + " |"
            )
    report.append("")
    report.append("## 回测对比")
    report.append("")
    table_rows = []
    for name, metrics in results["backtests"].items():
        table_rows.append(
            [
                name,
                metrics.get("horizon", "-"),
                format_metric(metrics["total_return"]),
                format_metric(metrics["annualized_return"]),
                format_metric(metrics["sharpe"]),
                format_metric(metrics["max_drawdown"]),
                format_metric(metrics["turnover"]),
                format_metric(metrics["average_position"]),
                format_metric(metrics["trade_count"]),
            ]
        )
    report.append("| model | horizon | total_return | annualized | sharpe | max_drawdown | turnover | avg_pos | trades |")
    report.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in table_rows:
        report.append("| " + " | ".join(row) + " |")
    report.append("")
    report.append("## 预测指标")
    report.append("")
    report.append("| model | horizon | mae | rmse | hit_rate | rank_ic | rank_icir |")
    report.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model_name, eval_result in results["evaluation"].items():
        for horizon, metric in eval_result["horizon_metrics"].items():
            report.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        horizon,
                        format_metric(metric["mae"]),
                        format_metric(metric["rmse"]),
                        format_metric(metric["direction_hit_rate"]),
                        format_metric(metric["rank_ic"]),
                        format_metric(metric["rank_icir"]),
                    ]
                )
                + " |"
            )
    report.append("")
    report.append("## 稳定性")
    report.append("")
    for model_name, stability in results["stability"].items():
        report.append(f"### {model_name}")
        report.append("")
        report.append("Top symbols:")
        for row in stability["by_symbol_top"][:10]:
            report.append(f"- `{row['symbol']}` total_return={row['total_return']:.4f}, avg_pos={row['avg_position']:.4f}")
        report.append("")
        report.append("Monthly returns:")
        for row in stability["by_month"]:
            report.append(f"- `{row['month']}` total_return={row['total_return']:.4f}")
        report.append("")
    report.append("## 下游使用")
    report.append("")
    report.append("1. 使用同一套 `feature_windows` 和 `strategy_horizons` 生成线上特征。")
    report.append("2. 每分钟收盘后，用最近 `lookback=256` 根 1min bar 生成输入窗口。")
    report.append("3. 使用 cache 里的 `bar_mean/std` 和 `feature_mean/std` 做标准化，不能用线上全样本重新拟合。")
    report.append("4. 模型输出 `pred_ret_{h}` 后，策略层选择 horizon，并按阈值转成 `target_pos`。")
    report.append("5. 回测或实盘必须使用同一套固定成本假设；如果成本设为 0，报告和配置里必须标注。")
    report.append("")
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--horizons", default="5,15")
    parser.add_argument("--feature-windows", default="5,15,30")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold-bps", type=float, default=0.0)
    parser.add_argument("--fixed-fee-bps", type=float, default=0.0)
    parser.add_argument("--fixed-slippage-bps", type=float, default=0.0)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = ensure_dataset(args)
    train_config = TrainConfig(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        device=args.device,
        threshold_bps=args.threshold_bps,
        validation_samples=args.validation_samples,
    )
    (out_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), "train_config": vars(train_config)}, indent=2) + "\n"
    )

    print("dataset", dataset.n_symbols, dataset.n_times, dataset.bar_dim, dataset.feature_dim, flush=True)
    print("device", choose_device(args.device), flush=True)
    started = time.time()

    models = {}
    for name, use_features in [("ohlcv_tcn", False), ("full_feature_tcn", True)]:
        model, train_info = train_model(dataset, use_features, train_config, out_dir, name)
        models[name] = (model, use_features, train_info)

    results = {
        "training": {name: info["history"] for name, (_, _, info) in models.items()},
        "evaluation": {},
        "backtests": {},
        "stability": {},
        "cost_config": {
            "threshold_bps": args.threshold_bps,
            "fixed_fee_bps": args.fixed_fee_bps,
            "fixed_slippage_bps": args.fixed_slippage_bps,
        },
        "elapsed_sec": time.time() - started,
    }
    for name, (model, use_features, _) in models.items():
        eval_result = evaluate_model(dataset, model, use_features, "test", args.batch_size)
        pred = eval_result.pop("pred")
        true = eval_result.pop("true")
        pos_score = eval_result.pop("pos_score")
        results["evaluation"][name] = eval_result
        for h_idx, horizon in enumerate(dataset.horizons):
            metrics, positions, portfolio_returns = backtest_scores(
                dataset,
                pred,
                "test",
                h_idx,
                threshold_bps=args.threshold_bps,
                fixed_fee_bps=args.fixed_fee_bps,
                fixed_slippage_bps=args.fixed_slippage_bps,
            )
            metrics["horizon"] = f"{horizon}m"
            key = f"{name}_{horizon}m"
            results["backtests"][key] = metrics
            if name == "full_feature_tcn":
                results["stability"][key] = {
                    "by_symbol_top": by_symbol_summary(dataset, positions, "test"),
                    "by_month": by_month_summary(dataset, portfolio_returns, "test"),
                }
        np.savez_compressed(out_dir / f"{name}_test_predictions.npz", pred=pred, true=true, pos_score=pos_score)

    results["backtests"]["no_trade"] = {**no_trade_metrics(dataset, "test"), "horizon": "-"}
    results["backtests"]["buy_and_hold_equal_weight"] = {
        **backtest_buy_and_hold(
            dataset,
            "test",
            fixed_fee_bps=args.fixed_fee_bps,
            fixed_slippage_bps=args.fixed_slippage_bps,
        ),
        "horizon": "-",
    }
    for h_idx, horizon in enumerate(dataset.horizons):
        results["backtests"][f"naive_momentum_{horizon}m"] = {
            **backtest_rule_momentum(
                dataset,
                "test",
                h_idx,
                threshold_bps=args.threshold_bps,
                fixed_fee_bps=args.fixed_fee_bps,
                fixed_slippage_bps=args.fixed_slippage_bps,
            ),
            "horizon": f"{horizon}m",
        }

    results["elapsed_sec"] = time.time() - started
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(out_dir, dataset, results)
    print(f"wrote {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
