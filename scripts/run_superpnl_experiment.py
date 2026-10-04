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
from superpnl.provenance import validate_cache, prepared_data_contract
from superpnl.portfolio import simulate_weight_portfolio
from superpnl.tabular_training import evaluate_lightgbm_model, make_lightgbm_config, train_lightgbm_model
from superpnl.training import (
    LowTurnoverConfig,
    TrainConfig,
    backtest_buy_and_hold,
    backtest_low_turnover_scores,
    backtest_rule_momentum,
    backtest_scores,
    choose_device,
    evaluate_model,
    no_trade_metrics,
    train_model,
)


def ensure_dataset(args) -> object:
    cache_dir = Path(args.cache_dir)
    config = DatasetConfig(
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        lookback=args.lookback,
        horizons=tuple(int(x) for x in args.horizons.split(",")),
        feature_windows=tuple(int(x) for x in args.feature_windows.split(",")),
        factor_set=getattr(args, "factor_set", "base"),
    )
    if (cache_dir / "metadata.json").exists() and not args.rebuild_cache:
        metadata = json.loads((cache_dir / "metadata.json").read_text())
        validate_cache(json.loads(config.to_json()), metadata)
        return load_prepared_dataset(cache_dir, mmap=True)
    return prepare_dataset(config)


def format_metric(value: float | int | None) -> str:
    if value is None:
        return "N/A"
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


def by_symbol_summary(
    dataset, positions: np.ndarray, split: str,
    fixed_fee_bps: float = 0.0, fixed_slippage_bps: float = 0.0,
    rebalance_interval_bars: int = 1,
) -> list[dict]:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    next_returns = dataset.next_returns[:, start:end].astype("float64")
    # Both backtest variants return actual portfolio weights.
    ledger = simulate_weight_portfolio(
        next_returns, positions,
        cost_rate=(fixed_fee_bps + fixed_slippage_bps) / 10_000.0,
        rebalance_mask=np.arange(end - start) % max(1, rebalance_interval_bars) == 0,
    )
    out = []
    for i, symbol in enumerate(dataset.symbols):
        out.append(
            {
                "symbol": symbol,
                "pnl_contribution": float(ledger.pnl_by_symbol[i].sum()),
                "avg_weight": float(ledger.positions[i].mean()) if positions.shape[1] else 0.0,
            }
        )
    return sorted(out, key=lambda item: item["pnl_contribution"], reverse=True)


def by_month_summary(dataset, portfolio_returns: np.ndarray, split: str) -> list[dict]:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    ts = pd.to_datetime(dataset.timestamps[start + 2:end + 2], unit="ms", utc=True)
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
    report.append(f"- factor_set: `{results['data_config']['factor_set']}`")
    report.append(f"- cost: `fee={results['cost_config']['fixed_fee_bps']}bps, slippage={results['cost_config']['fixed_slippage_bps']}bps`")
    report.append(f"- threshold: `{results['cost_config']['threshold_bps']}bps`")
    if results.get("low_turnover_config"):
        report.append(f"- low_turnover_config: `{json.dumps(results['low_turnover_config'], ensure_ascii=False)}`")
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
                format_metric(metrics.get("gross_total_return", metrics["total_return"])),
                format_metric(metrics["total_return"]),
                format_metric(metrics["annualized_return"]),
                format_metric(metrics["sharpe"]),
                format_metric(metrics["max_drawdown"]),
                format_metric(metrics["turnover"]),
                format_metric(metrics["average_position"]),
                format_metric(metrics["trade_count"]),
            ]
        )
    report.append("| model | horizon | gross_return | net_return | annualized | sharpe | max_drawdown | turnover | avg_pos | trades |")
    report.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
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
            report.append(f"- `{row['symbol']}` pnl_contribution={row['pnl_contribution']:.4f}, avg_weight={row['avg_weight']:.4f}")
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
    parser.add_argument("--horizons", default="5,15,30,60,240,1440")
    parser.add_argument("--feature-windows", default="5,15,30,60,240,1440")
    parser.add_argument("--factor-set", choices=["base", "expanded"], default="base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--position-loss-weight", type=float, default=0.15)
    parser.add_argument("--model-selection-metric", default="val_rank_ic_mean")
    parser.add_argument("--validation-samples", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold-bps", type=float, default=0.0)
    parser.add_argument("--fixed-fee-bps", type=float, default=0.0)
    parser.add_argument("--fixed-slippage-bps", type=float, default=0.0)
    parser.add_argument("--low-turnover-backtest", action="store_true")
    parser.add_argument("--low-turnover-top-k", type=int, default=3)
    parser.add_argument("--rebalance-interval-bars", type=int, default=15)
    parser.add_argument("--min-holding-bars", type=int, default=30)
    parser.add_argument("--cooldown-bars", type=int, default=30)
    parser.add_argument("--max-position-per-symbol", type=float, default=0.20)
    parser.add_argument("--max-total-position", type=float, default=0.60)
    parser.add_argument("--max-turnover-per-step", type=float, default=1.0)
    parser.add_argument("--min-liquidity-rank", type=float, default=None)
    parser.add_argument("--max-illiquidity-rank", type=float, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-tcn", action="store_true", help="Skip PyTorch TCN training and only run LightGBM")
    parser.add_argument(
        "--lightgbm-objective",
        choices=["rank_regression", "lambdarank", "rank_xendcg", "huber", "regression_l2"],
        default="rank_regression",
    )
    parser.add_argument("--lightgbm-train-samples", type=int, default=None)
    parser.add_argument("--lightgbm-validation-samples", type=int, default=None)
    parser.add_argument("--lightgbm-rank-bins", type=int, default=5)
    parser.add_argument("--lightgbm-num-boost-round", type=int, default=600)
    parser.add_argument("--lightgbm-early-stopping-rounds", type=int, default=50)
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        position_loss_weight=args.position_loss_weight,
        device=args.device,
        threshold_bps=args.threshold_bps,
        validation_samples=args.validation_samples,
        model_selection_metric=args.model_selection_metric,
    )
    (out_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), "train_config": vars(train_config)}, indent=2) + "\n"
    )

    print("dataset", dataset.n_symbols, dataset.n_times, dataset.bar_dim, dataset.feature_dim, flush=True)
    print("device", choose_device(args.device), flush=True)
    started = time.time()

    models = {}
    if not args.skip_tcn:
        for name, use_features in [("ohlcv_tcn", False), ("full_feature_tcn", True)]:
            model, train_info = train_model(dataset, use_features, train_config, out_dir, name)
            models[name] = (model, use_features, train_info, "pytorch")

    lgb_config = make_lightgbm_config(
        train_config,
        objective=args.lightgbm_objective,
        train_samples=args.lightgbm_train_samples,
        validation_samples=args.lightgbm_validation_samples,
        rank_bins=args.lightgbm_rank_bins,
        num_boost_round=args.lightgbm_num_boost_round,
        early_stopping_rounds=args.lightgbm_early_stopping_rounds,
    )
    lgb_model, lgb_info = train_lightgbm_model(dataset, train_config, out_dir, "lightgbm_trader", lgb_config)
    models["lightgbm_trader"] = (lgb_model, True, lgb_info, "lightgbm")

    results = {
        "training": {name: info["history"] for name, (_, _, info, _) in models.items()},
        "evaluation": {},
        "backtests": {},
        "stability": {},
        "data_config": {
            "factor_set": args.factor_set,
            "feature_windows": args.feature_windows,
        },
        "cost_config": {
            "threshold_bps": args.threshold_bps,
            "fixed_fee_bps": args.fixed_fee_bps,
            "fixed_slippage_bps": args.fixed_slippage_bps,
        },
        "low_turnover_config": None,
        "elapsed_sec": time.time() - started,
    }
    low_turnover_config = LowTurnoverConfig(
        top_k=args.low_turnover_top_k,
        rebalance_interval_bars=args.rebalance_interval_bars,
        min_holding_bars=args.min_holding_bars,
        cooldown_bars=args.cooldown_bars,
        max_position_per_symbol=args.max_position_per_symbol,
        max_total_position=args.max_total_position,
        max_turnover_per_step=args.max_turnover_per_step,
        min_liquidity_rank=args.min_liquidity_rank,
        max_illiquidity_rank=args.max_illiquidity_rank,
    )
    if args.low_turnover_backtest:
        results["low_turnover_config"] = vars(low_turnover_config)
    for name, (model, use_features, _, model_type) in models.items():
        if model_type == "pytorch":
            eval_result = evaluate_model(dataset, model, use_features, "test", args.batch_size)
        else:
            eval_result = evaluate_lightgbm_model(dataset, model, "test")
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
                    "by_symbol_top": by_symbol_summary(
                        dataset, positions, "test", args.fixed_fee_bps, args.fixed_slippage_bps
                    ),
                    "by_month": by_month_summary(dataset, portfolio_returns, "test"),
                }
            if args.low_turnover_backtest:
                lt_metrics, lt_positions, lt_portfolio_returns = backtest_low_turnover_scores(
                    dataset,
                    pred,
                    "test",
                    h_idx,
                    threshold_bps=args.threshold_bps,
                    fixed_fee_bps=args.fixed_fee_bps,
                    fixed_slippage_bps=args.fixed_slippage_bps,
                    config=low_turnover_config,
                )
                lt_metrics["horizon"] = f"{horizon}m"
                lt_key = f"{name}_{horizon}m_low_turnover"
                results["backtests"][lt_key] = lt_metrics
                if name == "full_feature_tcn":
                    results["stability"][lt_key] = {
                        "by_symbol_top": by_symbol_summary(
                            dataset, lt_positions, "test", args.fixed_fee_bps,
                            args.fixed_slippage_bps, args.rebalance_interval_bars,
                        ),
                        "by_month": by_month_summary(dataset, lt_portfolio_returns, "test"),
                    }
        np.savez_compressed(
            out_dir / f"{name}_test_predictions.npz", pred=pred, true=true, pos_score=pos_score,
            data_contract=np.array(json.dumps(prepared_data_contract(dataset), sort_keys=True)),
        )

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
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    write_report(out_dir, dataset, results)
    print(f"wrote {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
