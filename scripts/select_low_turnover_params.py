#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from superpnl.data import load_prepared_dataset
from superpnl.training import (
    LowTurnoverConfig,
    backtest_buy_and_hold,
    backtest_low_turnover_scores,
    no_trade_metrics,
)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_optional_float(value: str) -> float | None:
    value = value.strip().lower()
    if value in {"none", "null", "nan", ""}:
        return None
    return float(value)


def parse_filters(values: list[str]) -> list[dict]:
    filters = []
    for raw in values:
        value = raw.strip()
        if value.lower() == "none":
            filters.append(
                {
                    "filter_name": "none",
                    "min_liquidity_rank": None,
                    "max_illiquidity_rank": None,
                }
            )
            continue
        if ":" not in value:
            raise ValueError(f"liquidity filter must be 'none' or 'min_liquidity:max_illiquidity', got {raw!r}")
        min_liq, max_illiq = value.split(":", 1)
        filters.append(
            {
                "filter_name": value,
                "min_liquidity_rank": parse_optional_float(min_liq),
                "max_illiquidity_rank": parse_optional_float(max_illiq),
            }
        )
    return filters


def safe_number(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def compact_metrics(metrics: dict) -> dict:
    keys = [
        "net_total_return",
        "gross_total_return",
        "cost_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "turnover",
        "average_position",
        "max_total_position_observed",
        "average_holding_minutes",
        "trade_count",
        "win_rate",
        "profit_factor",
        "threshold_bps",
        "top_k",
        "rebalance_interval_bars",
        "min_holding_bars",
        "cooldown_bars",
        "max_position_per_symbol",
        "max_total_position",
        "max_turnover_per_step",
        "min_liquidity_rank",
        "max_illiquidity_rank",
        "fixed_fee_bps",
        "fixed_slippage_bps",
    ]
    return {key: safe_number(metrics[key]) for key in keys if key in metrics}


def split_range(dataset, split: str) -> tuple[int, int]:
    return {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]


def by_symbol_summary(
    dataset,
    positions: np.ndarray,
    split: str,
    fixed_fee_bps: float,
    fixed_slippage_bps: float,
) -> list[dict]:
    start, end = split_range(dataset, split)
    next_returns = dataset.next_returns[:, start:end].astype("float64")
    cost = (fixed_fee_bps + fixed_slippage_bps) / 10_000.0
    turnover = np.abs(np.diff(positions, axis=1, prepend=0.0))
    rows = []
    for i, symbol in enumerate(dataset.symbols):
        gross_log_sum = float(np.nansum(positions[i] * next_returns[i]))
        net_log_sum = float(np.nansum(positions[i] * next_returns[i] - turnover[i] * cost))
        rows.append(
            {
                "symbol": symbol,
                "gross_total_return": float(np.exp(gross_log_sum) - 1.0),
                "net_total_return": float(np.exp(net_log_sum) - 1.0),
                "avg_position": float(np.nanmean(positions[i])),
                "trade_count": int((turnover[i] > 1e-6).sum()),
            }
        )
    return sorted(rows, key=lambda row: row["net_total_return"], reverse=True)


def by_month_summary(dataset, portfolio_returns: np.ndarray, split: str) -> list[dict]:
    start, end = split_range(dataset, split)
    timestamps = pd.to_datetime(dataset.timestamps[start:end], unit="ms", utc=True)
    frame = pd.DataFrame({"month": timestamps.strftime("%Y-%m"), "ret": portfolio_returns})
    rows = []
    for month, group in frame.groupby("month"):
        rows.append({"month": month, "net_total_return": float(np.exp(group["ret"].sum()) - 1.0)})
    return rows


def selected_key(row: dict, min_trades: int, min_average_position: float) -> tuple:
    metrics = row["metrics"]
    eligible = int(
        metrics.get("trade_count", 0) >= min_trades
        and metrics.get("average_position", 0.0) >= min_average_position
    )
    return (
        eligible,
        float(metrics.get("net_total_return", -999.0)),
        float(metrics.get("sharpe", -999.0)),
        float(metrics.get("max_drawdown", -999.0)),
        -float(metrics.get("turnover", 999.0)),
    )


def model_thresholds(model_name: str, args) -> list[float]:
    if args.threshold_bps:
        return parse_floats(args.threshold_bps)
    if "ohlcv" in model_name:
        return parse_floats(args.ohlcv_threshold_bps)
    return parse_floats(args.full_feature_threshold_bps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select low-turnover trading parameters on validation split, then apply once to test split."
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", default="ohlcv_tcn,full_feature_tcn")
    parser.add_argument("--output-name", default="validation_selected_low_turnover.json")
    parser.add_argument("--horizon-index", type=int, default=0)
    parser.add_argument("--fixed-fee-bps", type=float, default=8.0)
    parser.add_argument("--fixed-slippage-bps", type=float, default=0.0)
    parser.add_argument("--threshold-bps", default="")
    parser.add_argument("--ohlcv-threshold-bps", default="0,2,5,10,15,20")
    parser.add_argument("--full-feature-threshold-bps", default="-10,-5,-2,0,2,5,10")
    parser.add_argument("--top-k", default="1,2,3")
    parser.add_argument("--rebalance-interval-bars", default="30,60")
    parser.add_argument("--min-holding-bars", default="30,60,120")
    parser.add_argument("--cooldown-bars", default="30")
    parser.add_argument("--max-position-per-symbol", type=float, default=0.20)
    parser.add_argument("--max-total-position", type=float, default=0.60)
    parser.add_argument("--max-turnover-per-step", type=float, default=1.0)
    parser.add_argument(
        "--liquidity-filter",
        action="append",
        default=None,
        help="Repeatable. Use 'none' or 'min_liquidity_rank:max_illiquidity_rank'.",
    )
    parser.add_argument("--min-select-trades", type=int, default=50)
    parser.add_argument("--min-select-average-position", type=float, default=0.002)
    parser.add_argument("--top-rows", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    report_path = out_dir / args.output_name

    dataset = load_prepared_dataset(cache_dir, mmap=True)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    filters = parse_filters(args.liquidity_filter or ["none", "-0.2:0.2"])
    top_k_values = parse_ints(args.top_k)
    rebalance_values = parse_ints(args.rebalance_interval_bars)
    holding_values = parse_ints(args.min_holding_bars)
    cooldown_values = parse_ints(args.cooldown_bars)

    results = {
        "selection_note": "Parameters are selected on validation split only, then applied once to test split.",
        "selection_rule": (
            "Prefer candidates with enough validation trades and average position, then sort by "
            "validation net_total_return, sharpe, max_drawdown, and lower turnover."
        ),
        "dataset": {
            "cache_dir": str(cache_dir),
            "n_symbols": dataset.n_symbols,
            "n_times": dataset.n_times,
            "feature_dim": dataset.feature_dim,
            "horizons": list(dataset.horizons),
            "val_range": list(dataset.val_range),
            "test_range": list(dataset.test_range),
        },
        "cost_assumption": {
            "fixed_fee_bps": args.fixed_fee_bps,
            "fixed_slippage_bps": args.fixed_slippage_bps,
        },
        "selection_constraints": {
            "min_select_trades": args.min_select_trades,
            "min_select_average_position": args.min_select_average_position,
        },
        "baselines": {
            "val": {
                "no_trade": compact_metrics(no_trade_metrics(dataset, "val")),
                "buy_and_hold": compact_metrics(
                    backtest_buy_and_hold(dataset, "val", args.fixed_fee_bps, args.fixed_slippage_bps)
                ),
            },
            "test": {
                "no_trade": compact_metrics(no_trade_metrics(dataset, "test")),
                "buy_and_hold": compact_metrics(
                    backtest_buy_and_hold(dataset, "test", args.fixed_fee_bps, args.fixed_slippage_bps)
                ),
            },
        },
        "models": {},
    }

    started_all = time.time()
    for model_name in models:
        val_path = out_dir / f"{model_name}_val_predictions.npz"
        test_path = out_dir / f"{model_name}_test_predictions.npz"
        if not val_path.exists():
            raise FileNotFoundError(f"missing validation predictions: {val_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"missing test predictions: {test_path}")

        val_pred = np.load(val_path)["pred"]
        test_pred = np.load(test_path)["pred"]
        thresholds = model_thresholds(model_name, args)
        total = (
            len(thresholds)
            * len(top_k_values)
            * len(rebalance_values)
            * len(holding_values)
            * len(cooldown_values)
            * len(filters)
        )
        rows = []
        started = time.time()
        done = 0
        for threshold_bps in thresholds:
            for top_k in top_k_values:
                for rebalance in rebalance_values:
                    for holding in holding_values:
                        for cooldown in cooldown_values:
                            for filter_cfg in filters:
                                config = LowTurnoverConfig(
                                    top_k=top_k,
                                    rebalance_interval_bars=rebalance,
                                    min_holding_bars=holding,
                                    cooldown_bars=cooldown,
                                    max_position_per_symbol=args.max_position_per_symbol,
                                    max_total_position=args.max_total_position,
                                    max_turnover_per_step=args.max_turnover_per_step,
                                    min_liquidity_rank=filter_cfg["min_liquidity_rank"],
                                    max_illiquidity_rank=filter_cfg["max_illiquidity_rank"],
                                )
                                metrics, _, _ = backtest_low_turnover_scores(
                                    dataset,
                                    val_pred,
                                    split="val",
                                    horizon_index=args.horizon_index,
                                    threshold_bps=threshold_bps,
                                    fixed_fee_bps=args.fixed_fee_bps,
                                    fixed_slippage_bps=args.fixed_slippage_bps,
                                    config=config,
                                )
                                rows.append(
                                    {
                                        "model": model_name,
                                        "filter_name": filter_cfg["filter_name"],
                                        "config": {**asdict(config), "threshold_bps": threshold_bps},
                                        "metrics": compact_metrics(metrics),
                                    }
                                )
                                done += 1
                                if done % 50 == 0 or done == total:
                                    best = max(
                                        rows,
                                        key=lambda row: selected_key(
                                            row, args.min_select_trades, args.min_select_average_position
                                        ),
                                    )
                                    best_metrics = best["metrics"]
                                    print(
                                        f"{model_name}: {done}/{total} "
                                        f"best_val_net={best_metrics.get('net_total_return', 0):.4f} "
                                        f"sharpe={best_metrics.get('sharpe', 0):.2f} "
                                        f"trades={best_metrics.get('trade_count', 0)}",
                                        flush=True,
                                    )

        rows_sorted = sorted(
            rows,
            key=lambda row: selected_key(row, args.min_select_trades, args.min_select_average_position),
            reverse=True,
        )
        selected = rows_sorted[0]
        cfg_dict = selected["config"]
        selected_config = LowTurnoverConfig(
            top_k=cfg_dict["top_k"],
            rebalance_interval_bars=cfg_dict["rebalance_interval_bars"],
            min_holding_bars=cfg_dict["min_holding_bars"],
            cooldown_bars=cfg_dict["cooldown_bars"],
            max_position_per_symbol=cfg_dict["max_position_per_symbol"],
            max_total_position=cfg_dict["max_total_position"],
            max_turnover_per_step=cfg_dict["max_turnover_per_step"],
            min_liquidity_rank=cfg_dict["min_liquidity_rank"],
            max_illiquidity_rank=cfg_dict["max_illiquidity_rank"],
        )
        val_metrics, val_positions, val_returns = backtest_low_turnover_scores(
            dataset,
            val_pred,
            split="val",
            horizon_index=args.horizon_index,
            threshold_bps=cfg_dict["threshold_bps"],
            fixed_fee_bps=args.fixed_fee_bps,
            fixed_slippage_bps=args.fixed_slippage_bps,
            config=selected_config,
        )
        test_metrics, test_positions, test_returns = backtest_low_turnover_scores(
            dataset,
            test_pred,
            split="test",
            horizon_index=args.horizon_index,
            threshold_bps=cfg_dict["threshold_bps"],
            fixed_fee_bps=args.fixed_fee_bps,
            fixed_slippage_bps=args.fixed_slippage_bps,
            config=selected_config,
        )
        results["models"][model_name] = {
            "selected_is_eligible": bool(
                selected_key(selected, args.min_select_trades, args.min_select_average_position)[0]
            ),
            "selected_filter_name": selected["filter_name"],
            "selected_config": cfg_dict,
            "validation_metrics": compact_metrics(val_metrics),
            "test_metrics": compact_metrics(test_metrics),
            "attribution": {
                "val": {
                    "by_symbol": by_symbol_summary(
                        dataset, val_positions, "val", args.fixed_fee_bps, args.fixed_slippage_bps
                    ),
                    "by_month": by_month_summary(dataset, val_returns, "val"),
                },
                "test": {
                    "by_symbol": by_symbol_summary(
                        dataset, test_positions, "test", args.fixed_fee_bps, args.fixed_slippage_bps
                    ),
                    "by_month": by_month_summary(dataset, test_returns, "test"),
                },
            },
            "top_validation_rows": [
                {"filter_name": row["filter_name"], "config": row["config"], "metrics": row["metrics"]}
                for row in rows_sorted[: args.top_rows]
            ],
            "searched_combinations": len(rows),
            "elapsed_sec": time.time() - started,
        }
        print(
            f"selected {model_name}: val_net={selected['metrics'].get('net_total_return', 0):.4f} "
            f"test_net={results['models'][model_name]['test_metrics'].get('net_total_return', 0):.4f} "
            f"config={cfg_dict}",
            flush=True,
        )

    results["elapsed_sec"] = time.time() - started_all
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
