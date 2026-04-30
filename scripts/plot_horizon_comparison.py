#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def pct(value: float) -> float:
    return value * 100.0


def collect_metrics(h5_15_path: Path, h30_path: Path) -> dict:
    h5_15 = load_json(h5_15_path)
    h30 = load_json(h30_path)
    rows = {}
    for horizon in ("5m", "15m"):
        rows[horizon] = {
            "ohlcv": h5_15["backtests"][f"ohlcv_tcn_{horizon}"],
            "full": h5_15["backtests"][f"full_feature_tcn_{horizon}"],
            "naive": h5_15["backtests"][f"naive_momentum_{horizon}"],
            "buy_hold": h5_15["backtests"]["buy_and_hold_equal_weight"],
            "ohlcv_eval": h5_15["evaluation"]["ohlcv_tcn"]["horizon_metrics"][horizon],
            "full_eval": h5_15["evaluation"]["full_feature_tcn"]["horizon_metrics"][horizon],
        }
    rows["30m"] = {
        "ohlcv": h30["backtests"]["ohlcv_tcn_30m"],
        "full": h30["backtests"]["full_feature_tcn_30m"],
        "naive": h30["backtests"]["naive_momentum_30m"],
        "buy_hold": h30["backtests"]["buy_and_hold_equal_weight"],
        "ohlcv_eval": h30["evaluation"]["ohlcv_tcn"]["horizon_metrics"]["30m"],
        "full_eval": h30["evaluation"]["full_feature_tcn"]["horizon_metrics"]["30m"],
    }
    return rows


def annotate_bars(ax, bars, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = float(bar.get_height())
        va = "bottom" if height >= 0 else "top"
        offset = 2 if height >= 0 else -3
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )


def plot(rows: dict, out_path: Path) -> None:
    horizons = ["5m", "15m", "30m"]
    x = np.arange(len(horizons))
    width = 0.22

    ohlcv_return = [pct(rows[h]["ohlcv"]["total_return"]) for h in horizons]
    full_return = [pct(rows[h]["full"]["total_return"]) for h in horizons]
    naive_return = [pct(rows[h]["naive"]["total_return"]) for h in horizons]
    buy_hold_return = [pct(rows[h]["buy_hold"]["total_return"]) for h in horizons]

    ohlcv_sharpe = [rows[h]["ohlcv"]["sharpe"] for h in horizons]
    full_sharpe = [rows[h]["full"]["sharpe"] for h in horizons]
    naive_sharpe = [rows[h]["naive"]["sharpe"] for h in horizons]
    buy_hold_sharpe = [rows[h]["buy_hold"]["sharpe"] for h in horizons]

    ohlcv_dd = [pct(rows[h]["ohlcv"]["max_drawdown"]) for h in horizons]
    full_dd = [pct(rows[h]["full"]["max_drawdown"]) for h in horizons]
    naive_dd = [pct(rows[h]["naive"]["max_drawdown"]) for h in horizons]
    buy_hold_dd = [pct(rows[h]["buy_hold"]["max_drawdown"]) for h in horizons]

    ohlcv_ic = [rows[h]["ohlcv_eval"]["rank_ic"] for h in horizons]
    full_ic = [rows[h]["full_eval"]["rank_ic"] for h in horizons]

    colors = {
        "ohlcv": "#64748b",
        "full": "#2563eb",
        "naive": "#f97316",
        "buy_hold": "#16a34a",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=160)
    fig.suptitle("SuperPnL Top20 12M Test: Horizon Comparison", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    bars1 = ax.bar(x - width, ohlcv_return, width, label="OHLCV TCN", color=colors["ohlcv"])
    bars2 = ax.bar(x, full_return, width, label="Full Feature TCN", color=colors["full"])
    bars3 = ax.bar(x + width, naive_return, width, label="Naive Momentum", color=colors["naive"])
    ax.plot(x, buy_hold_return, color=colors["buy_hold"], marker="o", linestyle="--", label="Buy & Hold")
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    annotate_bars(ax, bars3)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Total Return (%)")
    ax.set_xticks(x, horizons)
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)

    ax = axes[0, 1]
    bars1 = ax.bar(x - width, ohlcv_sharpe, width, label="OHLCV TCN", color=colors["ohlcv"])
    bars2 = ax.bar(x, full_sharpe, width, label="Full Feature TCN", color=colors["full"])
    bars3 = ax.bar(x + width, naive_sharpe, width, label="Naive Momentum", color=colors["naive"])
    ax.plot(x, buy_hold_sharpe, color=colors["buy_hold"], marker="o", linestyle="--", label="Buy & Hold")
    annotate_bars(ax, bars1, "{:.1f}")
    annotate_bars(ax, bars2, "{:.1f}")
    annotate_bars(ax, bars3, "{:.1f}")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Sharpe")
    ax.set_xticks(x, horizons)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    bars1 = ax.bar(x - width, ohlcv_dd, width, label="OHLCV TCN", color=colors["ohlcv"])
    bars2 = ax.bar(x, full_dd, width, label="Full Feature TCN", color=colors["full"])
    bars3 = ax.bar(x + width, naive_dd, width, label="Naive Momentum", color=colors["naive"])
    ax.plot(x, buy_hold_dd, color=colors["buy_hold"], marker="o", linestyle="--", label="Buy & Hold")
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    annotate_bars(ax, bars3)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Max Drawdown (%)")
    ax.set_xticks(x, horizons)
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    bars1 = ax.bar(x - width / 2, ohlcv_ic, width, label="OHLCV TCN", color=colors["ohlcv"])
    bars2 = ax.bar(x + width / 2, full_ic, width, label="Full Feature TCN", color=colors["full"])
    annotate_bars(ax, bars1, "{:.3f}")
    annotate_bars(ax, bars2, "{:.3f}")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Rank IC")
    ax.set_xticks(x, horizons)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.text(
        0.01,
        0.01,
        "Cost assumption: fee=0bps, slippage=0bps. 15m is the only positive full-feature horizon; costs still break it due to high turnover.",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5-15-metrics",
        default="outputs/superpnl_top20_365d_l256_h5_15_hd64_e3/metrics.json",
    )
    parser.add_argument(
        "--h30-metrics",
        default="outputs/superpnl_top20_365d_l256_h30_hd64_e3/metrics.json",
    )
    parser.add_argument(
        "--out",
        default="docs/charts/superpnl_horizon_comparison.png",
    )
    args = parser.parse_args()
    rows = collect_metrics(Path(args.h5_15_metrics), Path(args.h30_metrics))
    plot(rows, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
