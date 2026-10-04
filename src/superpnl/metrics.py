from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


MINUTES_PER_YEAR = 365 * 24 * 60


@dataclass
class PnLMetrics:
    total_return: float
    annualized_return: float | None
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float | None
    win_rate: float
    profit_factor: float
    turnover: float
    average_position: float
    average_holding_minutes: float
    trade_count: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "turnover": self.turnover,
            "average_position": self.average_position,
            "average_holding_minutes": self.average_holding_minutes,
            "trade_count": self.trade_count,
        }


def compute_pnl_metrics(portfolio_returns: np.ndarray, positions: np.ndarray) -> PnLMetrics:
    returns = np.nan_to_num(portfolio_returns.astype("float64"), nan=0.0)
    equity = np.exp(np.cumsum(returns))
    total_return = float(equity[-1] - 1.0) if len(equity) else 0.0
    if len(returns) > 1:
        mean = returns.mean()
        std = returns.std(ddof=1)
        sharpe = float(mean / std * math.sqrt(MINUTES_PER_YEAR)) if std > 1e-12 else 0.0
        negative_returns = returns[returns < 0]
        downside = negative_returns.std(ddof=1) if len(negative_returns) > 1 else 0.0
        sortino = float(mean / downside * math.sqrt(MINUTES_PER_YEAR)) if downside > 1e-12 else 0.0
        try:
            annualized_return = math.expm1(float(mean) * MINUTES_PER_YEAR)
        except OverflowError:
            annualized_return = None
        if annualized_return is not None and not math.isfinite(annualized_return):
            annualized_return = None
    else:
        sharpe = 0.0
        sortino = 0.0
        annualized_return = total_return
    equity_with_initial = np.concatenate(([1.0], equity))
    peak = np.maximum.accumulate(equity_with_initial)
    drawdown = equity_with_initial / peak - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    if annualized_return is None:
        calmar = None
    else:
        calmar = float(annualized_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-12 else 0.0
        if not math.isfinite(calmar):
            calmar = None
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 0.0
    pos = np.nan_to_num(positions.astype("float64"), nan=0.0)
    turnover = float(np.abs(np.diff(pos, axis=1, prepend=0.0)).mean()) if pos.ndim == 2 and pos.size else 0.0
    average_position = float(pos.mean()) if pos.size else 0.0
    trade_count = int((np.abs(np.diff(pos, axis=1, prepend=0.0)) > 1e-6).sum()) if pos.ndim == 2 else 0
    holding = _average_holding_minutes(pos)
    return PnLMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        turnover=turnover,
        average_position=average_position,
        average_holding_minutes=holding,
        trade_count=trade_count,
    )


def _average_holding_minutes(positions: np.ndarray) -> float:
    durations = []
    for row in positions:
        active = row > 1e-6
        start = None
        for i, value in enumerate(active):
            if value and start is None:
                start = i
            elif not value and start is not None:
                durations.append(i - start)
                start = None
        if start is not None:
            durations.append(len(active) - start)
    return float(np.mean(durations)) if durations else 0.0


def rank_ic_by_time(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    # pred/true: [S, T]
    ics = []
    rank_ics = []
    for t in range(pred.shape[1]):
        p = pred[:, t]
        y = true[:, t]
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        # Use the same precision for the variance gate and corrcoef. A
        # constant float32 vector can have nonzero std from mean rounding.
        p = p[mask].astype("float64")
        y = y[mask].astype("float64")
        if np.std(p) > 1e-12 and np.std(y) > 1e-12:
            correlation = float(np.corrcoef(p, y)[0, 1])
            if np.isfinite(correlation):
                ics.append(correlation)
        rank = pd.Series(p).rank().to_numpy()
        yrank = pd.Series(y).rank().to_numpy()
        if np.std(rank) > 1e-12 and np.std(yrank) > 1e-12:
            correlation = float(np.corrcoef(rank, yrank)[0, 1])
            if np.isfinite(correlation):
                rank_ics.append(correlation)
    def summarize(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        arr = np.asarray(values)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
        ir = float(mean / std) if std > 1e-12 else 0.0
        return mean, ir
    ic, icir = summarize(ics)
    ric, ricir = summarize(rank_ics)
    return {"ic": ic, "icir": icir, "rank_ic": ric, "rank_icir": ricir}


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return {"mae": 0.0, "rmse": 0.0, "direction_hit_rate": 0.0}
    diff = pred[mask] - true[mask]
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "direction_hit_rate": float((np.sign(pred[mask]) == np.sign(true[mask])).mean()),
    }
