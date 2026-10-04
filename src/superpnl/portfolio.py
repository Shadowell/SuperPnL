from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PortfolioLedger:
    portfolio_returns: np.ndarray
    positions: np.ndarray
    traded_notional: np.ndarray
    costs: np.ndarray
    pnl_by_symbol: np.ndarray
    equity: np.ndarray
    cash: np.ndarray


def _post_cost_equity(equity: float, holdings: np.ndarray, weights: np.ndarray, cost_rate: float) -> float:
    """Solve E_after + cost * sum(abs(weights * E_after - holdings)) = E."""
    if cost_rate == 0.0:
        return equity
    estimate = equity
    # The piecewise-linear root moves downward. Each crossing can turn one
    # asset from a buy into a sell, so at most S + 1 segments need evaluation.
    for _ in range(len(weights) + 1):
        direction = np.where(weights * estimate >= holdings, 1.0, -1.0)
        updated = (equity + cost_rate * np.dot(direction, holdings)) / (
            1.0 + cost_rate * np.dot(direction, weights)
        )
        if abs(updated - estimate) <= 1e-13 * equity:
            return float(updated)
        estimate = updated
    return float(estimate)


def simulate_portfolio(
    log_returns: np.ndarray,
    signals: np.ndarray,
    cost_rate: float = 0.0,
    rebalance: bool = True,
) -> PortfolioLedger:
    """Simulate long-only signals/N; keep the original equal-sleeve API."""
    signals = np.asarray(signals, dtype="float64")
    if signals.ndim != 2 or signals.shape[0] == 0:
        raise ValueError("signals must have a [symbols, time] shape")
    if not np.isfinite(signals).all() or np.any((signals < 0.0) | (signals > 1.0)):
        raise ValueError("signals must be finite and in [0, 1]")
    mask = np.ones(signals.shape[1], dtype=bool)
    if not rebalance:
        mask[1:] = False
    return simulate_weight_portfolio(log_returns, signals / signals.shape[0], cost_rate, mask)


def simulate_weight_portfolio(
    log_returns: np.ndarray,
    target_weights: np.ndarray,
    cost_rate: float = 0.0,
    rebalance_mask: np.ndarray | None = None,
) -> PortfolioLedger:
    """Simulate actual portfolio target weights using self-financed trades.

    Inputs have shape [symbols, time]. Trades occur before each supplied
    asset return. Positions are actual post-trade portfolio weights; costs,
    traded notional and PnL contributions are amounts per initial capital 1.
    Only rebalance_mask=True bars trade; holdings drift between those bars.
    Final holdings are marked to market without a liquidation fee.
    """
    returns = np.asarray(log_returns, dtype="float64")
    target_weights = np.asarray(target_weights, dtype="float64")
    if returns.ndim != 2 or target_weights.shape != returns.shape or returns.shape[0] == 0:
        raise ValueError("log_returns and target_weights must have matching [symbols, time] shapes")
    if not np.isfinite(returns).all():
        raise ValueError("log_returns must be finite")
    if (not np.isfinite(target_weights).all() or np.any(target_weights < 0.0)
            or np.any(target_weights.sum(axis=0) > 1.0 + 1e-12)):
        raise ValueError("target_weights must be finite, nonnegative and sum to at most 1")
    mask = np.ones(returns.shape[1], dtype=bool) if rebalance_mask is None else np.asarray(rebalance_mask)
    if mask.shape != (returns.shape[1],) or mask.dtype != np.dtype(bool):
        raise ValueError("rebalance_mask must be a boolean vector with one value per time step")
    if not np.isfinite(cost_rate) or not 0.0 <= cost_rate < 1.0:
        raise ValueError("cost_rate must be finite and in [0, 1)")
    simple_returns = np.expm1(returns)
    if not np.isfinite(simple_returns).all() or np.any(simple_returns <= -1.0):
        raise ValueError("asset gross returns must be positive and finite")

    n_symbols, n_times = returns.shape
    positions = np.zeros_like(returns)
    traded_notional = np.zeros_like(returns)
    costs = np.zeros_like(returns)
    pnl_by_symbol = np.zeros_like(returns)
    equity_path = np.empty(n_times, dtype="float64")
    cash_path = np.empty(n_times, dtype="float64")
    portfolio_returns = np.empty(n_times, dtype="float64")
    holdings = np.zeros(n_symbols, dtype="float64")
    cash = 1.0
    equity = 1.0
    for t in range(n_times):
        if mask[t]:
            weights = target_weights[:, t]
            post_cost_equity = _post_cost_equity(equity, holdings, weights, cost_rate)
            target_holdings = weights * post_cost_equity
            traded_notional[:, t] = np.abs(target_holdings - holdings)
            costs[:, t] = traded_notional[:, t] * cost_rate
            cash = equity - costs[:, t].sum() - target_holdings.sum()
            if cash < -1e-12 * equity:
                raise ArithmeticError("portfolio rebalance would require borrowing")
            cash = max(0.0, float(cash))
            holdings = target_holdings
        post_cost_equity = cash + holdings.sum()
        positions[:, t] = holdings / post_cost_equity
        asset_pnl = holdings * simple_returns[:, t]
        pnl_by_symbol[:, t] = asset_pnl - costs[:, t]
        holdings = holdings + asset_pnl
        next_equity = float(cash + holdings.sum())
        if not np.isfinite(next_equity) or next_equity <= 0.0:
            raise ArithmeticError("portfolio equity must remain positive and finite")
        portfolio_returns[t] = np.log(next_equity / equity)
        equity = next_equity
        equity_path[t] = equity
        cash_path[t] = cash
    return PortfolioLedger(
        portfolio_returns=portfolio_returns,
        positions=positions,
        traded_notional=traded_notional,
        costs=costs,
        pnl_by_symbol=pnl_by_symbol,
        equity=equity_path,
        cash=cash_path,
    )
