from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import PreparedDataset, WindowBatcher
from .metrics import compute_pnl_metrics, rank_ic_by_time, regression_metrics
from .model import SuperPnLModel
from .portfolio import PortfolioLedger, simulate_portfolio, simulate_weight_portfolio
from .provenance import prepared_data_contract


@dataclass
class TrainConfig:
    hidden_dim: int = 128
    dropout: float = 0.05
    batch_size: int = 256
    epochs: int = 5
    samples_per_epoch: int = 200_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    position_loss_weight: float = 0.15
    threshold_bps: float = 0.0
    validation_samples: int | None = 100_000
    model_selection_metric: str = "val_rank_ic_mean"
    seed: int = 17
    device: str = "auto"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class LowTurnoverConfig:
    top_k: int = 3
    rebalance_interval_bars: int = 15
    min_holding_bars: int = 30
    cooldown_bars: int = 30
    max_position_per_symbol: float = 0.20
    max_total_position: float = 0.60
    max_turnover_per_step: float = 1.0
    min_liquidity_rank: float | None = None
    max_illiquidity_rank: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def choose_device(configured: str = "auto") -> torch.device:
    if configured != "auto":
        return torch.device(configured)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    dataset: PreparedDataset,
    use_features: bool,
    config: TrainConfig,
    out_dir: str | Path,
    name: str,
) -> tuple[SuperPnLModel, dict]:
    data_contract = prepared_data_contract(dataset)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    model = SuperPnLModel(
        bar_dim=dataset.bar_dim,
        feature_dim=dataset.feature_dim if use_features else 0,
        num_horizons=len(dataset.horizons),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        use_features=use_features,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    huber = nn.HuberLoss(delta=0.001)
    bce = nn.BCEWithLogitsLoss()
    threshold = config.threshold_bps / 10_000.0
    history = []
    best_score = -float("inf")
    best_state = None
    best_record = None
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        batcher = WindowBatcher(
            dataset,
            split="train",
            use_features=use_features,
            batch_size=config.batch_size,
            samples_per_epoch=config.samples_per_epoch,
            seed=config.seed + epoch,
        )
        losses = []
        for bar_np, feat_np, label_np, _, _ in batcher.iter_batches(shuffle=True):
            bar = torch.from_numpy(bar_np).to(device)
            feat = torch.from_numpy(feat_np).to(device) if use_features else None
            labels = torch.from_numpy(label_np).to(device)
            pos_label = (labels > threshold).float()
            pred_ret, pos_logit = model(bar, feat)
            loss_ret = huber(pred_ret, labels)
            loss_pos = bce(pos_logit, pos_label)
            loss = loss_ret + config.position_loss_weight * loss_pos
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate_model(
            dataset,
            model,
            use_features,
            "val",
            config.batch_size,
            device,
            max_samples=config.validation_samples,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_mae_mean": float(np.mean([v["mae"] for v in val["horizon_metrics"].values()])),
            "val_rank_ic_mean": float(np.mean([v["rank_ic"] for v in val["horizon_metrics"].values()])),
            "elapsed_sec": time.time() - started,
        }
        history.append(record)
        score = float(record.get(config.model_selection_metric, record["val_rank_ic_mean"]))
        if score > best_score:
            best_score = score
            best_record = record
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        print(f"{name} epoch {epoch}: {record}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = out / f"{name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "use_features": use_features,
            "best_record": best_record,
            "best_score": best_score,
            "data_contract": data_contract,
        },
        model_path,
    )
    (out / f"{name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, {"history": history, "model_path": str(model_path)}


@torch.no_grad()
def predict_split(
    dataset: PreparedDataset,
    model: SuperPnLModel,
    use_features: bool,
    split: str,
    batch_size: int,
    device: torch.device,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    pred = np.full((dataset.n_symbols, end - start, len(dataset.horizons)), np.nan, dtype="float32")
    pos_score = np.full_like(pred, np.nan)
    true = dataset.realized_horizon_returns[:, start:end, :].astype("float32")
    batcher = WindowBatcher(dataset, split, use_features, batch_size, samples_per_epoch=max_samples, seed=123)
    for bar_np, feat_np, _, sym_idx, time_idx in batcher.iter_batches(shuffle=False):
        bar = torch.from_numpy(bar_np).to(device)
        feat = torch.from_numpy(feat_np).to(device) if use_features else None
        pred_ret, pos_logit = model(bar, feat)
        p = pred_ret.detach().cpu().numpy().astype("float32")
        s = torch.sigmoid(pos_logit).detach().cpu().numpy().astype("float32")
        local_t = time_idx - start
        pred[sym_idx, local_t, :] = p
        pos_score[sym_idx, local_t, :] = s
    return pred, true, pos_score


def evaluate_model(
    dataset: PreparedDataset,
    model: SuperPnLModel,
    use_features: bool,
    split: str,
    batch_size: int,
    device: torch.device | None = None,
    max_samples: int | None = None,
) -> dict:
    device = device or next(model.parameters()).device
    pred, true, pos_score = predict_split(dataset, model, use_features, split, batch_size, device, max_samples)
    horizon_metrics = {}
    for i, horizon in enumerate(dataset.horizons):
        reg = regression_metrics(pred[:, :, i], true[:, :, i])
        ic = rank_ic_by_time(pred[:, :, i], true[:, :, i])
        horizon_metrics[f"{horizon}m"] = {**reg, **ic}
    return {"horizon_metrics": horizon_metrics, "pred": pred, "true": true, "pos_score": pos_score}


def _gross_net_summary(ledger: PortfolioLedger, gross_ledger: PortfolioLedger) -> dict[str, float]:
    return {
        "gross_total_return": float(gross_ledger.equity[-1] - 1.0) if len(gross_ledger.equity) else 0.0,
        "net_total_return": float(ledger.equity[-1] - 1.0) if len(ledger.equity) else 0.0,
        # Actual paid cost as a fraction of initial capital, without double-counting compounding.
        "cost_return": float(ledger.costs.sum()),
    }


def _raw_feature(dataset: PreparedDataset, feature_name: str, start: int, end: int) -> np.ndarray | None:
    if feature_name not in dataset.feature_names:
        return None
    idx = dataset.feature_names.index(feature_name)
    values = dataset.feature_inputs[:, start:end, idx].astype("float64")
    if dataset.feature_mean is not None and dataset.feature_std is not None:
        values = values * float(dataset.feature_std[idx]) + float(dataset.feature_mean[idx])
    return values


def _raw_feature_first(dataset: PreparedDataset, prefixes: tuple[str, ...], start: int, end: int) -> np.ndarray | None:
    for prefix in prefixes:
        for name in dataset.feature_names:
            if name.startswith(prefix):
                return _raw_feature(dataset, name, start, end)
    return None


def backtest_scores(
    dataset: PreparedDataset,
    pred: np.ndarray,
    split: str,
    horizon_index: int,
    threshold_bps: float = 0.0,
    fixed_fee_bps: float = 0.0,
    fixed_slippage_bps: float = 0.0,
) -> tuple[dict, np.ndarray, np.ndarray]:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    threshold = threshold_bps / 10_000.0
    if not np.isfinite(threshold):
        raise ValueError("backtest threshold_bps must be finite")
    cost = (fixed_fee_bps + fixed_slippage_bps) / 10_000.0
    scores = pred[:, :, horizon_index]
    if not np.isfinite(scores).all():
        raise ValueError("backtest scores must all be finite; predict the full split before backtesting")
    signals = (scores > threshold).astype("float64")
    next_returns = dataset.next_returns[:, start:end].astype("float64")
    ledger = simulate_portfolio(next_returns, signals, cost_rate=cost)
    gross_ledger = simulate_portfolio(next_returns, signals) if cost else ledger
    metrics = _ledger_metrics(ledger)
    metrics.update(
        {
            **_gross_net_summary(ledger, gross_ledger),
            "threshold_bps": threshold_bps,
            "fixed_fee_bps": fixed_fee_bps,
            "fixed_slippage_bps": fixed_slippage_bps,
        }
    )
    return metrics, ledger.positions, ledger.portfolio_returns


def _ledger_metrics(ledger: PortfolioLedger) -> dict:
    metrics = compute_pnl_metrics(ledger.portfolio_returns, ledger.positions).as_dict()
    if len(ledger.equity):
        previous_equity = np.r_[1.0, ledger.equity[:-1]]
        traded_fraction = ledger.traded_notional / previous_equity[None, :]
        metrics["turnover"] = float(traded_fraction.sum(axis=0).mean())
        metrics["trade_count"] = int((traded_fraction > 1e-12).sum())
        metrics["average_position"] = float(ledger.positions.sum(axis=0).mean())
    return metrics


def backtest_low_turnover_scores(
    dataset: PreparedDataset,
    pred: np.ndarray,
    split: str,
    horizon_index: int,
    threshold_bps: float = 0.0,
    fixed_fee_bps: float = 0.0,
    fixed_slippage_bps: float = 0.0,
    config: LowTurnoverConfig | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    cfg = config or LowTurnoverConfig()
    threshold = threshold_bps / 10_000.0
    if not np.isfinite(threshold):
        raise ValueError("backtest threshold_bps must be finite")
    cost = (fixed_fee_bps + fixed_slippage_bps) / 10_000.0
    scores = pred[:, :, horizon_index].astype("float64")
    if not np.isfinite(scores).all():
        raise ValueError("backtest scores must all be finite; predict the full split before backtesting")
    n_symbols, n_times = scores.shape
    positions = np.zeros((n_symbols, n_times), dtype="float64")
    current = np.zeros(n_symbols, dtype="float64")
    entry_time = np.full(n_symbols, -10**9, dtype="int64")
    cooldown_until = np.zeros(n_symbols, dtype="int64")
    rebalance_interval = max(1, int(cfg.rebalance_interval_bars))
    min_holding = max(0, int(cfg.min_holding_bars))
    cooldown = max(0, int(cfg.cooldown_bars))
    top_k = max(0, int(cfg.top_k))

    liquidity_rank = _raw_feature_first(dataset, ("cross_section_amount_rank_30m", "cross_section_amount_rank_"), start, end)
    illiquidity_rank = _raw_feature_first(
        dataset, ("cross_section_amihud_rank_30m", "cross_section_amihud_rank_"), start, end
    )

    for t in range(n_times):
        if t % rebalance_interval != 0:
            positions[:, t] = current
            continue

        eligible = np.isfinite(scores[:, t]) & (scores[:, t] > threshold)
        if cfg.min_liquidity_rank is not None and liquidity_rank is not None:
            eligible &= liquidity_rank[:, t] >= cfg.min_liquidity_rank
        if cfg.max_illiquidity_rank is not None and illiquidity_rank is not None:
            eligible &= illiquidity_rank[:, t] <= cfg.max_illiquidity_rank

        ranked = np.argsort(scores[:, t])[::-1]
        selected = [int(i) for i in ranked if eligible[i]][:top_k]
        target = np.zeros(n_symbols, dtype="float64")
        if selected:
            slot = min(cfg.max_position_per_symbol, cfg.max_total_position / len(selected))
            target[selected] = slot

        for i in range(n_symbols):
            if current[i] > target[i] and t - entry_time[i] < min_holding:
                target[i] = current[i]
            if current[i] <= 1e-12 and target[i] > 0 and t < cooldown_until[i]:
                target[i] = 0.0

        total_target = target.sum()
        if total_target > cfg.max_total_position and total_target > 0:
            protected = (current > 0) & (target >= current) & ((t - entry_time) < min_holding)
            flexible = ~protected
            protected_total = target[protected].sum()
            flexible_total = target[flexible].sum()
            room = max(cfg.max_total_position - protected_total, 0.0)
            if flexible_total > 0:
                target[flexible] *= room / flexible_total

        if cfg.max_turnover_per_step < 1.0:
            delta = target - current
            step_cap = max(float(cfg.max_turnover_per_step), 0.0)
            largest_change = float(np.max(np.abs(delta))) if len(delta) else 0.0
            scale = min(1.0, step_cap / largest_change) if largest_change > 0.0 else 1.0
            # A convex combination preserves both portfolio limits while
            # respecting the maximum change allowed for every symbol.
            target = current + scale * delta

        sold = (current > 1e-12) & (target <= 1e-12)
        cooldown_until[sold] = t + cooldown
        bought = (current <= 1e-12) & (target > 1e-12)
        entry_time[bought] = t
        current = target
        positions[:, t] = current

    next_returns = dataset.next_returns[:, start:end].astype("float64")
    rebalance_mask = np.arange(n_times) % rebalance_interval == 0
    ledger = simulate_weight_portfolio(next_returns, positions, cost, rebalance_mask)
    gross_ledger = simulate_weight_portfolio(next_returns, positions, 0.0, rebalance_mask) if cost else ledger
    metrics = _ledger_metrics(ledger)
    total_position = ledger.positions.sum(axis=0)
    metrics.update(
        {
            **_gross_net_summary(ledger, gross_ledger),
            "max_total_position_observed": float(np.nanmax(total_position)) if len(total_position) else 0.0,
            "threshold_bps": threshold_bps,
            "fixed_fee_bps": fixed_fee_bps,
            "fixed_slippage_bps": fixed_slippage_bps,
            "top_k": cfg.top_k,
            "rebalance_interval_bars": cfg.rebalance_interval_bars,
            "min_holding_bars": cfg.min_holding_bars,
            "cooldown_bars": cfg.cooldown_bars,
            "max_position_per_symbol": cfg.max_position_per_symbol,
            "max_total_position": cfg.max_total_position,
            "max_turnover_per_step": cfg.max_turnover_per_step,
            "min_liquidity_rank": cfg.min_liquidity_rank,
            "max_illiquidity_rank": cfg.max_illiquidity_rank,
        }
    )
    return metrics, ledger.positions, ledger.portfolio_returns


def backtest_rule_momentum(
    dataset: PreparedDataset,
    split: str,
    horizon_index: int,
    threshold_bps: float = 0.0,
    fixed_fee_bps: float = 0.0,
    fixed_slippage_bps: float = 0.0,
) -> dict:
    # Keep the same raw-return rule for every prediction horizon.
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    try:
        feature_idx = dataset.feature_names.index("ret_30m")
    except ValueError:
        candidates = [
            (int(match.group(1)), idx)
            for idx, feature in enumerate(dataset.feature_names)
            if (match := re.fullmatch(r"ret_([1-9][0-9]*)m", feature)) is not None
        ]
        if not candidates:
            raise ValueError("momentum baseline requires an available ret_{window}m return feature")
        _, feature_idx = max(candidates)
    if dataset.feature_mean is None or dataset.feature_std is None:
        raise ValueError("momentum baseline requires training feature normalization statistics")
    means = np.asarray(dataset.feature_mean)
    stds = np.asarray(dataset.feature_std)
    expected_shape = (len(dataset.feature_names),)
    if means.shape != expected_shape or stds.shape != expected_shape:
        raise ValueError("feature normalization statistics do not match the feature schema")
    mean = float(means[feature_idx])
    std = float(stds[feature_idx])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise ValueError("momentum feature normalization statistics must be finite with positive std")
    score = dataset.feature_inputs[:, start:end, feature_idx].astype("float64") * std + mean
    return backtest_scores(
        dataset,
        pred=score[:, :, None],
        split=split,
        horizon_index=0,
        threshold_bps=threshold_bps,
        fixed_fee_bps=fixed_fee_bps,
        fixed_slippage_bps=fixed_slippage_bps,
    )[0]


def backtest_buy_and_hold(
    dataset: PreparedDataset,
    split: str,
    fixed_fee_bps: float = 0.0,
    fixed_slippage_bps: float = 0.0,
) -> dict:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    signals = np.ones((dataset.n_symbols, end - start), dtype="float64")
    next_returns = dataset.next_returns[:, start:end].astype("float64")
    cost = (fixed_fee_bps + fixed_slippage_bps) / 10_000.0
    ledger = simulate_portfolio(next_returns, signals, cost_rate=cost, rebalance=False)
    gross_ledger = simulate_portfolio(next_returns, signals, rebalance=False) if cost else ledger
    metrics = _ledger_metrics(ledger)
    metrics.update({**_gross_net_summary(ledger, gross_ledger),
                    "fixed_fee_bps": fixed_fee_bps, "fixed_slippage_bps": fixed_slippage_bps})
    return metrics


def no_trade_metrics(dataset: PreparedDataset, split: str) -> dict:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    positions = np.zeros((dataset.n_symbols, end - start), dtype="float64")
    returns = np.zeros(end - start, dtype="float64")
    metrics = compute_pnl_metrics(returns, positions).as_dict()
    metrics.update({"gross_total_return": 0.0, "net_total_return": 0.0, "cost_return": 0.0})
    return metrics
