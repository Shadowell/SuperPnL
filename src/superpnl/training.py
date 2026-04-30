from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import PreparedDataset, WindowBatcher
from .metrics import compute_pnl_metrics, rank_ic_by_time, regression_metrics
from .model import SuperPnLModel


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
    seed: int = 17
    device: str = "auto"

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
        print(f"{name} epoch {epoch}: {record}", flush=True)
    model_path = out / f"{name}.pt"
    torch.save({"model": model.state_dict(), "config": asdict(config), "use_features": use_features}, model_path)
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
    cost = (fixed_fee_bps + fixed_slippage_bps) / 10_000.0
    scores = pred[:, :, horizon_index]
    positions = (scores > threshold).astype("float64")
    next_returns = dataset.next_returns[:, start:end].astype("float64")
    pnl_by_symbol = positions * next_returns
    turnover = np.abs(np.diff(positions, axis=1, prepend=0.0))
    pnl_by_symbol = pnl_by_symbol - turnover * cost
    portfolio_returns = np.nanmean(pnl_by_symbol, axis=0)
    metrics = compute_pnl_metrics(portfolio_returns, positions).as_dict()
    metrics.update(
        {
            "threshold_bps": threshold_bps,
            "fixed_fee_bps": fixed_fee_bps,
            "fixed_slippage_bps": fixed_slippage_bps,
        }
    )
    return metrics, positions, portfolio_returns


def backtest_rule_momentum(
    dataset: PreparedDataset,
    split: str,
    horizon_index: int,
    threshold_bps: float = 0.0,
    fixed_fee_bps: float = 0.0,
    fixed_slippage_bps: float = 0.0,
) -> dict:
    # Use ret_30m feature if present; otherwise fallback to first feature.
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    try:
        feature_idx = dataset.feature_names.index("ret_30m")
    except ValueError:
        feature_idx = 0
    score = dataset.feature_inputs[:, start:end, feature_idx]
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
    positions = np.ones((dataset.n_symbols, end - start), dtype="float64")
    next_returns = dataset.next_returns[:, start:end].astype("float64")
    cost = (fixed_fee_bps + fixed_slippage_bps) / 10_000.0
    turnover = np.abs(np.diff(positions, axis=1, prepend=0.0))
    portfolio_returns = np.nanmean(positions * next_returns - turnover * cost, axis=0)
    metrics = compute_pnl_metrics(portfolio_returns, positions).as_dict()
    metrics.update({"fixed_fee_bps": fixed_fee_bps, "fixed_slippage_bps": fixed_slippage_bps})
    return metrics


def no_trade_metrics(dataset: PreparedDataset, split: str) -> dict:
    start, end = {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]
    positions = np.zeros((dataset.n_symbols, end - start), dtype="float64")
    returns = np.zeros(end - start, dtype="float64")
    return compute_pnl_metrics(returns, positions).as_dict()
