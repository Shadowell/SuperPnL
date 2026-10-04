from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .data import PreparedDataset
from .metrics import rank_ic_by_time, regression_metrics
from .provenance import prepared_data_contract
from .training import TrainConfig


@dataclass
class LightGBMTrainConfig:
    objective: str = "rank_regression"
    metric: str = "ndcg"
    rank_bins: int = 5
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 5
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l1: float = 0.0
    lambda_l2: float = 1.0
    max_bin: int = 63
    num_boost_round: int = 600
    early_stopping_rounds: int = 50
    train_samples: int | None = None
    validation_samples: int | None = None
    num_threads: int | None = None
    seed: int = 17

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, allow_nan=False)


class LightGBMWrapper:
    def __init__(
        self,
        models: dict[int, lgb.Booster],
        horizons: tuple[int, ...],
        objective: str = "rank_regression",
        data_contract: dict | None = None,
        num_threads: int = 1,
    ):
        self.models = models
        self.horizons = horizons
        self.objective = objective
        self.data_contract = data_contract
        self.num_threads = num_threads

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.data_contract:
            raise ValueError("LightGBM model has no data contract; retrain the model")
        if (features.ndim != 2 or features.shape[1] != self.data_contract["feature_dim"]
                or not np.isfinite(features).all()):
            raise ValueError("LightGBM features must be finite and match the data contract")
        preds = []
        for i, _ in enumerate(self.horizons):
            preds.append(self.models[i].predict(
                features, num_iteration=self.models[i].best_iteration, num_threads=self.num_threads,
            ))
        result = np.stack(preds, axis=-1)
        if not np.isfinite(result).all():
            raise ValueError("LightGBM predictions must be finite")
        return result

    def save(self, path: Path, name: str = "lightgbm_trader") -> None:
        if not self.data_contract:
            raise ValueError("LightGBM model has no data contract; retrain the model")
        path.mkdir(parents=True, exist_ok=True)
        for i, horizon in enumerate(self.horizons):
            self.models[i].save_model(str(path / f"{name}_{horizon}m.txt"))
        metadata = {
            "model_type": "lightgbm", "horizons": list(self.horizons), "objective": self.objective,
            "data_contract": self.data_contract, "num_threads": self.num_threads,
            "output_kind": "rank_score" if self.objective in {"rank_regression", "lambdarank", "rank_xendcg"} else "log_return",
        }
        (path / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: Path, horizons: tuple[int, ...], name: str = "lightgbm_trader") -> LightGBMWrapper:
        metadata_path = path / f"{name}_metadata.json"
        if not metadata_path.exists():
            raise ValueError("LightGBM metadata/data contract missing; retrain the model")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("model_type") != "lightgbm" or not metadata.get("data_contract"):
            raise ValueError("LightGBM data contract missing or unsupported model type; retrain the model")
        data_contract = metadata["data_contract"]
        if list(horizons) != metadata.get("horizons") or list(horizons) != data_contract.get("horizons"):
            raise ValueError("LightGBM horizon order does not match the saved data contract")
        objective = metadata["objective"]
        num_threads = int(metadata.get("num_threads", 1))
        if num_threads <= 0:
            raise ValueError("LightGBM num_threads must be positive")
        models = {}
        for i, horizon in enumerate(horizons):
            models[i] = lgb.Booster(model_file=str(path / f"{name}_{horizon}m.txt"))
            if models[i].num_feature() != data_contract["feature_dim"]:
                raise ValueError("LightGBM model feature count does not match the data contract")
        return cls(models, horizons, objective=objective, data_contract=data_contract, num_threads=num_threads)


def _split_range(dataset: PreparedDataset, split: str) -> tuple[int, int]:
    return {
        "train": dataset.train_range,
        "val": dataset.val_range,
        "test": dataset.test_range,
    }[split]


def _sample_time_indices(n_times: int, max_samples: int | None, n_symbols: int, seed: int) -> np.ndarray:
    if max_samples is None:
        return np.arange(n_times, dtype="int64")
    max_times = max(1, int(max_samples) // max(1, n_symbols))
    if max_times >= n_times:
        return np.arange(n_times, dtype="int64")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_times, size=max_times, replace=False).astype("int64"))


def _cross_section_relevance(labels_time_symbol: np.ndarray, rank_bins: int) -> np.ndarray:
    n_times, n_symbols = labels_time_symbol.shape
    order = np.argsort(np.nan_to_num(labels_time_symbol, nan=0.0), axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n_times)[:, None], order] = np.arange(n_symbols)[None, :]
    bins = max(2, int(rank_bins))
    relevance = np.floor(ranks * bins / n_symbols).astype("int32")
    return np.clip(relevance, 0, bins - 1)


def _cross_section_rank_target(labels_time_symbol: np.ndarray) -> np.ndarray:
    n_times, n_symbols = labels_time_symbol.shape
    order = np.argsort(np.nan_to_num(labels_time_symbol, nan=0.0), axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n_times)[:, None], order] = np.arange(n_symbols)[None, :]
    if n_symbols <= 1:
        return np.zeros_like(labels_time_symbol, dtype="float32")
    return (ranks.astype("float32") / float(n_symbols - 1)) - 0.5


def _build_split_matrix(
    dataset: PreparedDataset,
    split: str,
    horizon_index: int,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start, end = _split_range(dataset, split)
    local_times = _sample_time_indices(end - start, max_samples, dataset.n_symbols, seed)
    absolute_times = start + local_times
    features = dataset.feature_inputs[:, absolute_times, :].astype("float32")
    labels = dataset.labels[:, absolute_times, horizon_index].astype("float32")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("LightGBM training features and labels must be finite")

    x_time_major = np.transpose(features, (1, 0, 2)).reshape(-1, dataset.feature_dim)
    y_time_major = labels.T.reshape(-1)
    group = np.full(len(local_times), dataset.n_symbols, dtype="int32")
    return x_time_major, y_time_major, group, local_times


def _lgb_params(config: LightGBMTrainConfig) -> dict:
    num_threads = config.num_threads
    if num_threads is None:
        num_threads = int(os.environ.get("LGB_NUM_THREADS", max(1, (os.cpu_count() or 4) - 1)))
    if num_threads <= 0:
        raise ValueError("LightGBM num_threads must be positive")
    objective = config.objective
    metric = config.metric
    if objective == "rank_regression":
        objective = "regression_l2"
        metric = "l2"
    elif objective in {"huber", "regression_l2"}:
        metric = "l2"
    params = {
        "objective": objective,
        "metric": metric,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "lambda_l1": config.lambda_l1,
        "lambda_l2": config.lambda_l2,
        "max_bin": config.max_bin,
        "verbosity": -1,
        "seed": config.seed,
        "num_threads": num_threads,
    }
    if objective in {"lambdarank", "rank_xendcg"}:
        params["label_gain"] = [float((1 << i) - 1) for i in range(max(2, config.rank_bins))]
    return params


def make_lightgbm_config(
    train_config: TrainConfig,
    *,
    objective: str = "rank_regression",
    train_samples: int | None = None,
    validation_samples: int | None = None,
    rank_bins: int = 5,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 50,
) -> LightGBMTrainConfig:
    return LightGBMTrainConfig(
        objective=objective,
        rank_bins=rank_bins,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        train_samples=train_samples if train_samples is not None else train_config.samples_per_epoch,
        validation_samples=validation_samples
        if validation_samples is not None
        else train_config.validation_samples,
        seed=train_config.seed,
    )


def train_lightgbm_model(
    dataset: PreparedDataset,
    config: TrainConfig,
    out_dir: str | Path,
    name: str,
    lgb_config: LightGBMTrainConfig | None = None,
) -> tuple[LightGBMWrapper, dict]:
    data_contract = prepared_data_contract(dataset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lgb_config = lgb_config or make_lightgbm_config(config)
    started = time.time()
    params = _lgb_params(lgb_config)

    models = {}
    lgb_details = []
    for i, horizon in enumerate(dataset.horizons):
        print(f"Training LightGBM {lgb_config.objective} for {horizon}m horizon...", flush=True)
        x_train, y_train_raw, train_group, _ = _build_split_matrix(
            dataset,
            "train",
            i,
            lgb_config.train_samples,
            lgb_config.seed + i,
        )
        x_val, y_val_raw, val_group, _ = _build_split_matrix(
            dataset,
            "val",
            i,
            lgb_config.validation_samples,
            lgb_config.seed + 1000 + i,
        )
        if lgb_config.objective in {"lambdarank", "rank_xendcg"}:
            y_train = _cross_section_relevance(
                y_train_raw.reshape(-1, dataset.n_symbols),
                lgb_config.rank_bins,
            ).reshape(-1)
            y_val = _cross_section_relevance(
                y_val_raw.reshape(-1, dataset.n_symbols),
                lgb_config.rank_bins,
            ).reshape(-1)
        elif lgb_config.objective == "rank_regression":
            y_train = _cross_section_rank_target(
                y_train_raw.reshape(-1, dataset.n_symbols),
            ).reshape(-1)
            y_val = _cross_section_rank_target(
                y_val_raw.reshape(-1, dataset.n_symbols),
            ).reshape(-1)
        else:
            y_train = y_train_raw
            y_val = y_val_raw

        train_data = lgb.Dataset(x_train, label=y_train, group=train_group, free_raw_data=True)
        val_data = lgb.Dataset(x_val, label=y_val, group=val_group, reference=train_data, free_raw_data=True)
        evals_result = {}
        callbacks = [lgb.record_evaluation(evals_result)]
        if lgb_config.early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=lgb_config.early_stopping_rounds, verbose=False))
        model = lgb.train(
            params,
            train_data,
            num_boost_round=lgb_config.num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
        models[i] = model
        metric_name = next(iter(model.best_score.get("val", {"metric": 0.0}).keys()))
        lgb_details.append(
            {
                "horizon": f"{horizon}m",
                "objective": lgb_config.objective,
                "metric": metric_name,
                "best_iteration": int(model.best_iteration or lgb_config.num_boost_round),
                "best_score": float(model.best_score["val"][metric_name]),
                "train_groups": int(len(train_group)),
                "val_groups": int(len(val_group)),
            }
        )

    wrapper = LightGBMWrapper(
        models, dataset.horizons, objective=lgb_config.objective,
        data_contract=data_contract, num_threads=params["num_threads"],
    )
    val_pred, val_true, _ = predict_lightgbm_split(dataset, wrapper, "val")
    horizon_metrics = {}
    for i, horizon in enumerate(dataset.horizons):
        reg = regression_metrics(val_pred[:, :, i], val_true[:, :, i])
        ic = rank_ic_by_time(val_pred[:, :, i], val_true[:, :, i])
        horizon_metrics[f"{horizon}m"] = {**reg, **ic}

    best_record = {
        "train_loss": 0.0,
        "val_mae_mean": float(np.mean([v["mae"] for v in horizon_metrics.values()])),
        "val_rank_ic_mean": float(np.mean([v["rank_ic"] for v in horizon_metrics.values()])),
        "elapsed_sec": time.time() - started,
    }
    history = [{"epoch": 1, **best_record}]

    wrapper.save(out, name=name)
    (out / f"{name}_config.json").write_text(lgb_config.to_json() + "\n")
    (out / f"{name}_details.json").write_text(json.dumps(lgb_details, indent=2, allow_nan=False) + "\n")
    (out / f"{name}_history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    return wrapper, {
        "history": history,
        "best_record": best_record,
        "lgb_details": lgb_details,
        "config": asdict(lgb_config),
    }


def predict_lightgbm_split(
    dataset: PreparedDataset,
    model: LightGBMWrapper,
    split: str,
    chunk_times: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model.data_contract != prepared_data_contract(dataset):
        raise ValueError("LightGBM dataset/model data contract mismatch; use the matching cache or retrain")
    if chunk_times <= 0:
        raise ValueError("chunk_times must be positive")
    start, end = _split_range(dataset, split)
    n_times = end - start
    pred = np.full((dataset.n_symbols, n_times, len(dataset.horizons)), np.nan, dtype="float32")
    for local_start in range(0, n_times, chunk_times):
        local_end = min(local_start + chunk_times, n_times)
        absolute = np.arange(start + local_start, start + local_end)
        features = dataset.feature_inputs[:, absolute, :].astype("float32")
        x_flat = np.transpose(features, (1, 0, 2)).reshape(-1, dataset.feature_dim)
        pred_time_major = model.predict(x_flat).reshape(local_end - local_start, dataset.n_symbols, -1)
        pred[:, local_start:local_end, :] = np.transpose(pred_time_major, (1, 0, 2)).astype("float32")
    true = dataset.realized_horizon_returns[:, start:end, :].astype("float32")
    pos_score = (1.0 / (1.0 + np.exp(-np.clip(pred, -50, 50)))).astype("float32")
    return pred, true, pos_score


def evaluate_lightgbm_model(
    dataset: PreparedDataset,
    model: LightGBMWrapper,
    split: str,
) -> dict:
    pred, true, pos_score = predict_lightgbm_split(dataset, model, split)
    horizon_metrics = {}
    for i, horizon in enumerate(dataset.horizons):
        reg = regression_metrics(pred[:, :, i], true[:, :, i])
        ic = rank_ic_by_time(pred[:, :, i], true[:, :, i])
        horizon_metrics[f"{horizon}m"] = {**reg, **ic}
    return {"horizon_metrics": horizon_metrics, "pred": pred, "true": true, "pos_score": pos_score}
