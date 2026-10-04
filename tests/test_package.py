from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import package_superpnl_model as packaging
from superpnl.model import SuperPnLModel


def make_package_inputs(tmp_path: Path, use_features: bool = True):
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    run_dir.mkdir()
    cache_dir.mkdir()
    train_config = {"hidden_dim": 4, "dropout": 0.2}
    (run_dir / "run_config.json").write_text(json.dumps({"train_config": train_config}))
    (run_dir / "metrics.json").write_text(json.dumps({"backtests": {}, "evaluation": {}}))
    metadata = {
        "horizons": [5, 15],
        "bar_dim": 6,
        "feature_dim": 2,
        "lookback": 16,
        "feature_names": ["ret_5m", "ret_15m"],
        "config": {"feature_windows": [5, 15]},
        "symbols": ["BTC-USDT"],
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata))
    for name, dim in [("bar", 6), ("feature", 2)]:
        np.save(cache_dir / f"{name}_mean.npy", np.zeros(dim, dtype="float32"))
        np.save(cache_dir / f"{name}_std.npy", np.ones(dim, dtype="float32"))
    model = SuperPnLModel(6, 2 if use_features else 0, 2, **train_config, use_features=use_features)
    model.eval()
    name = "full_feature_tcn" if use_features else "ohlcv_tcn"
    torch.save(
        {"model": model.state_dict(), "config": train_config, "use_features": use_features},
        run_dir / f"{name}.pt",
    )
    return run_dir, cache_dir, name, model


def run_package(monkeypatch, run_dir, cache_dir, model_name, package_dir, *extra):
    monkeypatch.setattr(sys, "argv", [
        "package_superpnl_model.py", "--run-dir", str(run_dir),
        "--cache-dir", str(cache_dir), "--model-name", model_name,
        "--package-dir", str(package_dir), *extra,
    ])
    packaging.main()


@pytest.mark.parametrize("use_features", [False, True])
@pytest.mark.parametrize("stale_run_config", [False, True])
def test_exported_architecture_loads_checkpoint_and_preserves_predictions(
    tmp_path, monkeypatch, use_features, stale_run_config,
):
    run_dir, cache_dir, name, original = make_package_inputs(tmp_path, use_features)
    if stale_run_config:
        (run_dir / "run_config.json").write_text(json.dumps({
            "train_config": {"hidden_dim": 8, "dropout": 0.7},
        }))
    package_dir = tmp_path / "package"
    run_package(monkeypatch, run_dir, cache_dir, name, package_dir)
    config = json.loads((package_dir / "model_config.json").read_text())
    checkpoint = torch.load(package_dir / "model.pt", map_location="cpu", weights_only=True)
    loaded = SuperPnLModel(**{key: config[key] for key in (
        "bar_dim", "feature_dim", "num_horizons", "hidden_dim", "dropout", "use_features",
    )})
    loaded.load_state_dict(checkpoint["model"])
    loaded.eval()
    bar = torch.randn(2, 16, 6)
    features = torch.randn(2, 16, 2) if use_features else None
    with torch.no_grad():
        actual = loaded(bar, features)
        expected = original(bar, features)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    assert config["use_features"] is use_features
    assert config["dropout"] == 0.2
    if not use_features:
        assert config["feature_dim"] == 0
        assert "features" not in config["input_shapes"]
