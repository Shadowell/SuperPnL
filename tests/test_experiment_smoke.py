import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from superpnl.data import load_prepared_dataset
from superpnl.model import SuperPnLModel
from superpnl.provenance import cache_data_contract
from test_data import make_raw_frame, write_raw_frame


ROOT = Path(__file__).resolve().parents[1]


def test_small_cpu_experiment_packages_both_models(tmp_path):
    raw, cache, run = (tmp_path / name for name in ("raw", "cache", "run"))
    for index, symbol in enumerate(("BTC-USDT", "ETH-USDT", "SOL-USDT")):
        frame = make_raw_frame(400)
        frame[["open", "high", "low", "close", "amount"]] *= index + 1
        write_raw_frame(raw, frame, symbol)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    command = [sys.executable, "-B", "scripts/run_superpnl_experiment.py",
               "--raw-dir", str(raw), "--cache-dir", str(cache), "--out-dir", str(run),
               "--lookback", "32", "--epochs", "1", "--samples-per-epoch", "8",
               "--batch-size", "4", "--hidden-dim", "4", "--validation-samples", "12",
               "--device", "cpu", "--fixed-fee-bps", "10", "--fixed-slippage-bps", "2"]
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    metrics = json.loads((run / "metrics.json").read_text(),
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert {"no_trade", "buy_and_hold_equal_weight", "ohlcv_tcn_15m", "full_feature_tcn_15m"} <= metrics["backtests"].keys()
    assert metrics["backtests"]["no_trade"]["total_return"] == 0
    for name, stability in metrics["stability"].items():
        contribution = sum(row["pnl_contribution"] for row in stability["by_symbol_top"])
        np.testing.assert_allclose(contribution, metrics["backtests"][name]["total_return"], atol=1e-12)
    dataset = load_prepared_dataset(cache)
    contract = cache_data_contract(cache)
    for model_name in ("ohlcv_tcn", "full_feature_tcn"):
        package = tmp_path / ("package-" + model_name)
        packed = subprocess.run(
            [sys.executable, "-B", "scripts/package_superpnl_model.py",
             "--run-dir", str(run), "--cache-dir", str(cache), "--model-name", model_name,
             "--package-dir", str(package)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        assert packed.returncode == 0, packed.stdout + packed.stderr
        config = json.loads((package / "model_config.json").read_text())
        keys = ("bar_dim", "feature_dim", "num_horizons", "hidden_dim", "dropout", "use_features")
        model = SuperPnLModel(**{key: config[key] for key in keys}).eval()
        checkpoint = torch.load(package / "model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        assert checkpoint["data_contract"] == contract
        t = dataset.test_range[0]
        window = slice(t - dataset.lookback + 1, t + 1)
        bar = torch.from_numpy(np.array(dataset.bar_inputs[:1, window, :], copy=True))
        features = torch.from_numpy(np.array(dataset.feature_inputs[:1, window, :], copy=True))
        with torch.no_grad():
            prediction, score = model(bar, features if config["use_features"] else None)
        assert prediction.shape == score.shape == (1, 2)
        assert torch.isfinite(prediction).all()
        assert (package / "manifest.json").exists()
        assert package.with_name(package.name + ".tar.gz").is_file()
