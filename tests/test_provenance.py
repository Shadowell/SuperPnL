from pathlib import Path

import pytest
import torch

from superpnl.data import DatasetConfig, prepare_dataset, load_prepared_dataset
from superpnl.training import TrainConfig, train_model
from test_data import make_raw_frame, write_raw_frame


@pytest.mark.parametrize("loaded", [False, True])
def test_checkpoint_records_the_exact_prepared_data_contract(tmp_path: Path, loaded: bool):
    raw = tmp_path / "raw"
    cache = tmp_path / "cache"
    write_raw_frame(raw, make_raw_frame())
    dataset = prepare_dataset(DatasetConfig(str(raw), str(cache), lookback=32, horizons=(5, 15), feature_windows=(5, 15, 30)))
    if loaded:
        dataset = load_prepared_dataset(cache)
    run = tmp_path / "run"
    train_model(dataset, False, TrainConfig(hidden_dim=4, epochs=0, device="cpu"), run, "ohlcv_tcn")
    checkpoint = torch.load(run / "ohlcv_tcn.pt", weights_only=True)

    assert "data_contract" in checkpoint, "Checkpoint must identify the data used to train it"
    from superpnl.provenance import cache_data_contract
    assert checkpoint["data_contract"] == cache_data_contract(cache)
    assert checkpoint["data_contract"]["horizons"] == [5, 15]


@pytest.mark.parametrize("corruption", ["horizon_order", "label_shape"])
def test_training_rejects_inconsistent_prepared_dataset(tmp_path, corruption):
    raw = tmp_path / "raw"
    write_raw_frame(raw, make_raw_frame())
    dataset = prepare_dataset(DatasetConfig(str(raw), str(tmp_path / "cache"), lookback=32, horizons=(5, 15), feature_windows=(5, 15, 30)))
    if corruption == "horizon_order":
        dataset.horizons = (15, 5)
    else:
        dataset.labels = dataset.labels[..., :1]
    with pytest.raises(ValueError, match="data contract.*rebuild"):
        train_model(dataset, False, TrainConfig(hidden_dim=4, epochs=0, device="cpu"), tmp_path / "run", "ohlcv_tcn")
    assert not (tmp_path / "run" / "ohlcv_tcn.pt").exists()
