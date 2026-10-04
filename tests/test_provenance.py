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
    dataset = prepare_dataset(DatasetConfig(str(raw), str(cache), lookback=32))
    if loaded:
        dataset = load_prepared_dataset(cache)
    run = tmp_path / "run"
    train_model(dataset, False, TrainConfig(hidden_dim=4, epochs=0, device="cpu"), run, "ohlcv_tcn")
    checkpoint = torch.load(run / "ohlcv_tcn.pt", weights_only=True)

    assert "data_contract" in checkpoint, "Checkpoint must identify the data used to train it"
    from superpnl.provenance import cache_data_contract
    assert checkpoint["data_contract"] == cache_data_contract(cache)
    assert checkpoint["data_contract"]["horizons"] == [5, 15]
