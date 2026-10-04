from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_superpnl_experiment import ensure_dataset
from superpnl.data import DatasetConfig, load_prepared_dataset, prepare_dataset
from superpnl.provenance import cache_data_contract
from test_data import make_raw_frame, write_raw_frame


@pytest.fixture
def cached_args(tmp_path):
    raw = tmp_path / "raw"
    write_raw_frame(raw, make_raw_frame())
    cache = tmp_path / "cache"
    prepare_dataset(DatasetConfig(str(raw), str(cache), lookback=32, horizons=(5, 15), feature_windows=(5, 15, 30)))
    return Namespace(raw_dir=str(raw), cache_dir=str(cache), lookback=32,
                     horizons="5,15", feature_windows="5,15,30", rebuild_cache=False)


def test_matching_cache_is_reused(cached_args):
    result = ensure_dataset(cached_args)
    assert result.lookback == 32
    assert result.horizons == (5, 15)


@pytest.mark.parametrize(("name", "value"), [
    ("lookback", 99), ("horizons", "30"), ("horizons", "15,5"),
    ("feature_windows", "7,21"),
])
def test_changed_configuration_rejects_cache(cached_args, name, value):
    setattr(cached_args, name, value)
    with pytest.raises(ValueError, match="rebuild-cache"):
        ensure_dataset(cached_args)


def test_changed_source_bytes_reject_cache(cached_args):
    frame = make_raw_frame()
    frame.loc[100, "volume"] += 1
    write_raw_frame(Path(cached_args.raw_dir), frame)
    with pytest.raises(ValueError, match="rebuild-cache"):
        ensure_dataset(cached_args)


def test_changed_source_path_rejects_cache(cached_args, tmp_path):
    raw = tmp_path / "other_raw"
    write_raw_frame(raw, make_raw_frame())
    cached_args.raw_dir = str(raw)
    with pytest.raises(ValueError, match="rebuild-cache"):
        ensure_dataset(cached_args)


def test_missing_source_rejects_cache(cached_args, tmp_path):
    cached_args.raw_dir = str(tmp_path / "missing")
    with pytest.raises((ValueError, FileNotFoundError)):
        ensure_dataset(cached_args)


def test_legacy_cache_rejected_on_direct_load(cached_args):
    path = Path(cached_args.cache_dir) / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.pop("cache_version", None)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="rebuild-cache"):
        load_prepared_dataset(cached_args.cache_dir)


@pytest.mark.parametrize("entrypoint", ["ensure", "load", "contract"])
@pytest.mark.parametrize("inconsistency", [
    "horizon_order", "lookback", "feature_count", "symbol_count", "empty_split",
    "negative_start", "overlap", "unpurged_labels", "past_data_end",
])
def test_inconsistent_cache_metadata_is_rejected(cached_args, entrypoint, inconsistency):
    path = Path(cached_args.cache_dir) / "metadata.json"
    metadata = json.loads(path.read_text())
    if inconsistency == "horizon_order":
        metadata["horizons"] = [15, 5]
    elif inconsistency == "lookback":
        metadata["lookback"] = 64
    elif inconsistency == "feature_count":
        metadata["feature_names"] = metadata["feature_names"][:-1]
    elif inconsistency == "symbol_count":
        metadata["n_symbols"] = 2
    elif inconsistency == "empty_split":
        metadata["train_range"][1] = metadata["train_range"][0]
    elif inconsistency == "negative_start":
        metadata["train_range"][0] = -1
    elif inconsistency == "overlap":
        metadata["val_range"][0] = metadata["train_range"][0]
    elif inconsistency == "unpurged_labels":
        metadata["val_range"][0] = metadata["train_range"][1]
    else:
        metadata["test_range"][1] = metadata["n_times"]
    path.write_text(json.dumps(metadata))
    actions = {
        "ensure": lambda: ensure_dataset(cached_args),
        "load": lambda: load_prepared_dataset(cached_args.cache_dir),
        "contract": lambda: cache_data_contract(cached_args.cache_dir),
    }
    with pytest.raises(ValueError, match="data contract.*rebuild"):
        actions[entrypoint]()


@pytest.mark.parametrize("array_name", [
    "timestamps", "bar_inputs", "feature_inputs", "labels", "next_returns",
    "realized_horizon_returns", "bar_mean",
])
@pytest.mark.parametrize("entrypoint", ["ensure", "load"])
def test_cached_array_shapes_must_match_metadata(cached_args, array_name, entrypoint):
    path = Path(cached_args.cache_dir) / f"{array_name}.npy"
    values = np.load(path)
    np.save(path, values[..., :-1])
    with pytest.raises(ValueError, match="data contract.*rebuild"):
        if entrypoint == "ensure":
            ensure_dataset(cached_args)
        else:
            load_prepared_dataset(cached_args.cache_dir)
