from argparse import Namespace
import json
from pathlib import Path

import pytest

from scripts.run_superpnl_experiment import ensure_dataset
from superpnl.data import DatasetConfig, load_prepared_dataset, prepare_dataset
from test_data import make_raw_frame, write_raw_frame


@pytest.fixture
def cached_args(tmp_path):
    raw = tmp_path / "raw"
    write_raw_frame(raw, make_raw_frame())
    cache = tmp_path / "cache"
    prepare_dataset(DatasetConfig(str(raw), str(cache), lookback=32))
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
