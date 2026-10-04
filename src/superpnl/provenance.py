"""Versioned source identity shared by dataset caches and model checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_VERSION = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_identity(config: dict) -> dict:
    result = dict(config)
    result.pop("cache_dir", None)
    result["raw_dir"] = str(Path(result["raw_dir"]).resolve())
    return result


def raw_fingerprint(raw_dir: str | Path) -> dict[str, str]:
    raw = Path(raw_dir)
    metadata_path = raw / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    symbols = metadata.get("symbols") or [
        path.name.removesuffix(".csv.gz") for path in sorted((raw / "csv").glob("*.csv.gz"))
    ]
    if not symbols:
        raise ValueError(f"no raw symbols under {raw}; use --rebuild-cache with valid source data")
    paths = [raw / "csv" / f"{symbol}.csv.gz" for symbol in symbols]
    if metadata_path.exists():
        paths.append(metadata_path)
    return {str(path.relative_to(raw)): file_sha256(path) for path in paths}


def validate_cache(config: dict, metadata: dict) -> None:
    instruction = "use --rebuild-cache to rebuild the prepared dataset"
    if metadata.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"unsupported or legacy cache version; {instruction}")
    if not metadata.get("config") or config_identity(metadata["config"]) != config_identity(config):
        raise ValueError(f"cache configuration mismatch; {instruction}")
    if metadata.get("source_fingerprint") != raw_fingerprint(config["raw_dir"]):
        raise ValueError(f"raw data changed since cache preparation; {instruction}")


CONTRACT_FIELDS = (
    "symbols", "feature_names", "horizons", "lookback", "train_range",
    "val_range", "test_range", "n_times", "n_symbols", "bar_dim", "feature_dim",
)
STATS_NAMES = ("bar_mean", "bar_std", "feature_mean", "feature_std")


def build_data_contract(metadata: dict, stats: dict) -> dict:
    """Bind semantic schema and normalization to a versioned data source."""
    import numpy as np

    if metadata.get("cache_version") != CACHE_VERSION or not metadata.get("source_fingerprint"):
        raise ValueError("Missing versioned data contract; rebuild cache and retrain the model")
    required = (*CONTRACT_FIELDS, "config")
    if any(key not in metadata for key in required):
        raise ValueError("Incomplete data contract; rebuild cache and retrain the model")
    config = {key: value for key, value in metadata["config"].items()
              if key not in {"cache_dir", "raw_dir"}}
    hashes = {}
    for name in STATS_NAMES:
        if name not in stats or stats[name] is None:
            raise ValueError(f"Missing normalization in data contract: {name}; rebuild cache and retrain")
        array = np.asarray(stats[name], dtype="<f4")
        dimension = metadata["bar_dim"] if name.startswith("bar_") else metadata["feature_dim"]
        if array.shape != (dimension,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid normalization in data contract: {name}")
        if name.endswith("_std") and np.any(array <= 0):
            raise ValueError(f"Non-positive standard deviation in data contract: {name}")
        hashes[name] = hashlib.sha256(array.tobytes()).hexdigest()
    contract = {
        "cache_version": CACHE_VERSION,
        "return_convention": "log(open[t+2]/open[t+1]); signal at close[t]",
        **{key: metadata[key] for key in CONTRACT_FIELDS},
        "preparation_config": config,
        "source_fingerprint": metadata["source_fingerprint"],
        "normalization_sha256": hashes,
    }
    # JSON canonicalizes tuples to lists across fresh and mmap-loaded datasets.
    return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))


def cache_data_contract(cache_dir: str | Path) -> dict:
    import numpy as np

    cache = Path(cache_dir)
    metadata = json.loads((cache / "metadata.json").read_text())
    stats = {name: np.load(cache / f"{name}.npy", allow_pickle=False) for name in STATS_NAMES}
    return build_data_contract(metadata, stats)


def prepared_data_contract(dataset) -> dict:
    if dataset.cache_metadata is None:
        raise ValueError("Missing prepared data contract; rebuild cache and retrain the model")
    metadata = dict(dataset.cache_metadata)
    metadata.update({key: getattr(dataset, key) for key in CONTRACT_FIELDS})
    stats = {name: getattr(dataset, name) for name in STATS_NAMES}
    return build_data_contract(metadata, stats)
