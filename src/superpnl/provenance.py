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
