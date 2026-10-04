from __future__ import annotations

import json
import os
import sys
import tarfile
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


def snapshot_tree(directory):
    if not directory.exists():
        return None
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def existing_package(tmp_path, monkeypatch, package_name="package"):
    run_dir, cache_dir, name, _ = make_package_inputs(tmp_path)
    package_dir = tmp_path / package_name
    run_package(monkeypatch, run_dir, cache_dir, name, package_dir)
    tar_path = tmp_path / f"{package_name}.tar.gz"
    return run_dir, cache_dir, name, package_dir, tar_path


@pytest.mark.parametrize("invalid_input", ["checkpoint", "stats", "metadata", "horizon"])
def test_invalid_inputs_preserve_existing_package_and_tarball(tmp_path, monkeypatch, invalid_input):
    run_dir, cache_dir, name, package_dir, tar_path = existing_package(tmp_path, monkeypatch)
    before, tar_before = snapshot_tree(package_dir), tar_path.read_bytes()
    extra = []
    if invalid_input == "checkpoint":
        (run_dir / f"{name}.pt").unlink()
    elif invalid_input == "stats":
        (cache_dir / "bar_std.npy").unlink()
    elif invalid_input == "metadata":
        (cache_dir / "metadata.json").write_text("invalid JSON")
    else:
        extra = ["--recommended-horizon", "30m"]
    with pytest.raises((FileNotFoundError, ValueError)):
        run_package(monkeypatch, run_dir, cache_dir, name, package_dir, "--force", *extra)
    assert snapshot_tree(package_dir) == before
    assert tar_path.read_bytes() == tar_before


def test_tarball_build_failure_preserves_both_old_outputs(tmp_path, monkeypatch):
    run_dir, cache_dir, name, package_dir, tar_path = existing_package(tmp_path, monkeypatch)
    before, tar_before = snapshot_tree(package_dir), tar_path.read_bytes()

    def fail_tar_write(path, *args, **kwargs):
        Path(path).write_bytes(b"partial new archive")
        raise OSError("injected archive write failure")

    monkeypatch.setattr(tarfile, "open", fail_tar_write)
    with pytest.raises(OSError, match="injected"):
        run_package(monkeypatch, run_dir, cache_dir, name, package_dir, "--force")
    assert snapshot_tree(package_dir) == before
    assert tar_path.read_bytes() == tar_before


@pytest.mark.parametrize("fail_on", ["backup_package", "backup_tar", "install_package", "install_tar"])
def test_replace_failure_restores_both_old_outputs(tmp_path, monkeypatch, fail_on):
    run_dir, cache_dir, name, package_dir, tar_path = existing_package(tmp_path, monkeypatch)
    before, tar_before = snapshot_tree(package_dir), tar_path.read_bytes()
    real_replace = os.replace
    failed = False

    def fail_one_replace(source, destination):
        nonlocal failed
        source, destination = Path(source), Path(destination)
        selected = {
            "backup_package": source == package_dir,
            "backup_tar": source == tar_path,
            "install_package": destination == package_dir,
            "install_tar": destination == tar_path,
        }[fail_on]
        if selected and not failed:
            failed = True
            raise OSError("injected replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_one_replace)
    with pytest.raises(OSError, match="injected"):
        run_package(monkeypatch, run_dir, cache_dir, name, package_dir, "--force")
    assert snapshot_tree(package_dir) == before
    assert tar_path.read_bytes() == tar_before


@pytest.mark.parametrize("overlap", ["run", "cache", "ancestor", "tarball"])
def test_output_paths_cannot_overwrite_inputs(tmp_path, monkeypatch, overlap):
    run_dir, cache_dir, name, _ = make_package_inputs(tmp_path)
    package_dir = {"run": run_dir, "cache": cache_dir, "ancestor": tmp_path, "tarball": tmp_path / "package"}[overlap]
    if overlap == "tarball":
        relocated = tmp_path / "package.tar.gz"
        run_dir.rename(relocated)
        run_dir = relocated
    before_run, before_cache = snapshot_tree(run_dir), snapshot_tree(cache_dir)
    with pytest.raises(ValueError, match="overlap"):
        run_package(monkeypatch, run_dir, cache_dir, name, package_dir, "--force")
    assert snapshot_tree(run_dir) == before_run
    assert snapshot_tree(cache_dir) == before_cache


@pytest.mark.parametrize("package_name", ["package", "previous-0", "package.v1"])
def test_replacing_package_updates_both_outputs_with_same_files(tmp_path, monkeypatch, package_name):
    run_dir, cache_dir, name, package_dir, tar_path = existing_package(tmp_path, monkeypatch, package_name)
    (package_dir / "obsolete.txt").write_text("remove on successful replacement")
    run_package(monkeypatch, run_dir, cache_dir, name, package_dir, "--force")
    assert not (package_dir / "obsolete.txt").exists()
    with tarfile.open(tar_path) as archive:
        archived = {
            member.name.removeprefix(f"{package_name}/"): archive.extractfile(member).read()
            for member in archive.getmembers() if member.isfile()
        }
    assert archived == snapshot_tree(package_dir)


def test_persistent_filesystem_error_keeps_recovery_backup(tmp_path, monkeypatch):
    run_dir, cache_dir, name, package_dir, tar_path = existing_package(tmp_path, monkeypatch)
    before, tar_before = snapshot_tree(package_dir), tar_path.read_bytes()
    real_replace = os.replace

    def fail_archive_destination(source, destination):
        if Path(destination) == tar_path:
            raise OSError("archive destination unavailable")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_archive_destination)
    with pytest.raises(RuntimeError, match="recovery"):
        run_package(monkeypatch, run_dir, cache_dir, name, package_dir, "--force")
    assert snapshot_tree(package_dir) == before
    recoverable_files = [path for root in tmp_path.glob(".package.build-*") for path in root.rglob("*") if path.is_file()]
    assert any(path.read_bytes() == tar_before for path in recoverable_files)


def test_existing_archive_without_package_requires_force(tmp_path, monkeypatch):
    run_dir, cache_dir, name, _ = make_package_inputs(tmp_path)
    tar_path = tmp_path / "package.tar.gz"
    tar_path.write_bytes(b"previous archive")
    with pytest.raises(FileExistsError):
        run_package(monkeypatch, run_dir, cache_dir, name, tmp_path / "package")
    assert tar_path.read_bytes() == b"previous archive"
