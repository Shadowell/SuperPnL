from types import SimpleNamespace

import numpy as np
import pytest

from superpnl.training import LowTurnoverConfig, backtest_low_turnover_scores


def dataset_for(gross_returns):
    gross = np.asarray(gross_returns, dtype="float64")
    return SimpleNamespace(
        n_symbols=len(gross), next_returns=np.log(gross),
        train_range=(0, gross.shape[1]), val_range=(0, gross.shape[1]), test_range=(0, gross.shape[1]),
        feature_names=[], feature_inputs=np.empty((*gross.shape, 0)), feature_mean=np.array([]), feature_std=np.array([]),
    )


def policy(**overrides):
    values = dict(top_k=1, rebalance_interval_bars=10, min_holding_bars=0, cooldown_bars=0,
                  max_position_per_symbol=0.75, max_total_position=0.75)
    return LowTurnoverConfig(**(values | overrides))


def test_low_turnover_accepts_target_weight_above_equal_weight_sleeve():
    ds = dataset_for([[2.0], [1.0]])
    metrics, positions, returns = backtest_low_turnover_scores(ds, np.array([[[2.0]], [[1.0]]]), "test", 0, config=policy())

    assert metrics["total_return"] == pytest.approx(0.75)
    assert metrics["gross_total_return"] == pytest.approx(0.75)
    assert np.expm1(returns.sum()) == pytest.approx(0.75)
    np.testing.assert_allclose(positions, [[0.75], [0.0]])


def test_low_turnover_holds_units_and_charges_only_initial_buy_between_rebalances():
    ds = dataset_for([[2.0, 0.5], [1.0, 1.0]])
    pred = np.array([[[2.0], [2.0]], [[1.0], [1.0]]])
    metrics, positions, _ = backtest_low_turnover_scores(ds, pred, "test", 0, fixed_fee_bps=100.0, config=policy())

    assert metrics["total_return"] == pytest.approx(1.0 / 1.0075 - 1.0)
    assert metrics["gross_total_return"] == pytest.approx(0.0)
    assert metrics["cost_return"] == pytest.approx(0.0075 / 1.0075)
    assert metrics["trade_count"] == 1
    np.testing.assert_allclose(positions[:, 1], [1.5 / 1.75, 0.0])


def test_low_turnover_rebalances_at_configured_steps_only():
    ds = dataset_for([[2.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    pred = np.array([[[2.0], [-1.0], [-1.0]], [[1.0], [1.0], [1.0]]])
    metrics, positions, _ = backtest_low_turnover_scores(
        ds, pred, "test", 0, fixed_fee_bps=100.0, config=policy(rebalance_interval_bars=2)
    )

    np.testing.assert_allclose(positions[:, 0], [0.75, 0.0])
    np.testing.assert_allclose(positions[:, 1], [1.5 / 1.75, 0.0])
    np.testing.assert_allclose(positions[:, 2], [0.0, 0.75])
    assert metrics["trade_count"] == 3


def test_low_turnover_preserves_min_holding_and_cooldown_rules():
    ds = dataset_for([[1.0] * 6])
    pred = np.array([[[1.0], [-1.0], [-1.0], [1.0], [1.0], [1.0]]])
    _, positions, _ = backtest_low_turnover_scores(
        ds, pred, "test", 0,
        config=policy(rebalance_interval_bars=1, min_holding_bars=2, cooldown_bars=2),
    )

    np.testing.assert_allclose(positions, [[0.75, 0.75, 0.0, 0.0, 0.75, 0.75]])


def test_turnover_cap_preserves_total_position_limit_during_rotation():
    ds = dataset_for(np.ones((3, 7)))
    pred = np.full((3, 7, 1), -1.0)
    pred[0, :6, 0] = 2.0
    pred[1:, 6, 0] = [2.0, 1.0]
    cfg = policy(top_k=2, rebalance_interval_bars=1, max_position_per_symbol=0.6,
                 max_total_position=0.6, max_turnover_per_step=0.1)

    _, positions, _ = backtest_low_turnover_scores(ds, pred, "test", 0, config=cfg)

    np.testing.assert_allclose(positions[:, 5], [0.6, 0.0, 0.0])
    assert np.all(positions.sum(axis=0) <= 0.6 + 1e-12)
    assert np.abs(np.diff(positions, axis=1, prepend=0.0)).max() <= 0.1 + 1e-12
    np.testing.assert_allclose(positions[:, 6], [0.5, 0.05, 0.05])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["score", "threshold"])
def test_low_turnover_rejects_nonfinite_inputs(invalid, field):
    ds = dataset_for([[1.0, 1.0]])
    pred = np.ones((1, 2, 1))
    if field == "score":
        pred[0, 1, 0] = invalid  # Also reject invalid scores on non-rebalance bars.
    with pytest.raises(ValueError, match="finite"):
        backtest_low_turnover_scores(ds, pred, "test", 0,
                                     threshold_bps=invalid if field == "threshold" else 0.0, config=policy())


def test_best_validation_checkpoint_keeps_training_data_contract(tmp_path, monkeypatch):
    import torch

    from superpnl import training
    from superpnl.data import DatasetConfig, prepare_dataset
    from superpnl.provenance import prepared_data_contract
    from test_data import make_raw_frame, write_raw_frame

    write_raw_frame(tmp_path / "raw", make_raw_frame())
    ds = prepare_dataset(DatasetConfig(str(tmp_path / "raw"), str(tmp_path / "cache"),
                                       lookback=8, horizons=(1, 5), feature_windows=(5, 15, 30)))
    epoch_states = []

    def validation_result(dataset, model, *args, **kwargs):
        epoch_states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        score = 0.5 if len(epoch_states) == 1 else 0.1
        return {"horizon_metrics": {"1m": {"mae": 0.01, "rank_ic": score}}}

    monkeypatch.setattr(training, "evaluate_model", validation_result)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model, result = training.train_model(
            ds, False, training.TrainConfig(hidden_dim=4, epochs=2, samples_per_epoch=2,
                                             batch_size=2, device="cpu"), tmp_path / "run", "tiny"
        )
    finally:
        torch.set_num_threads(previous_threads)
    checkpoint = torch.load(result["model_path"], map_location="cpu", weights_only=True)

    assert checkpoint["best_record"]["epoch"] == 1
    assert checkpoint["best_score"] == 0.5
    assert checkpoint["data_contract"] == prepared_data_contract(ds)
    assert len(result["history"]) == 2
    assert any(not torch.equal(epoch_states[0][key], epoch_states[1][key]) for key in epoch_states[0])
    for key in epoch_states[0]:
        torch.testing.assert_close(checkpoint["model"][key], epoch_states[0][key])
        torch.testing.assert_close(model.state_dict()[key], epoch_states[0][key])
