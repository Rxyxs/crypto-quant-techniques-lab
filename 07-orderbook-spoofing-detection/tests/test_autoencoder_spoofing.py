import numpy as np

from autoencoder_spoofing import (
    ACTIVATIONS,
    compare_activations,
    fit_predict_autoencoder,
    reconstruction_loss,
)
from detect_spoofing import L2_FEATURE_COLS, engineer_features, simulate_orderbook

import torch


def _sim(n=600, seed=7):
    rng = np.random.default_rng(seed)
    return engineer_features(simulate_orderbook(n, rng))


def test_reconstruction_loss_is_per_row_and_nonnegative():
    x = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    x_hat = torch.tensor([[1.0, 2.0], [1.0, 1.0]])
    loss = reconstruction_loss(x_hat, x)
    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert loss[0].item() == 0.0
    assert loss[1].item() > 0.0


def test_fit_predict_autoencoder_returns_expected_shapes():
    df = _sim()
    model, scaler, is_anomaly, scores, history = fit_predict_autoencoder(df, feature_cols=L2_FEATURE_COLS)
    assert len(is_anomaly) == df.height
    assert len(scores) == df.height
    assert is_anomaly.dtype == bool
    assert len(history) > 0
    # loss should generally decrease over training
    assert history[-1] <= history[0]


def test_contamination_controls_flag_rate():
    df = _sim()
    _, _, is_anomaly_tight, _, _ = fit_predict_autoencoder(df, contamination=0.01, seed=1)
    _, _, is_anomaly_loose, _, _ = fit_predict_autoencoder(df, contamination=0.05, seed=1)
    assert is_anomaly_loose.sum() >= is_anomaly_tight.sum()


def test_compare_activations_covers_all_three():
    df = _sim(n=500, seed=11)
    comparison = compare_activations(df)
    assert set(comparison["activation"].to_list()) == set(ACTIVATIONS.keys())
    for col in ("precision", "recall", "f1"):
        assert (comparison[col] >= 0).all()
        assert (comparison[col] <= 1).all()
