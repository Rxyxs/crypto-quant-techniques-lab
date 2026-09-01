from __future__ import annotations

import torch

from sklearn.preprocessing import StandardScaler

from xgboost_impact import chronological_split, engineer_features, TEST_FRAC
from pytorch_impact import evaluate, huber_loss, train_model, ACTIVATIONS


def test_huber_loss_matches_reference_formula():
    y_pred = torch.tensor([0.0, 5.0])
    y_true = torch.tensor([0.5, 0.0])
    delta = 1.0
    loss = huber_loss(y_pred, y_true, delta=delta)
    # error[0] = 0.5 (quadratic region), error[1] = -5.0 (linear region)
    expected = (0.5 * 0.5 ** 2 + (delta * (5.0 - delta) + 0.5 * delta ** 2)) / 2
    assert abs(loss.item() - expected) < 1e-5


def test_train_and_evaluate_small_mlp(synthetic_raw_df):
    df, feature_cols, _ = engineer_features(synthetic_raw_df, [1])
    train_df, test_df = chronological_split(df, TEST_FRAC)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.select(feature_cols).to_numpy())
    y_train = train_df["future_return_1"].to_numpy()
    X_test = scaler.transform(test_df.select(feature_cols).to_numpy())
    y_test = test_df["future_return_1"].to_numpy()

    model, loss_history = train_model(X_train, y_train, ACTIVATIONS["relu"], seed=0, epochs=3)
    assert len(loss_history) == 3
    assert all(v >= 0 for v in loss_history)

    metrics, y_pred = evaluate(model, X_test, y_test, float(y_train.mean()))
    assert {"rmse", "mae", "r2", "baseline_rmse", "n_test"} <= metrics.keys()
    assert len(y_pred) == len(y_test)


def test_all_three_activations_are_registered():
    assert set(ACTIVATIONS.keys()) == {"relu", "gelu", "swish"}
