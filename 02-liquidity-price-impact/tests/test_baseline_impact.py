from __future__ import annotations

from xgboost_impact import chronological_split, engineer_features, TEST_FRAC
from baseline_impact import evaluate, fit_ridge


def test_fit_ridge_and_evaluate(synthetic_raw_df):
    df, feature_cols, _ = engineer_features(synthetic_raw_df, [1])
    train_df, test_df = chronological_split(df, TEST_FRAC)

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["future_return_1"].to_numpy()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df["future_return_1"].to_numpy()

    model, scaler = fit_ridge(X_train, y_train, alpha=1.0)
    assert model.coef_.shape[0] == len(feature_cols)

    metrics, y_pred = evaluate(model, scaler, X_test, y_test, float(y_train.mean()))
    assert {"rmse", "mae", "r2", "baseline_rmse", "n_test"} <= metrics.keys()
    assert len(y_pred) == len(y_test)
    assert metrics["rmse"] >= 0
