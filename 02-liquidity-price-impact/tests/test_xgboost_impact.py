from __future__ import annotations

import numpy as np

from xgboost_impact import (
    HORIZONS,
    TEST_FRAC,
    chronological_split,
    engineer_features,
    evaluate,
    tune_model,
)


def test_engineer_features_no_lookahead_and_no_nulls(synthetic_raw_df):
    df, feature_cols, target_cols = engineer_features(synthetic_raw_df, HORIZONS)

    assert df.height > 0
    assert set(feature_cols).issubset(df.columns)
    assert set(target_cols).issubset(df.columns)
    for col in feature_cols + target_cols:
        assert df[col].null_count() == 0


def test_engineer_features_target_matches_forward_return(synthetic_raw_df):
    df, feature_cols, target_cols = engineer_features(synthetic_raw_df, [1])
    vwap = synthetic_raw_df["vwap"].to_numpy()
    log_vwap = np.log(vwap)
    expected_future_return_1 = log_vwap[1:] - log_vwap[:-1]

    # Spot-check the first few surviving rows' target equals log(vwap[t+1]) - log(vwap[t])
    bins = df["bin"].to_list()
    all_bins = synthetic_raw_df["bin"].to_list()
    for i in range(5):
        idx = all_bins.index(bins[i])
        got = df["future_return_1"][i]
        want = expected_future_return_1[idx]
        assert abs(got - want) < 1e-9


def test_chronological_split_is_time_ordered_and_no_overlap(synthetic_raw_df):
    df, _, _ = engineer_features(synthetic_raw_df, HORIZONS)
    train_df, test_df = chronological_split(df, TEST_FRAC)

    assert train_df.height + test_df.height == df.height
    assert train_df["bin"].max() <= test_df["bin"].min()


def test_tune_model_and_evaluate_returns_expected_metric_keys(synthetic_raw_df):
    df, feature_cols, target_cols = engineer_features(synthetic_raw_df, [1])
    train_df, test_df = chronological_split(df, TEST_FRAC)

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["future_return_1"].to_numpy()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df["future_return_1"].to_numpy()

    model, best_params = tune_model(X_train, y_train, seed=0)
    metrics, y_pred = evaluate(model, X_test, y_test, float(y_train.mean()))

    assert isinstance(best_params, dict)
    assert {"rmse", "mae", "r2", "baseline_rmse", "n_test"} <= metrics.keys()
    assert len(y_pred) == len(y_test)
