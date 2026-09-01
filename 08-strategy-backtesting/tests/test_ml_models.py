import numpy as np

from backtest_engine import build_features_and_labels, chronological_split, generate_synthetic_crypto_prices
from ml_models import train_lightgbm, train_logistic_regression


def _train_test_split():
    prices = generate_synthetic_crypto_prices(n_days=400, seed=11)
    df, feature_cols = build_features_and_labels(prices)
    train_df, test_df = chronological_split(df)
    return train_df, test_df, feature_cols


def test_train_logistic_regression_outputs_are_valid():
    train_df, test_df, feature_cols = _train_test_split()
    signal, proba = train_logistic_regression(train_df, test_df, feature_cols)
    assert len(signal) == len(test_df)
    assert set(np.unique(signal)) <= {0, 1}
    assert (proba >= 0).all() and (proba <= 1).all()
    # La señal debe ser exactamente el umbral de 0.5 sobre la probabilidad.
    assert np.array_equal(signal, (proba >= 0.5).astype(int))


def test_train_lightgbm_outputs_are_valid():
    train_df, test_df, feature_cols = _train_test_split()
    signal, proba = train_lightgbm(train_df, test_df, feature_cols)
    assert len(signal) == len(test_df)
    assert set(np.unique(signal)) <= {0, 1}
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.array_equal(signal, (proba >= 0.5).astype(int))
