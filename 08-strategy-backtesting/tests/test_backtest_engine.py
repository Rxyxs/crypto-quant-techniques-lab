import numpy as np
import pandas as pd

from backtest_engine import (
    build_features_and_labels,
    chronological_split,
    compute_metrics,
    generate_synthetic_crypto_prices,
    run_backtest,
)


def test_generate_synthetic_crypto_prices_shape_and_positivity():
    df = generate_synthetic_crypto_prices(n_days=100, seed=1)
    assert len(df) == 100
    assert {"date", "price"} <= set(df.columns)
    assert (df["price"] > 0).all()


def test_generate_synthetic_crypto_prices_is_deterministic_given_seed():
    a = generate_synthetic_crypto_prices(n_days=50, seed=7)
    b = generate_synthetic_crypto_prices(n_days=50, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_build_features_and_labels_has_no_lookahead_nans_left():
    prices = generate_synthetic_crypto_prices(n_days=300, seed=2)
    df, feature_cols = build_features_and_labels(prices)
    assert len(feature_cols) > 0
    assert not df[feature_cols].isna().any().any()
    assert set(df["label_up"].unique()) <= {0, 1}


def test_chronological_split_preserves_order_and_fraction():
    prices = generate_synthetic_crypto_prices(n_days=300, seed=3)
    df, _ = build_features_and_labels(prices)
    train_df, test_df = chronological_split(df, train_fraction=0.7)
    assert len(train_df) + len(test_df) == len(df)
    assert train_df["date"].max() <= test_df["date"].min()
    assert abs(len(train_df) / len(df) - 0.7) < 0.01


def test_run_backtest_flat_signal_has_zero_return_and_no_drawdown():
    prices = generate_synthetic_crypto_prices(n_days=300, seed=4)
    df, _ = build_features_and_labels(prices)
    _, test_df = chronological_split(df)
    flat_signal = np.zeros(len(test_df), dtype=int)
    bt = run_backtest(test_df, flat_signal)
    assert (bt["strategy_return"] == 0).all()
    assert (bt["equity_curve"] == 1.0).all()
    assert (bt["drawdown"] == 0).all()


def test_compute_metrics_returns_expected_keys():
    prices = generate_synthetic_crypto_prices(n_days=300, seed=5)
    df, _ = build_features_and_labels(prices)
    _, test_df = chronological_split(df)
    rng = np.random.default_rng(0)
    signal = rng.integers(0, 2, len(test_df))
    bt = run_backtest(test_df, signal)
    metrics = compute_metrics(bt)
    expected_keys = {
        "Sharpe Ratio (annualized)", "Win Rate", "Max Drawdown",
        "Total Return", "Signal Accuracy", "Trades Taken",
    }
    assert expected_keys <= set(metrics.keys())
