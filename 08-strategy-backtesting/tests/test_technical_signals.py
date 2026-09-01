import numpy as np
import pandas as pd

from technical_signals import (
    compute_all_technical_signals,
    momentum_sign_signal,
    rsi,
    rsi_mean_reversion_signal,
    sma_crossover_signal,
)


def _synthetic_prices(n=200, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices)


def test_sma_crossover_signal_is_binary_and_no_lookahead():
    price = _synthetic_prices()
    signal = sma_crossover_signal(price, fast=5, slow=20)
    assert set(signal.dropna().unique()) <= {0, 1}
    # La señal del día t solo debe depender de precios hasta t-1: recalcular
    # con una cola distinta después del punto de decisión no debe cambiar
    # el valor ya emitido en un índice anterior.
    price_alt = price.copy()
    price_alt.iloc[50:] = price_alt.iloc[50:] * 2
    signal_alt = sma_crossover_signal(price_alt, fast=5, slow=20)
    pd.testing.assert_series_equal(signal.iloc[:49], signal_alt.iloc[:49])


def test_rsi_bounded_between_0_and_100():
    price = _synthetic_prices()
    values = rsi(price, window=14).dropna()
    assert (values >= 0).all()
    assert (values <= 100).all()


def test_rsi_mean_reversion_signal_binary():
    price = _synthetic_prices()
    signal = rsi_mean_reversion_signal(price)
    assert set(signal.dropna().unique()) <= {0, 1}


def test_momentum_sign_signal_matches_sign_of_lagged_mean():
    returns = pd.Series(np.linspace(-0.01, 0.01, 100))
    signal = momentum_sign_signal(returns, window=5)
    momentum = returns.shift(1).rolling(5).mean()
    expected = (momentum > 0).astype(int)
    pd.testing.assert_series_equal(signal, expected)


def test_compute_all_technical_signals_adds_expected_columns():
    price = _synthetic_prices()
    df = pd.DataFrame({"price": price})
    df["return"] = np.log(df["price"]).diff()
    out = compute_all_technical_signals(df)
    for col in ("signal_sma_crossover", "signal_rsi_mean_reversion", "signal_momentum"):
        assert col in out.columns
        assert len(out) == len(df)
