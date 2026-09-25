"""Dos cosas que test_backtest_engine.py no cubría:

1. Invariancia por truncamiento de `build_features_and_labels`: el criterio
   verificable de "sin look-ahead" (agregar filas futuras al final de la
   serie no puede cambiar el valor de una feature ya calculada para un día
   anterior), no solo la ausencia de NaN que ya prueba el test existente.
2. Corrección numérica de equity curve, Max Drawdown y Sharpe contra un
   ejemplo calculado a mano -- el test existente (`test_compute_metrics_
   returns_expected_keys`) solo comprueba que las claves del dict existen,
   nunca que los números sean correctos.

Después agregué lo que encontré que faltaba: `sortino_ratio` en
`backtest_engine.py`, con sus propios tests al final de este archivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_engine import (
    build_features_and_labels,
    compute_metrics,
    sortino_ratio,
    generate_synthetic_crypto_prices,
    run_backtest,
)

CORTE = 150  # bien pasado la ventana rolling mas larga (20d) + los lags


# --------------------------------------------------- invariancia por corte

def test_build_features_and_labels_es_invariante_por_truncamiento():
    prices = generate_synthetic_crypto_prices(n_days=300, seed=11)

    df_completo, feature_cols = build_features_and_labels(prices)
    df_truncado, _ = build_features_and_labels(prices.iloc[:CORTE])

    # Alinear por fecha en vez de por posicion: dropna() en build_features_and_labels
    # descarta un numero distinto de filas iniciales en cada corrida (mismo motivo:
    # las features rezagadas necesitan `max(ventanas)` dias de historia antes de
    # producir un valor), asi que la comparacion tiene que ser por fecha real, no
    # por indice entero.
    comunes = df_truncado["date"]
    completo_alineado = df_completo[df_completo["date"].isin(comunes)].reset_index(drop=True)
    truncado_alineado = df_truncado[df_truncado["date"].isin(comunes)].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        completo_alineado[feature_cols], truncado_alineado[feature_cols],
    )


def test_features_no_dependen_del_precio_del_dia_siguiente():
    """Chequeo directo y no solo estructural: mover el precio de un dia
    futuro no puede cambiar ninguna feature de un dia pasado."""
    prices = generate_synthetic_crypto_prices(n_days=300, seed=12)
    prices_alterado = prices.copy()
    prices_alterado.loc[250, "price"] *= 5.0  # shock enorme, bien despues de CORTE

    df_original, feature_cols = build_features_and_labels(prices)
    df_alterado, _ = build_features_and_labels(prices_alterado)

    pd.testing.assert_frame_equal(
        df_original.loc[df_original["date"] < prices.loc[240, "date"], feature_cols].reset_index(drop=True),
        df_alterado.loc[df_alterado["date"] < prices.loc[240, "date"], feature_cols].reset_index(drop=True),
    )


# ------------------------------------------------- corrección numérica

def test_equity_curve_reproduce_el_producto_acumulado_a_mano():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    signal = np.array([1, 1, 0, 1, 1])
    test_df = pd.DataFrame({"date": dates, "return": returns, "price": 100 * np.cumprod(1 + returns)})

    bt = run_backtest(test_df, signal)

    strategy_returns_esperados = signal * returns
    equity_esperada = np.cumprod(1 + strategy_returns_esperados)

    np.testing.assert_allclose(bt["strategy_return"].to_numpy(), strategy_returns_esperados)
    np.testing.assert_allclose(bt["equity_curve"].to_numpy(), equity_esperada)


def test_max_drawdown_reproduce_una_caida_conocida_a_mano():
    # Equity sube a 1.20, cae a 0.90 (drawdown = 0.90/1.20 - 1 = -0.25),
    # despues sube a 1.10 sin superar el maximo anterior de la caida.
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    equity_a_mano = np.array([1.20, 1.00, 0.90, 1.10])
    returns = np.empty(4)
    returns[0] = equity_a_mano[0] - 1
    returns[1:] = equity_a_mano[1:] / equity_a_mano[:-1] - 1
    signal = np.ones(4, dtype=int)
    # compute_metrics necesita label_up para accuracy_score -- el valor en si
    # es irrelevante para este test (no se comprueba "Signal Accuracy" aca).
    test_df = pd.DataFrame({
        "date": dates, "return": returns, "price": 100.0,
        "label_up": (returns > 0).astype(int),
    })

    bt = run_backtest(test_df, signal)
    metrics = compute_metrics(bt)

    np.testing.assert_allclose(bt["equity_curve"].to_numpy(), equity_a_mano, rtol=1e-9)
    assert metrics["Max Drawdown"] == pytest.approx(0.90 / 1.20 - 1, abs=1e-6)


def test_sharpe_reproduce_la_formula_a_mano_sobre_retornos_conocidos():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    returns = np.array([0.02, -0.01, 0.015, 0.005, -0.02, 0.01])
    signal = np.ones(6, dtype=int)  # siempre long: strategy_return == return
    test_df = pd.DataFrame({
        "date": dates, "return": returns, "price": 100.0,
        "label_up": (returns > 0).astype(int),
    })

    bt = run_backtest(test_df, signal)
    metrics = compute_metrics(bt)

    # ddof=1: compute_metrics usa pandas Series.std(), cuyo default es la
    # desviacion estandar muestral (N-1), no la poblacional (N) de numpy.
    # compute_metrics tambien redondea a 3 decimales antes de devolver.
    sharpe_a_mano = round(returns.mean() / returns.std(ddof=1) * np.sqrt(365), 3)
    assert metrics["Sharpe Ratio (annualized)"] == pytest.approx(sharpe_a_mano, abs=1e-9)


def test_total_return_es_el_ultimo_valor_de_la_equity_curve_menos_uno():
    prices = generate_synthetic_crypto_prices(n_days=200, seed=13)
    df, _ = build_features_and_labels(prices)
    rng = np.random.default_rng(1)
    signal = rng.integers(0, 2, len(df))

    bt = run_backtest(df, signal)
    metrics = compute_metrics(bt)

    # compute_metrics redondea "Total Return" a 3 decimales antes de
    # devolverlo -- comparar contra el valor crudo de la equity curve con
    # una tolerancia de 1e-9 fallaria por el propio redondeo, no por un bug.
    esperado = round(float(bt["equity_curve"].iloc[-1] - 1), 3)
    assert metrics["Total Return"] == pytest.approx(esperado, abs=1e-9)


# --------------------------------------------------------- Sortino

def test_sortino_reproduce_la_formula_a_mano():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    # downside_deviation = sqrt(mean([0, 0.0001, 0, 0.0004, 0])) = 0.01
    esperado = returns.mean() / 0.01 * np.sqrt(365)
    assert sortino_ratio(returns) == pytest.approx(esperado, rel=1e-9)


def test_sortino_es_nan_sin_retornos_por_debajo_del_mar():
    """Sin ningun dia por debajo del minimum acceptable return, la
    desviacion a la baja es 0: el ratio queda indefinido (NaN), no una
    ZeroDivisionError ni un 0.0 que se leeria como "sin retorno ajustado
    por riesgo" cuando en realidad podria ser cualquier cosa."""
    returns = pd.Series([0.01, 0.02, 0.005, 0.03])
    assert np.isnan(sortino_ratio(returns))


def test_sortino_es_nan_con_retornos_todos_cero():
    assert np.isnan(sortino_ratio(pd.Series([0.0, 0.0, 0.0])))


def test_compute_metrics_incluye_sortino_finito_con_signal_siempre_long():
    prices = generate_synthetic_crypto_prices(n_days=200, seed=14)
    df, _ = build_features_and_labels(prices)
    signal = np.ones(len(df), dtype=int)  # siempre long: mezcla dias buenos y malos

    bt = run_backtest(df, signal)
    metrics = compute_metrics(bt)

    assert "Sortino Ratio (annualized)" in metrics
    assert np.isfinite(metrics["Sortino Ratio (annualized)"])
