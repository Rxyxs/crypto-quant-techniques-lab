"""Auditoría de look-ahead bias con un criterio verificable en código, no
solo lectura del código: "la señal en el día t no cambia si le agrego más
datos futuros después de t" (invariancia por truncamiento).

Historia de este archivo:

- Encontré y probé el hallazgo: `run_full_pipeline` llamaba
  `estimate_hedge_ratio` UNA sola vez sobre el panel completo y reusaba ese
  beta para puntuar la serie entera, incluidos los primeros días -- que en
  la práctica ocurrieron antes de que existieran los datos con los que se
  calculó ese beta. `test_hedge_ratio_estatico_...` demuestra esto con
  números sobre las funciones en sí (`estimate_hedge_ratio` + `compute_spread`
  siguen sin ser invariantes por truncamiento si se usan así, por diseño --
  un OLS de muestra completa no puede serlo).
- Corregí `run_full_pipeline` para que la señal real -- la que
  efectivamente entra al backtest -- use el hedge ratio dinámico del filtro
  de Kalman en vez del OLS estático. `test_pipeline_de_kalman_produce_
  posiciones_invariantes_por_truncamiento` prueba la ruta que el pipeline
  usa hoy, de punta a punta (beta -> zscore -> posición).

El primer test se mantiene tal cual: sigue siendo verdad sobre esas dos
funciones usadas de esa forma, y es la evidencia de por qué el pipeline dejó
de usarlas así.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalman_pairs import run_kalman_hedge_ratio
from pairs_trading_engine import (
    compute_spread,
    estimate_hedge_ratio,
    generate_signals,
    rolling_zscore,
)

N = 500
CORTE = 300  # bien pasado cualquier burn-in/ventana usada abajo
RNG = np.random.default_rng(42)


def _cointegrated_pair(n: int = N, beta: float = 0.5, alpha: float = 1.0, noise_std: float = 0.05):
    """Mismo generador que test_pairs_trading_engine.py -- y = alpha + beta*x
    + ruido AR(1) estacionario, cointegrados por construcción."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    x = pd.Series(np.cumsum(RNG.normal(0, 0.02, n)) + 4.0, index=idx, name="X")
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.6 * eps[t - 1] + RNG.normal(0, noise_std)
    y = pd.Series(alpha + beta * x.to_numpy() + eps, index=idx, name="Y")
    return y, x


# --------------------------------------------------------- piezas seguras

def test_rolling_zscore_es_invariante_por_truncamiento():
    y, x = _cointegrated_pair()
    hedge = estimate_hedge_ratio(y, x)  # fit único, solo para tener un spread fijo que testear
    spread = compute_spread(y, x, hedge)

    z_completo = rolling_zscore(spread)
    z_truncado = rolling_zscore(spread.iloc[:CORTE])

    pd.testing.assert_series_equal(
        z_completo.iloc[:CORTE], z_truncado, check_names=False,
    )


def test_kalman_hedge_ratio_es_invariante_por_truncamiento():
    """El filtro de Kalman es, por diseño, una recursión hacia adelante:
    theta_t solo depende de theta_{t-1} y de la observación en t. Agregar
    filas futuras después del corte no debería cambiar ni un decimal de lo
    que el filtro ya calculó para las filas anteriores."""
    y, x = _cointegrated_pair()

    completo = run_kalman_hedge_ratio(y, x)
    truncado = run_kalman_hedge_ratio(y.iloc[:CORTE], x.iloc[:CORTE])

    pd.testing.assert_series_equal(completo.beta.iloc[:CORTE], truncado.beta, check_names=False)
    pd.testing.assert_series_equal(completo.spread.iloc[:CORTE], truncado.spread, check_names=False)
    pd.testing.assert_series_equal(completo.zscore.iloc[:CORTE], truncado.zscore, check_names=False)


# ------------------------------------------------- la pieza que SÍ filtra

def test_hedge_ratio_estatico_no_es_invariante_por_truncamiento():
    """Esto es un hallazgo, no una aspiración: `run_full_pipeline` llama
    `estimate_hedge_ratio(log_y, log_x)` UNA sola vez sobre el panel de
    precios completo, y usa ese mismo beta para construir el spread (y por
    lo tanto la señal) de la serie ENTERA -- incluidos los primeros días del
    backtest, que en la práctica ocurrieron antes de que existieran los
    datos con los que se calculó ese beta.

    Esta prueba lo demuestra con números, no con una opinión: si el beta
    ajustado sobre el prefijo (lo único que un trader real tendría
    disponible en el día `CORTE`) fuera igual al beta de la serie completa,
    el spread de los primeros `CORTE` días también sería igual bajo ambos
    ajustes -- y no lo es.
    """
    y, x = _cointegrated_pair()

    hedge_completo = estimate_hedge_ratio(y, x)
    hedge_prefijo = estimate_hedge_ratio(y.iloc[:CORTE], x.iloc[:CORTE])

    # El beta con toda la muestra "sabe" de datos que en el día CORTE todavía
    # no existían -- por eso difiere del beta que solo ve el prefijo.
    assert hedge_completo.beta != pytest.approx(hedge_prefijo.beta, rel=1e-6)

    spread_con_beta_completo = compute_spread(y, x, hedge_completo).iloc[:CORTE]
    spread_con_beta_prefijo = compute_spread(y.iloc[:CORTE], x.iloc[:CORTE], hedge_prefijo)

    diferencia_media_abs = (spread_con_beta_completo - spread_con_beta_prefijo).abs().mean()
    assert diferencia_media_abs > 1e-3, (
        "el spread de los primeros dias cambia segun si el hedge ratio vio "
        "datos futuros o no -- esa dependencia del futuro es exactamente la "
        "fuga que este test documenta"
    )


# ------------------------------------- la ruta que el pipeline usa hoy

def test_pipeline_de_kalman_produce_posiciones_invariantes_por_truncamiento():
    """Prueba de punta a punta sobre exactamente lo que `run_full_pipeline`
    corre hoy: `run_kalman_hedge_ratio` -> `generate_signals`.
    `generate_signals` es un loop hacia adelante con estado (mantiene la
    posición previa entre bandas de entrada/salida) -- esta prueba confirma
    que ese estado tampoco introduce dependencia del futuro: la posición del
    día t bajo la serie completa tiene que ser idéntica a la posición del
    día t calculada solo con datos hasta t."""
    y, x = _cointegrated_pair()

    kalman_completo = run_kalman_hedge_ratio(y, x)
    kalman_truncado = run_kalman_hedge_ratio(y.iloc[:CORTE], x.iloc[:CORTE])

    posiciones_completo = generate_signals(kalman_completo.zscore)
    posiciones_truncado = generate_signals(kalman_truncado.zscore)

    pd.testing.assert_series_equal(
        posiciones_completo.iloc[:CORTE], posiciones_truncado, check_names=False,
    )
