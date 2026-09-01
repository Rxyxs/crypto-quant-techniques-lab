"""Señales técnicas tradicionales (rule-based), usadas como benchmark frente
al clasificador de Gradient Boosting en `compare_strategies.py`.

Las tres son reglas ampliamente conocidas en análisis técnico clásico, cada
una representando una familia distinta de estrategia:

- **Cruce de medias móviles (SMA crossover)**: seguimiento de tendencia.
- **RSI de reversión a la media**: apuesta a que un activo sobrevendido
  rebota.
- **Señal de momentum (signo)**: seguimiento de tendencia de corto plazo,
  la versión más simple posible de "el retorno reciente predice el
  siguiente".

Todas producen una señal 0/1 (flat/long) para el día *t* usando
exclusivamente información disponible hasta el cierre de *t-1* -- el mismo
estándar sin lookahead que exige `backtest_engine.build_features_and_labels`
para el modelo de ML, así que la comparación es limpia: ninguna estrategia,
técnica o de ML, ve el futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma_crossover_signal(price: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """Long cuando la media móvil rápida (rezagada 1 día) está por encima de
    la lenta -- la regla de seguimiento de tendencia más clásica."""
    sma_fast = price.rolling(fast).mean().shift(1)
    sma_slow = price.rolling(slow).mean().shift(1)
    return (sma_fast > sma_slow).astype(int)


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_mean_reversion_signal(price: pd.Series, window: int = 14, oversold: float = 30.0) -> pd.Series:
    """Long cuando el RSI (rezagado 1 día) cae bajo el umbral de sobreventa
    -- apuesta a un rebote de corto plazo, la lógica clásica de
    mean-reversion técnico."""
    rsi_values = rsi(price, window).shift(1)
    return (rsi_values < oversold).astype(int)


def momentum_sign_signal(returns: pd.Series, window: int = 10) -> pd.Series:
    """Long cuando el retorno promedio de los últimos `window` días
    (rezagado 1 día) es positivo -- la forma más simple de seguimiento de
    tendencia de corto plazo."""
    momentum = returns.shift(1).rolling(window).mean()
    return (momentum > 0).astype(int)


def compute_all_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las tres columnas de señal técnica a un DataFrame que ya tiene
    `price` y `return`."""
    out = df.copy()
    out["signal_sma_crossover"] = sma_crossover_signal(out["price"])
    out["signal_rsi_mean_reversion"] = rsi_mean_reversion_signal(out["price"])
    out["signal_momentum"] = momentum_sign_signal(out["return"])
    return out
