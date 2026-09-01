"""Hedge ratio dinamico via Filtro de Kalman en espacio de estados, como
alternativa al OLS estatico de `pairs_trading_engine.py`.

Formulacion (estandar en la literatura de pairs trading, ver p.ej. Chan,
"Algorithmic Trading", cap. 3 -- no una variante inventada acá):

    Estado (camino aleatorio, sin dinamica determinista):
        theta_t = [alpha_t, beta_t]'
        theta_t = theta_{t-1} + w_t,      w_t ~ N(0, Q)

    Observacion:
        y_t = F_t @ theta_t + v_t,        v_t ~ N(0, R)
        F_t = [1, x_t]

Es decir: el hedge ratio (beta_t) y el intercepto (alpha_t) ya NO son un
numero fijo estimado una sola vez sobre toda la muestra -- son un estado
oculto que el filtro re-estima cada dia, dejando que la relacion de
cointegracion "derive" con el tiempo (regimenes cambiantes de mercado).

Una propiedad util de esta formulacion, no solo una curiosidad matematica: el
**error de innovacion** `e_t = y_t - F_t @ theta_{t|t-1}` (la prediccion del
filtro para y_t, ANTES de ver el dato real de y_t, usando el estado de ayer)
es exactamente el spread -- ya no hace falta computarlo por separado restando
alpha+beta*x. Y su varianza de innovacion `S_t` (que el filtro ya calcula
como parte de la recursion) da un z-score sin necesidad de una ventana
rolling arbitraria: `zscore_t = e_t / sqrt(S_t)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# "delta" controla que tan rapido puede derivar theta_t de un dia a otro: un
# delta mas grande permite una adaptacion mas rapida (pero mas ruidosa) del
# hedge ratio; uno mas chico lo hace mas suave (pero mas lento para
# reaccionar a un cambio real de regimen). 1e-4 es el valor de referencia
# estandar en la literatura de pairs trading (Chan, cap. 3) para datos
# diarios -- no una eleccion arbitraria de este proyecto.
DEFAULT_DELTA = 1e-4
# Ventana de "burn-in": los primeros `BURN_IN_DAYS` dias se usan solo para
# estimar un R inicial (via un OLS estatico corto) antes de arrancar el
# filtro -- sin esto, R_0 tendria que asumirse arbitrariamente.
BURN_IN_DAYS = 60
# Prior difuso sobre el estado inicial: una covarianza inicial grande deja
# que el filtro converja rapido a partir de los datos reales, en vez de
# anclarse a un theta_0 = [0, 0] que no significa nada.
DIFFUSE_PRIOR_SCALE = 1e4
# R (varianza de observacion) se deja evolucionar via EWMA de la innovacion
# al cuadrado, en vez de quedar fija en la estimacion del burn-in para
# siempre. Dos alternativas fijas se probaron primero y ambas fallaron
# empiricamente: R fijo desde el burn-in de 60 dias (z-score con desvio
# ~0.46 en vez de ~1, apenas 2 cruces del umbral de entrada en 1000 dias) y
# R fijo estimado sobre la muestra COMPLETA (peor aun: 0 trades). Ambos
# supuestos implican que la volatilidad residual del spread es constante en
# ~3 anos, lo cual la propia evidencia contradice. `R_EWM_LAMBDA` se fija en
# 2/(30+1) -- la conversion estandar de un "span" de EWMA a una ventana
# equivalente -- para que la velocidad de adaptacion de R combine con la
# misma ventana de 30 dias que ya usa `ROLLING_ZSCORE_WINDOW` en el enfoque
# OLS, no un valor elegido para maximizar el retorno del backtest.
R_EWM_LAMBDA = 2 / (30 + 1)


@dataclass
class KalmanFilterResult:
    alpha: pd.Series
    beta: pd.Series
    spread: pd.Series               # error de innovacion e_t (= el spread)
    innovation_variance: pd.Series  # S_t
    zscore: pd.Series               # e_t / sqrt(S_t)
    observation_variance_r: float
    process_variance_q: np.ndarray


def _estimate_observation_variance(y: pd.Series, x: pd.Series, burn_in_days: int = BURN_IN_DAYS) -> float:
    """R (varianza de observacion) se estima una sola vez, via la varianza
    residual de un OLS estatico sobre los primeros `burn_in_days` --
    consistente con el hedge ratio estatico de `pairs_trading_engine.py` en
    ese tramo inicial, antes de que el filtro empiece a dejar que theta_t derive."""
    import statsmodels.api as sm

    y_burn, x_burn = y.iloc[:burn_in_days], x.iloc[:burn_in_days]
    model = sm.OLS(y_burn, sm.add_constant(x_burn.rename("x"))).fit()
    residual_variance = float(np.var(model.resid, ddof=2))
    return max(residual_variance, 1e-8)  # piso para evitar R=0 si el burn-in es degenerado


def run_kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = DEFAULT_DELTA,
    burn_in_days: int = BURN_IN_DAYS,
    r_ewm_lambda: float = R_EWM_LAMBDA,
) -> KalmanFilterResult:
    """Corre el filtro de Kalman de punta a punta sobre `y` (log-precio) en
    funcion de `x` (log-precio), devolviendo theta_t = [alpha_t, beta_t] día
    a día junto con el spread (innovación) y su z-score natural.

    R se inicializa con el burn-in y luego se actualiza cada paso via EWMA
    de `e_t^2` (`R_t = (1-lambda)*R_{t-1} + lambda*e_t^2`) -- R fijo para
    siempre asume que la volatilidad residual del spread nunca cambia en
    ~3 años de datos, un supuesto que la validación empírica (ver
    `kalman_pairs.py` -- comentario junto a `R_EWM_LAMBDA`) mostró que rompe
    la calibración del z-score bien pasado el período de burn-in.
    """
    n = len(y)
    if len(x) != n:
        raise ValueError("y y x deben tener la misma longitud")

    R = _estimate_observation_variance(y, x, burn_in_days)
    Q = delta / (1 - delta) * np.eye(2)

    theta = np.zeros(2)
    P = DIFFUSE_PRIOR_SCALE * np.eye(2)

    alphas = np.empty(n)
    betas = np.empty(n)
    innovations = np.empty(n)
    innovation_variances = np.empty(n)
    r_values = np.empty(n)

    x_values = x.to_numpy()
    y_values = y.to_numpy()

    for t in range(n):
        F = np.array([1.0, x_values[t]])

        # --- prediccion: caminata aleatoria, la transicion es la identidad ---
        theta_pred = theta
        P_pred = P + Q

        # --- actualizacion ---
        y_hat = F @ theta_pred
        e = y_values[t] - y_hat
        S = float(F @ P_pred @ F + R)
        K = (P_pred @ F) / S

        theta = theta_pred + K * e
        P = P_pred - np.outer(K, F) @ P_pred

        alphas[t] = theta[0]
        betas[t] = theta[1]
        innovations[t] = e
        innovation_variances[t] = S
        r_values[t] = R

        # actualiza R para el proximo paso (no afecta la S ya calculada de este paso)
        R = (1 - r_ewm_lambda) * R + r_ewm_lambda * e**2

    index = y.index
    zscore = innovations / np.sqrt(innovation_variances)

    return KalmanFilterResult(
        alpha=pd.Series(alphas, index=index, name="kalman_alpha"),
        beta=pd.Series(betas, index=index, name="kalman_beta"),
        spread=pd.Series(innovations, index=index, name="spread"),
        innovation_variance=pd.Series(innovation_variances, index=index, name="innovation_variance"),
        zscore=pd.Series(zscore, index=index, name="zscore"),
        observation_variance_r=float(np.mean(r_values)),
        process_variance_q=Q,
    )
