"""Modern Portfolio Theory (Markowitz, 1952) applied to a basket of real
crypto assets: expected-return/covariance estimation, the efficient
frontier, the maximum-Sharpe portfolio, and the minimum-volatility
portfolio, all via constrained optimization (SciPy SLSQP).

Crypto trades 365 days/year (no weekend close, unlike equities), so
annualization throughout uses 365 trading days, not the usual 252.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 365
RISK_FREE_RATE = 0.045  # illustrative annualized risk-free rate


def load_prices(csv_path: str | Path) -> pd.DataFrame:
    prices = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    return prices.sort_index()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def annualize_mean_returns(daily_returns: pd.DataFrame) -> pd.Series:
    return daily_returns.mean() * TRADING_DAYS_PER_YEAR


def annualize_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    return daily_returns.cov() * TRADING_DAYS_PER_YEAR


def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series,
                           cov_matrix: pd.DataFrame) -> tuple[float, float, float]:
    """Returns (expected annual return, annual volatility, Sharpe ratio)."""
    port_return = float(np.dot(weights, mean_returns))
    port_vol = float(np.sqrt(weights @ cov_matrix.values @ weights))
    sharpe = (port_return - RISK_FREE_RATE) / port_vol
    return port_return, port_vol, sharpe


def _bounds_and_x0(n_assets: int) -> tuple[tuple, np.ndarray]:
    # Long-only: each weight in [0, 1], fully invested (sum of weights = 1).
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    x0 = np.repeat(1.0 / n_assets, n_assets)
    return bounds, x0


def optimize_max_sharpe(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
    n = len(mean_returns)
    bounds, x0 = _bounds_and_x0(n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w: np.ndarray) -> float:
        _, _, sharpe = portfolio_performance(w, mean_returns, cov_matrix)
        return -sharpe

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")
    return result.x


def optimize_min_volatility(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
    n = len(mean_returns)
    bounds, x0 = _bounds_and_x0(n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def volatility(w: np.ndarray) -> float:
        return float(np.sqrt(w @ cov_matrix.values @ w))

    result = minimize(volatility, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Min-volatility optimization failed: {result.message}")
    return result.x


def efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                        n_points: int = 60) -> pd.DataFrame:
    """For each target return between the min-vol and max-return assets,
    finds the minimum-volatility portfolio achieving at least that return
    -- the definition of the efficient frontier.
    """
    n = len(mean_returns)
    bounds, x0 = _bounds_and_x0(n)

    min_vol_weights = optimize_min_volatility(mean_returns, cov_matrix)
    min_vol_return, _, _ = portfolio_performance(min_vol_weights, mean_returns, cov_matrix)
    max_return = float(mean_returns.max())

    target_returns = np.linspace(min_vol_return, max_return, n_points)
    frontier_returns, frontier_vols = [], []

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=target: np.dot(w, mean_returns) - target},
        )

        def volatility(w: np.ndarray) -> float:
            return float(np.sqrt(w @ cov_matrix.values @ w))

        result = minimize(volatility, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            frontier_returns.append(target)
            frontier_vols.append(result.fun)

    return pd.DataFrame({"return": frontier_returns, "volatility": frontier_vols})


def random_portfolios(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                       n_portfolios: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Random long-only portfolios (Dirichlet-sampled weights), used only
    to visually contextualize the efficient frontier against the feasible
    set -- not part of the optimization itself.
    """
    rng = np.random.default_rng(seed)
    n = len(mean_returns)
    weights = rng.dirichlet(np.ones(n), size=n_portfolios)

    returns = weights @ mean_returns.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix.values, weights))
    sharpes = (returns - RISK_FREE_RATE) / vols
    return pd.DataFrame({"return": returns, "volatility": vols, "sharpe": sharpes})
