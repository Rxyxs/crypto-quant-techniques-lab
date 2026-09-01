"""Tests for portfolio_optimizer.py (Markowitz mean-variance optimization)
and results_store.py (DuckDB persistence of optimization runs).

Uses small synthetic price series (not the real Binance data) so tests are
fast and deterministic; the real data path is exercised by the notebook
itself (README §8).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import portfolio_optimizer as po
import results_store as rs


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_days = 500
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    assets = {
        "AAA": 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.02, n_days))),
        "BBB": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n_days))),
        "CCC": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.03, n_days))),
    }
    return pd.DataFrame(assets, index=dates)


@pytest.fixture
def stats(synthetic_prices):
    daily_returns = po.compute_log_returns(synthetic_prices)
    mean_returns = po.annualize_mean_returns(daily_returns)
    cov_matrix = po.annualize_covariance(daily_returns)
    return mean_returns, cov_matrix


def test_compute_log_returns_drops_first_row(synthetic_prices):
    daily_returns = po.compute_log_returns(synthetic_prices)
    assert len(daily_returns) == len(synthetic_prices) - 1
    assert not daily_returns.isna().any().any()


def test_annualize_covariance_is_symmetric_positive_semidefinite(stats):
    _, cov_matrix = stats
    values = cov_matrix.values
    assert np.allclose(values, values.T)
    eigenvalues = np.linalg.eigvalsh(values)
    assert (eigenvalues >= -1e-8).all()


def test_optimize_max_sharpe_weights_are_valid_long_only_portfolio(stats):
    mean_returns, cov_matrix = stats
    weights = po.optimize_max_sharpe(mean_returns, cov_matrix)
    assert weights.shape == (len(mean_returns),)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert (weights >= -1e-8).all()
    assert (weights <= 1 + 1e-8).all()


def test_optimize_min_volatility_has_lower_or_equal_vol_than_max_sharpe(stats):
    mean_returns, cov_matrix = stats
    w_sharpe = po.optimize_max_sharpe(mean_returns, cov_matrix)
    w_minvol = po.optimize_min_volatility(mean_returns, cov_matrix)

    _, vol_sharpe, _ = po.portfolio_performance(w_sharpe, mean_returns, cov_matrix)
    _, vol_minvol, _ = po.portfolio_performance(w_minvol, mean_returns, cov_matrix)

    # By construction, min-volatility must have volatility <= any other
    # feasible long-only portfolio, including the max-Sharpe one.
    assert vol_minvol <= vol_sharpe + 1e-6


def test_portfolio_performance_matches_manual_formula(stats):
    mean_returns, cov_matrix = stats
    n = len(mean_returns)
    weights = np.repeat(1.0 / n, n)

    port_return, port_vol, sharpe = po.portfolio_performance(weights, mean_returns, cov_matrix)

    expected_return = float(np.dot(weights, mean_returns))
    expected_vol = float(np.sqrt(weights @ cov_matrix.values @ weights))
    expected_sharpe = (expected_return - po.RISK_FREE_RATE) / expected_vol

    assert port_return == pytest.approx(expected_return)
    assert port_vol == pytest.approx(expected_vol)
    assert sharpe == pytest.approx(expected_sharpe)


def test_efficient_frontier_is_monotonic_in_return_and_bounded_below_by_minvol(stats):
    mean_returns, cov_matrix = stats
    w_minvol = po.optimize_min_volatility(mean_returns, cov_matrix)
    _, vol_minvol, _ = po.portfolio_performance(w_minvol, mean_returns, cov_matrix)

    frontier = po.efficient_frontier(mean_returns, cov_matrix, n_points=15)

    assert len(frontier) > 0
    assert frontier["return"].is_monotonic_increasing
    # Every point on the frontier must be at least as risky as the global
    # minimum-volatility portfolio (that's what makes it a *frontier*).
    assert (frontier["volatility"] >= vol_minvol - 1e-6).all()


def test_random_portfolios_weights_sum_to_one_and_are_long_only(stats):
    mean_returns, cov_matrix = stats
    random_ports = po.random_portfolios(mean_returns, cov_matrix, n_portfolios=200, seed=1)
    assert len(random_ports) == 200
    assert {"return", "volatility", "sharpe"}.issubset(random_ports.columns)
    assert (random_ports["volatility"] > 0).all()


def test_save_run_and_load_latest_performance_round_trip(tmp_path, stats):
    mean_returns, cov_matrix = stats
    w_sharpe = po.optimize_max_sharpe(mean_returns, cov_matrix)
    w_minvol = po.optimize_min_volatility(mean_returns, cov_matrix)
    perf_sharpe = po.portfolio_performance(w_sharpe, mean_returns, cov_matrix)
    perf_minvol = po.portfolio_performance(w_minvol, mean_returns, cov_matrix)
    frontier = po.efficient_frontier(mean_returns, cov_matrix, n_points=10)

    db_path = tmp_path / "test_results.duckdb"
    rs.save_run(
        db_path,
        run_label="unit-test-run",
        mean_returns=mean_returns,
        w_sharpe=w_sharpe,
        w_minvol=w_minvol,
        perf_sharpe=perf_sharpe,
        perf_minvol=perf_minvol,
        frontier=frontier,
    )

    result = rs.load_latest_performance(db_path)
    assert set(result["portfolio"]) == {"max_sharpe", "min_volatility"}
    assert result.loc[result["portfolio"] == "max_sharpe", "sharpe"].iloc[0] == pytest.approx(
        perf_sharpe[2]
    )
