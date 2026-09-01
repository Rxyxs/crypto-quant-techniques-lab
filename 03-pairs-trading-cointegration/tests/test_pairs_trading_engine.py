"""Tests unitarios para pairs_trading_engine.py, sobre series sinteticas
(sin llamadas de red a la API de Binance -- fetch_symbol_closes/fetch_price_panel
quedan fuera de este alcance por eso)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_trading_engine import (
    adf_test,
    backtest_spread_strategy,
    compute_spread,
    compute_transaction_cost_returns,
    engle_granger_test,
    estimate_hedge_ratio,
    generate_signals,
    rolling_zscore,
    screen_pairs_for_cointegration,
)

N = 500
RNG = np.random.default_rng(42)


def _cointegrated_pair(n: int = N, beta: float = 0.5, alpha: float = 1.0, noise_std: float = 0.05):
    """Genera un par (y, x) sinteticos donde x es un random walk en
    log-precio y y = alpha + beta*x + ruido estacionario (mean-reverting) --
    por construccion, cointegrados."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    x = pd.Series(np.cumsum(RNG.normal(0, 0.02, n)) + 4.0, index=idx, name="X")

    # ruido AR(1) estacionario para el spread, en vez de ruido blanco puro,
    # asi el ADF test tiene una serie mean-reverting no trivial que detectar.
    eps = np.zeros(n)
    for t in range(1, n):
        eps[t] = 0.6 * eps[t - 1] + RNG.normal(0, noise_std)
    y = pd.Series(alpha + beta * x.to_numpy() + eps, index=idx, name="Y")
    return y, x


def _independent_walks(n: int = N):
    """Dos random walks independientes -- NO deberian cointegrar."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    y = pd.Series(np.cumsum(RNG.normal(0, 0.02, n)) + 4.0, index=idx, name="Y")
    x = pd.Series(np.cumsum(RNG.normal(0, 0.02, n)) + 4.0, index=idx, name="X")
    return y, x


class TestEngleGranger:
    def test_cointegrated_pair_detected(self):
        y, x = _cointegrated_pair()
        result = engle_granger_test(y, x)
        assert result.is_cointegrated_5pct
        assert result.p_value < 0.05
        assert set(result.critical_values) == {"1%", "5%", "10%"}

    def test_independent_walks_usually_not_cointegrated(self):
        y, x = _independent_walks()
        result = engle_granger_test(y, x)
        # no es un test determinista (es una simulacion), pero con estas
        # semillas/longitud el par independiente no deberia cointegrar al 5%
        assert result.p_value > 0.05
        assert not result.is_cointegrated_5pct


class TestHedgeRatioAndSpread:
    def test_estimate_hedge_ratio_recovers_true_beta(self):
        y, x = _cointegrated_pair(beta=0.5, alpha=1.0)
        hedge = estimate_hedge_ratio(y, x)
        assert hedge.beta == pytest.approx(0.5, abs=0.15)
        assert hedge.alpha == pytest.approx(1.0, abs=0.5)
        assert 0.0 <= hedge.r_squared <= 1.0

    def test_compute_spread_is_stationary_for_cointegrated_pair(self):
        y, x = _cointegrated_pair()
        hedge = estimate_hedge_ratio(y, x)
        spread = compute_spread(y, x, hedge)
        adf_result = adf_test(spread)
        assert adf_result.is_stationary_5pct

    def test_compute_spread_formula(self):
        idx = pd.date_range("2023-01-01", periods=5)
        y = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

        class _Hedge:
            alpha = 2.0
            beta = 3.0

        spread = compute_spread(y, x, _Hedge())
        expected = y - (2.0 + 3.0 * x)
        pd.testing.assert_series_equal(spread, expected.rename("spread"))


class TestSignalsAndZscore:
    def test_rolling_zscore_shape_and_nan_burnin(self):
        y, x = _cointegrated_pair()
        hedge = estimate_hedge_ratio(y, x)
        spread = compute_spread(y, x, hedge)
        z = rolling_zscore(spread, window=30)
        assert len(z) == len(spread)
        assert z.iloc[:29].isna().all()
        assert z.iloc[29:].notna().all()

    def test_generate_signals_enters_and_exits(self):
        # z-score sintetico: sube por encima de entry_z, se mantiene, cae
        # bajo exit_z -- se espera entrada corta y luego salida.
        z = pd.Series([0.0, 0.5, 2.5, 2.2, 1.8, 0.3, 0.1])
        positions = generate_signals(z, entry_z=2.0, exit_z=0.5)
        assert list(positions) == [0, 0, -1, -1, -1, 0, 0]

    def test_generate_signals_long_side(self):
        z = pd.Series([0.0, -2.5, -2.1, -0.4])
        positions = generate_signals(z, entry_z=2.0, exit_z=0.5)
        assert list(positions) == [0, 1, 1, 0]

    def test_generate_signals_handles_nan(self):
        z = pd.Series([np.nan, np.nan, 2.5, 0.1])
        positions = generate_signals(z, entry_z=2.0, exit_z=0.5)
        assert list(positions) == [0, 0, -1, 0]


class TestTransactionCosts:
    def test_zero_cost_when_flat(self):
        positions = pd.Series([0, 0, 0, 0])
        costs = compute_transaction_cost_returns(positions)
        assert (costs == 0).all()

    def test_entry_and_exit_charged_once_each(self):
        positions = pd.Series([0, 1, 1, 0])
        costs = compute_transaction_cost_returns(
            positions, taker_fee_bps=10.0, entry_slippage_bps=5.0, exit_slippage_bps=2.0
        )
        entry_rate = (10.0 + 5.0) / 10_000 * 2
        exit_rate = (10.0 + 2.0) / 10_000 * 2
        assert costs.iloc[0] == 0.0
        assert costs.iloc[1] == pytest.approx(entry_rate)
        assert costs.iloc[2] == 0.0
        assert costs.iloc[3] == pytest.approx(exit_rate)

    def test_flip_charged_entry_plus_exit(self):
        positions = pd.Series([1, 1, -1])
        costs = compute_transaction_cost_returns(
            positions, taker_fee_bps=10.0, entry_slippage_bps=5.0, exit_slippage_bps=2.0
        )
        entry_rate = (10.0 + 5.0) / 10_000 * 2
        exit_rate = (10.0 + 2.0) / 10_000 * 2
        assert costs.iloc[2] == pytest.approx(entry_rate + exit_rate)


class TestBacktest:
    def test_backtest_no_position_yields_zero_return(self):
        idx = pd.date_range("2023-01-01", periods=10)
        y = pd.Series(np.linspace(10, 11, 10), index=idx)
        x = pd.Series(np.linspace(5, 5.5, 10), index=idx)
        positions = pd.Series(0.0, index=idx)
        result = backtest_spread_strategy(y, x, beta=0.5, positions=positions)
        assert result.total_return == pytest.approx(0.0)
        assert result.n_trades == 0
        assert result.total_transaction_cost == pytest.approx(0.0)

    def test_backtest_costs_reduce_net_relative_to_gross(self):
        y, x = _cointegrated_pair()
        hedge = estimate_hedge_ratio(y, x)
        spread = compute_spread(y, x, hedge)
        z = rolling_zscore(spread)
        positions = generate_signals(z)
        result = backtest_spread_strategy(y, x, hedge.beta, positions, include_costs=True)
        assert result.total_transaction_cost >= 0.0
        # el equity neto nunca puede superar al bruto cuando hay trades con costo > 0
        if result.n_trades > 0:
            assert result.total_return <= result.total_return_gross + 1e-9

    def test_backtest_accepts_kalman_style_beta_series(self):
        idx = pd.date_range("2023-01-01", periods=20)
        y = pd.Series(np.linspace(10, 12, 20), index=idx)
        x = pd.Series(np.linspace(5, 6, 20), index=idx)
        beta_series = pd.Series(np.linspace(0.4, 0.6, 20), index=idx)
        positions = pd.Series(0.0, index=idx)
        positions.iloc[5:10] = 1.0
        result = backtest_spread_strategy(y, x, beta_series, positions)
        assert isinstance(result.total_return, float)
        assert len(result.equity_curve) == 20


class TestScreening:
    def test_screen_pairs_ranks_by_pvalue_and_flags_cointegrated(self):
        y, x = _cointegrated_pair()
        y2, x2 = _independent_walks()
        panel = pd.DataFrame(
            {
                "AAAUSDT": np.exp(y.to_numpy()),
                "BBBUSDT": np.exp(x.to_numpy()),
                "CCCUSDT": np.exp(y2.to_numpy()),
            },
            index=y.index,
        )
        screening = screen_pairs_for_cointegration(panel)
        assert len(screening) == 3  # C(3,2)
        assert list(screening["eg_pvalue"]) == sorted(screening["eg_pvalue"])
        best = screening.iloc[0]
        assert {best.y, best.x} == {"AAAUSDT", "BBBUSDT"}
        assert best.cointegrated_5pct
