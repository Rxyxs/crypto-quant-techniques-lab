"""Tests unitarios para kalman_pairs.py, sobre series sinteticas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalman_pairs import run_kalman_hedge_ratio


N = 300
RNG = np.random.default_rng(7)


def _cointegrated_pair_constant_beta(n: int = N, beta: float = 0.7, alpha: float = 0.5, noise_std: float = 0.03):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    x = pd.Series(np.cumsum(RNG.normal(0, 0.02, n)) + 4.0, index=idx, name="X")
    y = pd.Series(alpha + beta * x.to_numpy() + RNG.normal(0, noise_std, n), index=idx, name="Y")
    return y, x


class TestKalmanHedgeRatio:
    def test_output_shapes_and_index_alignment(self):
        y, x = _cointegrated_pair_constant_beta()
        result = run_kalman_hedge_ratio(y, x)
        for series in (result.alpha, result.beta, result.spread, result.innovation_variance, result.zscore):
            assert len(series) == len(y)
            assert (series.index == y.index).all()

    def test_beta_converges_near_true_value(self):
        y, x = _cointegrated_pair_constant_beta(beta=0.7, alpha=0.5)
        result = run_kalman_hedge_ratio(y, x)
        # tras el burn-in y suficientes pasos de filtrado, beta_t deberia
        # rondar el valor verdadero (constante en este dataset sintetico)
        tail_beta = result.beta.iloc[-50:].mean()
        assert tail_beta == pytest.approx(0.7, abs=0.2)

    def test_innovation_variance_is_positive(self):
        y, x = _cointegrated_pair_constant_beta()
        result = run_kalman_hedge_ratio(y, x)
        assert (result.innovation_variance > 0).all()

    def test_zscore_matches_spread_over_sqrt_variance(self):
        y, x = _cointegrated_pair_constant_beta()
        result = run_kalman_hedge_ratio(y, x)
        expected = result.spread / np.sqrt(result.innovation_variance)
        pd.testing.assert_series_equal(result.zscore, expected.rename("zscore"))

    def test_raises_on_mismatched_length(self):
        y, x = _cointegrated_pair_constant_beta()
        with pytest.raises(ValueError):
            run_kalman_hedge_ratio(y, x.iloc[:-10])

    def test_process_variance_q_shape(self):
        y, x = _cointegrated_pair_constant_beta()
        result = run_kalman_hedge_ratio(y, x, delta=1e-4)
        assert result.process_variance_q.shape == (2, 2)
        expected_q = 1e-4 / (1 - 1e-4)
        assert result.process_variance_q[0, 0] == pytest.approx(expected_q)
        assert result.process_variance_q[1, 1] == pytest.approx(expected_q)
