import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_analysis import (
    REGIMES,
    TRANSITION,
    add_rolling_return_and_volatility,
    build_date_index,
    build_returns_frame,
    label_regimes,
    rolling_correlation_dispersion_drawdown,
    save_results_to_duckdb,
    select_k_by_silhouette,
    simulate_regime_path,
    simulate_returns,
)

ASSETS = ["A", "B", "C"]


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_simulate_regime_path_valid_states(rng):
    states = simulate_regime_path(200, TRANSITION, start_state=0, rng=rng)
    assert states.shape == (200,)
    assert states[0] == 0
    assert set(np.unique(states)).issubset(set(range(TRANSITION.shape[0])))


def test_transition_matrix_rows_sum_to_one():
    assert np.allclose(TRANSITION.sum(axis=1), 1.0)


def test_simulate_returns_shape_and_finite(rng):
    states = simulate_regime_path(100, TRANSITION, start_state=0, rng=rng)
    returns = simulate_returns(states, REGIMES, n_assets=len(ASSETS), rng=rng)
    assert returns.shape == (100, len(ASSETS))
    assert np.all(np.isfinite(returns))


def test_build_returns_frame_columns(rng):
    n = 30
    dates = build_date_index(n)
    states = simulate_regime_path(n, TRANSITION, start_state=0, rng=rng)
    returns = simulate_returns(states, REGIMES, len(ASSETS), rng)
    regime_names = [r["name"] for r in REGIMES]
    df = build_returns_frame(dates, returns, ASSETS, states, regime_names)
    assert df.height == n
    for col in ["date", *ASSETS, "true_regime"]:
        assert col in df.columns


def test_add_rolling_return_and_volatility_nan_warmup(rng):
    n = 40
    window = 10
    dates = build_date_index(n)
    states = simulate_regime_path(n, TRANSITION, start_state=0, rng=rng)
    returns = simulate_returns(states, REGIMES, len(ASSETS), rng)
    regime_names = [r["name"] for r in REGIMES]
    df = build_returns_frame(dates, returns, ASSETS, states, regime_names)
    df = add_rolling_return_and_volatility(df, ASSETS, window)
    assert df["avg_return"][: window - 1].is_null().all()
    assert df["avg_return"][window - 1 :].is_null().sum() == 0


def test_rolling_correlation_dispersion_drawdown_shapes_and_bounds(rng):
    n, window = 60, 15
    returns = rng.standard_normal((n, 4)) * 0.02
    avg_corr, dispersion, drawdown = rolling_correlation_dispersion_drawdown(returns, window)
    assert avg_corr.shape == (n,) == dispersion.shape == drawdown.shape
    assert np.isnan(avg_corr[: window - 1]).all()
    valid_corr = avg_corr[window - 1 :]
    assert np.all(valid_corr >= -1.0001) and np.all(valid_corr <= 1.0001)
    assert np.all(dispersion[window - 1 :] >= 0)
    assert np.all(drawdown[window - 1 :] <= 1e-9)


def test_rolling_correlation_perfectly_correlated_assets():
    # If every asset moves identically, average pairwise correlation must be ~1.
    n, window = 30, 10
    base = np.random.default_rng(1).standard_normal(n)
    returns = np.column_stack([base, base, base])
    avg_corr, _, _ = rolling_correlation_dispersion_drawdown(returns, window)
    assert np.allclose(avg_corr[window - 1 :], 1.0, atol=1e-6)


def test_label_regimes_assigns_bear_crash_to_low_return_high_vol():
    centroids = np.array(
        [
            [0.002, 0.01],   # Bull Quiet
            [-0.004, 0.06],  # Bear Crash
        ]
    )
    labels = label_regimes(centroids)
    assert labels[1] == "Bear Crash"
    assert labels[0] == "Bull Quiet"


def test_label_regimes_disambiguates_duplicate_labels():
    centroids = np.array(
        [
            [0.003, 0.005],
            [0.002, 0.001],
            [-0.003, 0.05],
            [-0.001, 0.001],
        ]
    )
    labels = label_regimes(centroids)
    assert len(labels) == len(set(labels))


def test_select_k_by_silhouette_returns_valid_k():
    rng_ = np.random.default_rng(2)
    cluster_a = rng_.normal(loc=0, scale=0.2, size=(50, 3))
    cluster_b = rng_.normal(loc=5, scale=0.2, size=(50, 3))
    X = np.vstack([cluster_a, cluster_b])
    best_k, scores = select_k_by_silhouette(X, range(2, 4))
    assert best_k in scores
    assert best_k == 2  # two well-separated blobs should be picked cleanly


def test_save_results_to_duckdb_roundtrip(tmp_path):
    regime_summary = pl.DataFrame(
        {"cluster": [0, 1], "n_days": [10, 20], "regime_label": ["Bull Quiet", "Bear Crash"]}
    )
    result_df = pl.DataFrame({"date": build_date_index(5), "cluster": [0, 0, 1, 1, 1]})

    db_path = tmp_path / "test_regimes.duckdb"
    save_results_to_duckdb(regime_summary, result_df, db_path)

    con = duckdb.connect(str(db_path))
    summary_rows = con.execute("SELECT cluster, n_days, regime_label FROM regime_summary ORDER BY cluster").fetchall()
    days_count = con.execute("SELECT COUNT(*) FROM regime_days").fetchone()[0]
    con.close()

    assert summary_rows == [(0, 10, "Bull Quiet"), (1, 20, "Bear Crash")]
    assert days_count == 5
