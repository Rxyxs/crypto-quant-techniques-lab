import numpy as np
import polars as pl

from alert_budget import (
    calibrate_alert_threshold,
    evaluate_at_budget,
    precision_at_budget_sweep,
    run_streaming_simulation,
)
from detect_spoofing import L2_FEATURE_COLS, engineer_features, fit_isolation_forest, simulate_orderbook


def _scored_df(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    df = engineer_features(simulate_orderbook(n, rng))
    X = df.select(L2_FEATURE_COLS).to_numpy()
    _, _, _, score = fit_isolation_forest(X)
    return df, score


def test_calibrate_alert_threshold_flags_exactly_the_budget():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=10_000)
    threshold = calibrate_alert_threshold(scores, budget_fraction=0.02)
    n_flagged = int((scores >= threshold).sum())
    assert abs(n_flagged - 200) <= 2  # small rounding tolerance at the exact quantile


def test_tighter_budget_never_decreases_precision_on_average():
    """Tighter alert budgets should trade recall for precision -- checked
    directly on real scored data, not assumed from theory."""
    df, score = _scored_df()
    sweep = precision_at_budget_sweep(df["true_is_spoof"].to_numpy(), score, budgets=[0.005, 0.02, 0.05])
    precisions = sweep.sort("budget_pct")["precision"].to_list()
    recalls = sweep.sort("budget_pct")["recall"].to_list()
    assert precisions[0] >= precisions[-1]
    assert recalls[0] <= recalls[-1]


def test_evaluate_at_budget_flags_match_n_flagged_field():
    df, score = _scored_df()
    metrics = evaluate_at_budget(df["true_is_spoof"].to_numpy(), score, budget_fraction=0.02)
    expected_n = round(0.02 * len(score))
    assert abs(metrics["n_flagged"] - expected_n) <= 2


def test_streaming_simulation_has_no_lookahead():
    """Corrupting snapshots strictly after a given point must never change
    the streaming score assigned to snapshots at or before that point."""
    df, _ = _scored_df(n=2500, seed=21)
    streamed_a = run_streaming_simulation(df, L2_FEATURE_COLS, initial_window=800, refit_interval=200)

    cutoff = 1500
    corrupted_top_size = df["top_size"].to_numpy().copy()
    corrupted_top_size[cutoff:] *= 50.0
    df_b = df.with_columns(pl.Series("top_size", corrupted_top_size))
    df_b = df_b.with_columns(
        (pl.col("top_size") / pl.col("rolling_avg_size")).alias("top_level_size_ratio")
    )
    streamed_b = run_streaming_simulation(df_b, L2_FEATURE_COLS, initial_window=800, refit_interval=200)

    before = streamed_a["streaming_score"][:cutoff].to_numpy()
    before_b = streamed_b["streaming_score"][:cutoff].to_numpy()
    np.testing.assert_allclose(before, before_b, equal_nan=True)


def test_streaming_simulation_warmup_snapshots_are_unscored():
    df, _ = _scored_df(n=1500, seed=31)
    streamed = run_streaming_simulation(df, L2_FEATURE_COLS, initial_window=900, refit_interval=200)
    warmup_scores = streamed["streaming_score"][:900].to_numpy()
    assert np.isnan(warmup_scores).all()
    post_warmup_scores = streamed["streaming_score"][900:].to_numpy()
    assert not np.isnan(post_warmup_scores).any()
