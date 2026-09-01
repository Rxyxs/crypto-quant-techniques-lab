"""Alert-budget threshold calibration and streaming (walk-forward)
Isolation Forest scoring.

A surveillance desk cannot review every flagged snapshot -- it has a fixed
daily *alert budget* (an analyst can only look at so many alerts). This
module replaces the single fixed `contamination` cutoff with an explicit
budget-driven threshold: pick the score percentile that flags exactly the
number of snapshots the budget allows, then measure precision/recall at
that budget. It also implements a genuine streaming simulation: the batch
pipeline in `detect_spoofing.py` fits one Isolation Forest on the entire
dataset at once, which is fine for offline validation but is not how a live
surveillance system runs -- this module scores each snapshot using only a
model fit on data strictly *before* it, refitting periodically on a
trailing rolling window, exactly as a production deployment would.
"""

from __future__ import annotations

import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import polars as pl

from detect_spoofing import (
    CONTAMINATION,
    L2_FEATURE_COLS,
    N_SNAPSHOTS,
    OUTPUT_DIR,
    SEED,
    engineer_features,
    evaluate,
    fit_isolation_forest,
    simulate_orderbook,
)

# Alert budgets expressed as a fraction of snapshots a surveillance desk
# could realistically review -- from a very tight 0.5% up to the original
# pipeline's 2% `contamination` assumption, for direct comparison.
ALERT_BUDGETS = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]

STREAMING_INITIAL_WINDOW = 1000
STREAMING_REFIT_INTERVAL = 250
STREAMING_ROLLING_WINDOW = 1500


def calibrate_alert_threshold(anomaly_score: np.ndarray, budget_fraction: float) -> float:
    """Returns the score threshold such that exactly `budget_fraction` of
    scored snapshots would be flagged -- the (1 - budget_fraction)
    percentile of the score distribution. This decouples the *alert
    policy* (how many alerts review capacity allows) from the *model fit*
    (which no longer needs its own `contamination` to determine the
    decision boundary)."""
    return float(np.quantile(anomaly_score, 1 - budget_fraction))


def evaluate_at_budget(true_is_spoof: np.ndarray, anomaly_score: np.ndarray, budget_fraction: float) -> dict:
    threshold = calibrate_alert_threshold(anomaly_score, budget_fraction)
    is_anomaly = anomaly_score >= threshold
    metrics = evaluate(true_is_spoof, is_anomaly)
    metrics.update(
        {
            "budget_fraction": budget_fraction,
            "threshold": threshold,
            "n_flagged": int(is_anomaly.sum()),
            "n_scored": int(len(true_is_spoof)),
        }
    )
    return metrics


def precision_at_budget_sweep(
    true_is_spoof: np.ndarray, anomaly_score: np.ndarray, budgets: list[float] = ALERT_BUDGETS
) -> pl.DataFrame:
    """The alert-budget precision matrix: one row per budget, with the
    resulting alert count, precision, recall, and F1 -- the direct
    "what do we get for reviewing N% of snapshots" table a surveillance
    desk actually needs, instead of a single fixed-contamination number."""
    rows = [evaluate_at_budget(true_is_spoof, anomaly_score, b) for b in budgets]
    return pl.DataFrame(
        [
            {
                "budget_pct": r["budget_fraction"] * 100,
                "n_flagged": r["n_flagged"],
                "threshold": r["threshold"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
            }
            for r in rows
        ]
    )


def run_streaming_simulation(
    df: pl.DataFrame,
    feature_cols: list[str],
    initial_window: int = STREAMING_INITIAL_WINDOW,
    refit_interval: int = STREAMING_REFIT_INTERVAL,
    rolling_window: int = STREAMING_ROLLING_WINDOW,
    contamination: float = CONTAMINATION,
    seed: int = SEED,
) -> pl.DataFrame:
    """Walk-forward Isolation Forest scoring: fits on a trailing rolling
    window and scores only the snapshots that come strictly after it, then
    refits on the next trailing window before scoring the next chunk --
    every score uses only data a live system would actually have had by
    that point in time. Snapshots inside `initial_window` (before any
    trailing baseline exists) get a null score and are excluded from
    evaluation, exactly as a real deployment would have no coverage during
    its own warm-up period.

    Returns `df` with `streaming_score` (null during warm-up) and
    `streaming_model_id` (which rolling refit produced that score, for
    diagnosing whether performance drifts across refits) appended.
    """
    n = df.height
    X_all = df.select(feature_cols).to_numpy()
    scores = np.full(n, np.nan)
    model_id = np.full(n, -1, dtype=int)

    train_end = initial_window
    current_model_no = 0
    while train_end < n:
        train_start = max(0, train_end - rolling_window)
        model, scaler, _, _ = fit_isolation_forest(
            X_all[train_start:train_end], contamination=contamination, seed=seed
        )

        chunk_end = min(n, train_end + refit_interval)
        X_chunk_scaled = scaler.transform(X_all[train_end:chunk_end])
        scores[train_end:chunk_end] = -model.score_samples(X_chunk_scaled)
        model_id[train_end:chunk_end] = current_model_no

        current_model_no += 1
        train_end = chunk_end

    return df.with_columns(
        [pl.Series("streaming_score", scores), pl.Series("streaming_model_id", model_id)]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = engineer_features(simulate_orderbook(N_SNAPSHOTS, rng))

    X = df.select(L2_FEATURE_COLS).to_numpy()
    _, _, _, batch_score = fit_isolation_forest(X)
    batch_sweep = precision_at_budget_sweep(df["true_is_spoof"].to_numpy(), batch_score)
    print("Batch alert-budget sweep:")
    print(batch_sweep)
    batch_sweep.write_csv(OUTPUT_DIR / "alert_budget_sweep_batch.csv")

    streamed = run_streaming_simulation(df, L2_FEATURE_COLS)
    scored_stream = streamed.filter(pl.col("streaming_score").is_not_nan())
    stream_sweep = precision_at_budget_sweep(
        scored_stream["true_is_spoof"].to_numpy(), scored_stream["streaming_score"].to_numpy()
    )
    print("\nStreaming (walk-forward) alert-budget sweep:")
    print(stream_sweep)
    stream_sweep.write_csv(OUTPUT_DIR / "alert_budget_sweep_streaming.csv")

    print(f"\nSaved sweep tables to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
