"""Persists comparative metrics across all three detection approaches
(z-score baseline, Isolation Forest, Autoencoder) into a local DuckDB
database (`outputs/metrics.duckdb`), so results are queryable rather than
scattered across separate CSVs / notebook print statements.
"""

from pathlib import Path

import duckdb

from detect_spoofing import OUTPUT_DIR

DB_PATH = OUTPUT_DIR / "metrics.duckdb"

CREATE_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS model_metrics (
    model VARCHAR,
    variant VARCHAR,
    precision DOUBLE,
    recall DOUBLE,
    f1 DOUBLE,
    n_flagged INTEGER,
    run_seed INTEGER
)
"""

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    model VARCHAR,
    snapshot_id INTEGER,
    anomaly_score DOUBLE,
    is_anomaly BOOLEAN,
    true_is_spoof BOOLEAN
)
"""


def get_connection(db_path=DB_PATH):
    Path(db_path).parent.mkdir(exist_ok=True, parents=True)
    con = duckdb.connect(str(db_path))
    con.execute(CREATE_METRICS_TABLE)
    con.execute(CREATE_PREDICTIONS_TABLE)
    return con


def record_metrics(con, model, variant, precision, recall, f1, n_flagged, seed):
    con.execute(
        "DELETE FROM model_metrics WHERE model = ? AND variant = ?", [model, variant]
    )
    con.execute(
        "INSERT INTO model_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
        [model, variant, precision, recall, f1, n_flagged, seed],
    )


def record_predictions(con, model, snapshot_ids, scores, is_anomaly, true_is_spoof):
    con.execute("DELETE FROM predictions WHERE model = ?", [model])
    con.executemany(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
        [
            (model, int(sid), float(score), bool(flag), bool(truth))
            for sid, score, flag, truth in zip(snapshot_ids, scores, is_anomaly, true_is_spoof)
        ],
    )


def comparison_table(con):
    return con.execute(
        "SELECT model, variant, precision, recall, f1, n_flagged FROM model_metrics ORDER BY f1 DESC"
    ).fetchdf()


def main():
    """Runs all three approaches end to end and persists their comparative
    metrics + per-snapshot predictions into DuckDB."""
    import numpy as np

    from autoencoder_spoofing import DEFAULT_ACTIVATION, evaluate_activation
    from detect_spoofing import (
        L2_FEATURE_COLS,
        N_SNAPSHOTS,
        SEED,
        engineer_features,
        evaluate,
        fit_isolation_forest,
        simulate_orderbook,
    )
    from zscore_baseline import evaluate_baseline

    rng = np.random.default_rng(SEED)
    df = engineer_features(simulate_orderbook(N_SNAPSHOTS, rng))
    y_true = df["true_is_spoof"].to_numpy()
    snapshot_ids = df["snapshot_id"].to_numpy()

    con = get_connection()

    # 1. Z-score baseline
    baseline = evaluate_baseline(df)
    record_metrics(con, "zscore_baseline", "order_book_imbalance", baseline["precision"], baseline["recall"], baseline["f1"], baseline["n_flagged"], SEED)
    record_predictions(con, "zscore_baseline", snapshot_ids, baseline["z_score"], baseline["is_anomaly"], y_true)

    # 2. Isolation Forest
    X = df.select(L2_FEATURE_COLS).to_numpy()
    _, _, if_is_anomaly, if_scores = fit_isolation_forest(X)
    if_metrics = evaluate(y_true, if_is_anomaly)
    record_metrics(con, "isolation_forest", "L2_aggregate_features", if_metrics["precision"], if_metrics["recall"], if_metrics["f1"], int(if_is_anomaly.sum()), SEED)
    record_predictions(con, "isolation_forest", snapshot_ids, if_scores, if_is_anomaly, y_true)

    # 3. Autoencoder (all three activations)
    for activation in ("ReLU", "GELU", "Swish"):
        result = evaluate_activation(df, activation)
        record_metrics(con, "autoencoder", activation, result["precision"], result["recall"], result["f1"], int(result["is_anomaly"].sum()), SEED)
        if activation == DEFAULT_ACTIVATION:
            record_predictions(con, "autoencoder", snapshot_ids, result["scores"], result["is_anomaly"], y_true)

    print(comparison_table(con))
    con.close()
    print(f"\nSaved comparative metrics to {DB_PATH}")


if __name__ == "__main__":
    main()
