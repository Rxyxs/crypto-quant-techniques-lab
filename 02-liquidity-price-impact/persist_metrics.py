"""Shared DuckDB persistence for all three price-impact models (linear
baseline, XGBoost, PyTorch MLP) so their metrics live in one place for
comparison instead of three separate JSON files.

Table `model_metrics` is upserted per model (old rows for that model are
replaced) so re-running any script keeps the table consistent with the
latest run.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

DB_PATH = Path("outputs/reports/metrics.duckdb")

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_metrics (
    model VARCHAR,
    horizon_minutes INTEGER,
    rmse DOUBLE,
    mae DOUBLE,
    r2 DOUBLE,
    baseline_rmse DOUBLE,
    rmse_reduction_pct DOUBLE,
    n_test INTEGER
)
"""


def persist_metrics(model_name: str, horizon_results: dict[int, dict], db_path: Path = DB_PATH) -> None:
    """Writes one row per horizon for `model_name` into `model_metrics`,
    replacing any existing rows for that model."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(SCHEMA)
        con.execute("DELETE FROM model_metrics WHERE model = ?", [model_name])
        for horizon, metrics in horizon_results.items():
            rmse = metrics["rmse"]
            baseline_rmse = metrics["baseline_rmse"]
            reduction = 100.0 * (baseline_rmse - rmse) / baseline_rmse if baseline_rmse else 0.0
            con.execute(
                "INSERT INTO model_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [model_name, int(horizon), rmse, metrics["mae"], metrics["r2"],
                 baseline_rmse, reduction, metrics["n_test"]],
            )
    finally:
        con.close()


def read_all_metrics(db_path: Path = DB_PATH):
    """Returns the full `model_metrics` table as a list of dict rows."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute(SCHEMA)
        rows = con.execute(
            "SELECT * FROM model_metrics ORDER BY horizon_minutes, model"
        ).fetchdf()
    finally:
        con.close()
    return rows
