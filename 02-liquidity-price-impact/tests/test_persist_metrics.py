from __future__ import annotations

from pathlib import Path

from persist_metrics import persist_metrics, read_all_metrics


def test_persist_and_read_metrics(tmp_path: Path):
    db_path = tmp_path / "metrics.duckdb"
    horizon_results = {
        1: {"rmse": 0.001, "mae": 0.0008, "r2": 0.1, "baseline_rmse": 0.0011, "n_test": 100},
        5: {"rmse": 0.002, "mae": 0.0015, "r2": -0.01, "baseline_rmse": 0.002, "n_test": 100},
    }
    persist_metrics("unit_test_model", horizon_results, db_path=db_path)

    df = read_all_metrics(db_path=db_path)
    rows = df[df["model"] == "unit_test_model"]
    assert len(rows) == 2
    assert set(rows["horizon_minutes"]) == {1, 5}


def test_persist_metrics_replaces_previous_rows_for_same_model(tmp_path: Path):
    db_path = tmp_path / "metrics.duckdb"
    first = {1: {"rmse": 0.001, "mae": 0.0008, "r2": 0.1, "baseline_rmse": 0.0011, "n_test": 100}}
    second = {1: {"rmse": 0.0005, "mae": 0.0004, "r2": 0.2, "baseline_rmse": 0.0011, "n_test": 100}}

    persist_metrics("m", first, db_path=db_path)
    persist_metrics("m", second, db_path=db_path)

    df = read_all_metrics(db_path=db_path)
    rows = df[df["model"] == "m"]
    assert len(rows) == 1
    assert abs(rows.iloc[0]["rmse"] - 0.0005) < 1e-9
