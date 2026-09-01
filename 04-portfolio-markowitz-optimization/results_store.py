"""Persistence layer for optimization results.

This project is a numerical-optimization problem (Markowitz mean-variance,
solved via SciPy SLSQP), not a supervised-learning one, so there is no
"model" to persist -- instead this module persists the *results* of each
optimization run (max-Sharpe weights, min-volatility weights, and the full
efficient frontier) to a local DuckDB file, so runs are queryable and
comparable across time without re-running the optimizer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path("outputs/portfolio_results.duckdb")


def save_run(
    db_path: str | Path,
    *,
    run_label: str,
    mean_returns: pd.Series,
    w_sharpe,
    w_minvol,
    perf_sharpe: tuple[float, float, float],
    perf_minvol: tuple[float, float, float],
    frontier: pd.DataFrame,
) -> None:
    """Persists one optimization run (allocations + performance + frontier)
    into DuckDB tables, creating them on first use.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()

    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS allocations (
                run_ts VARCHAR, run_label VARCHAR, portfolio VARCHAR,
                asset VARCHAR, weight DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS performance (
                run_ts VARCHAR, run_label VARCHAR, portfolio VARCHAR,
                expected_return DOUBLE, volatility DOUBLE, sharpe DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS efficient_frontier (
                run_ts VARCHAR, run_label VARCHAR, point_idx INTEGER,
                return DOUBLE, volatility DOUBLE
            )
            """
        )

        assets = list(mean_returns.index)
        alloc_rows = (
            [(run_ts, run_label, "max_sharpe", a, float(w)) for a, w in zip(assets, w_sharpe)]
            + [(run_ts, run_label, "min_volatility", a, float(w)) for a, w in zip(assets, w_minvol)]
        )
        con.executemany(
            "INSERT INTO allocations VALUES (?, ?, ?, ?, ?)", alloc_rows
        )

        perf_rows = [
            (run_ts, run_label, "max_sharpe", *perf_sharpe),
            (run_ts, run_label, "min_volatility", *perf_minvol),
        ]
        con.executemany(
            "INSERT INTO performance VALUES (?, ?, ?, ?, ?, ?)", perf_rows
        )

        frontier_rows = [
            (run_ts, run_label, i, float(row["return"]), float(row["volatility"]))
            for i, row in frontier.reset_index(drop=True).iterrows()
        ]
        con.executemany(
            "INSERT INTO efficient_frontier VALUES (?, ?, ?, ?, ?)", frontier_rows
        )


def load_latest_performance(db_path: str | Path) -> pd.DataFrame:
    """Returns the performance table for the most recent run in the DB."""
    db_path = Path(db_path)
    with duckdb.connect(str(db_path)) as con:
        return con.execute(
            """
            SELECT * FROM performance
            WHERE run_ts = (SELECT max(run_ts) FROM performance)
            ORDER BY portfolio
            """
        ).df()
