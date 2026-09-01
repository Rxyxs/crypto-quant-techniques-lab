"""Persistencia de resultados de backtest y screening de cointegracion en
DuckDB, como complemento aditivo a los reportes CSV/JSON que ya genera
`pairs_trading_engine.run_full_pipeline` en `outputs/reports/`.

No reemplaza esos artefactos -- los mismos `metrics` (dict) y la misma tabla
de `screen_pairs_for_cointegration` que ya se escriben a JSON/CSV se insertan
ademas en un archivo DuckDB (`outputs/reports/pairs_trading.duckdb`), para
poder consultar el historial de corridas con SQL (por ejemplo, comparar
Sharpe entre corridas con distintos parametros de entrada/salida) sin
depender de parsear JSON a mano.

Uso:
    from db_persistence import persist_metrics, persist_screening

    persist_metrics(metrics)                 # dict, igual al que se guarda en metrics.json
    persist_screening(screening_df, run_id)  # DataFrame de screen_pairs_for_cointegration
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "outputs" / "reports"
DB_PATH = REPORTS_DIR / "pairs_trading.duckdb"

_METRICS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id VARCHAR PRIMARY KEY,
    run_ts TIMESTAMP,
    pair VARCHAR,
    n_days INTEGER,
    date_start VARCHAR,
    date_end VARCHAR,
    engle_granger_pvalue DOUBLE,
    cointegrated_5pct BOOLEAN,
    hedge_ratio_beta DOUBLE,
    hedge_ratio_alpha DOUBLE,
    hedge_ratio_r_squared DOUBLE,
    adf_pvalue DOUBLE,
    spread_stationary_5pct BOOLEAN,
    total_return DOUBLE,
    total_return_gross DOUBLE,
    total_transaction_cost DOUBLE,
    annualized_sharpe DOUBLE,
    max_drawdown DOUBLE,
    n_trades INTEGER,
    win_rate DOUBLE
)
"""

_SCREENING_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cointegration_screening_runs (
    run_id VARCHAR,
    run_ts TIMESTAMP,
    pair VARCHAR,
    y VARCHAR,
    x VARCHAR,
    eg_statistic DOUBLE,
    eg_pvalue DOUBLE,
    cointegrated_5pct BOOLEAN
)
"""


def _connect() -> duckdb.DuckDBPyConnection:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(_METRICS_TABLE_SCHEMA)
    con.execute(_SCREENING_TABLE_SCHEMA)
    return con


def persist_metrics(metrics: dict, run_id: str | None = None) -> str:
    """Inserta una fila en `backtest_runs` a partir del mismo dict `metrics`
    que `run_full_pipeline` ya escribe en `metrics.json`. Devuelve el
    `run_id` generado (o el provisto) para poder correlacionar con
    `persist_screening` de la misma corrida."""
    run_id = run_id or str(uuid.uuid4())
    con = _connect()
    try:
        date_start, date_end = metrics["date_range"]
        con.execute(
            """
            INSERT OR REPLACE INTO backtest_runs VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                datetime.now(timezone.utc),
                metrics["pair"],
                metrics["n_days"],
                date_start,
                date_end,
                metrics["engle_granger_pvalue"],
                metrics["cointegrated_5pct"],
                metrics["hedge_ratio_beta"],
                metrics["hedge_ratio_alpha"],
                metrics["hedge_ratio_r_squared"],
                metrics["adf_pvalue"],
                metrics["spread_stationary_5pct"],
                metrics["total_return"],
                metrics["total_return_gross"],
                metrics["total_transaction_cost"],
                metrics["annualized_sharpe"],
                metrics["max_drawdown"],
                metrics["n_trades"],
                metrics["win_rate"],
            ],
        )
    finally:
        con.close()
    return run_id


def persist_screening(screening: pd.DataFrame, run_id: str | None = None) -> str:
    """Inserta todas las filas de la tabla de screening (una por par
    candidato) de una corrida, etiquetadas con el mismo `run_id` que
    `persist_metrics` para poder hacer JOIN entre ambas tablas."""
    run_id = run_id or str(uuid.uuid4())
    con = _connect()
    try:
        run_ts = datetime.now(timezone.utc)
        rows = [
            (
                run_id,
                run_ts,
                row.pair,
                row.y,
                row.x,
                float(row.eg_statistic),
                float(row.eg_pvalue),
                bool(row.cointegrated_5pct),
            )
            for row in screening.itertuples()
        ]
        con.executemany(
            "INSERT INTO cointegration_screening_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()
    return run_id


def load_backtest_runs() -> pd.DataFrame:
    """Devuelve el historial completo de `backtest_runs` como DataFrame,
    ordenado por fecha de corrida descendente."""
    con = _connect()
    try:
        return con.execute("SELECT * FROM backtest_runs ORDER BY run_ts DESC").fetchdf()
    finally:
        con.close()


def load_screening_runs(run_id: str | None = None) -> pd.DataFrame:
    """Devuelve el historial de `cointegration_screening_runs`, opcionalmente
    filtrado a un `run_id` especifico."""
    con = _connect()
    try:
        if run_id is not None:
            return con.execute(
                "SELECT * FROM cointegration_screening_runs WHERE run_id = ? ORDER BY eg_pvalue",
                [run_id],
            ).fetchdf()
        return con.execute(
            "SELECT * FROM cointegration_screening_runs ORDER BY run_ts DESC, eg_pvalue"
        ).fetchdf()
    finally:
        con.close()


if __name__ == "__main__":
    # Backfill: si ya existen metrics.json y cointegration_screening.csv de
    # una corrida previa de `pairs_trading_engine.py`, los carga a DuckDB sin
    # necesidad de re-correr el pipeline completo (que requiere red).
    metrics_path = REPORTS_DIR / "metrics.json"
    screening_path = REPORTS_DIR / "cointegration_screening.csv"

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        run_id = persist_metrics(metrics)
        print(f"backtest_runs: 1 fila insertada (run_id={run_id})")

        if screening_path.exists():
            screening = pd.read_csv(screening_path)
            persist_screening(screening, run_id=run_id)
            print(f"cointegration_screening_runs: {len(screening)} filas insertadas (run_id={run_id})")
    else:
        print(f"No se encontro {metrics_path} -- corre pairs_trading_engine.py primero.")

    print(f"\nBase de datos: {DB_PATH}")
    print(load_backtest_runs())
