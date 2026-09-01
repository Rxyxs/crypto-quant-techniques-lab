"""Persistencia de resultados de backtest en DuckDB.

`compare_strategies.py` ya escribe la tabla comparativa a CSV/JSON en cada
corrida, pero no queda un historial consultable entre corridas -- no hay
forma de responder "cuando corri esto por ultima vez, con que fuente de
datos, y como cambio el Sharpe neto de LightGBM desde entonces" sin abrir
manualmente los JSON uno por uno. Este modulo agrega una capa liviana de
persistencia analitica (un solo archivo DuckDB embebido, sin servidor) que
registra cada corrida del benchmark comparativo, para que ese historial sea
consultable con SQL.

No reemplaza los artefactos CSV/JSON existentes -- los complementa. Uso:

    python db_persistence.py            # corre el benchmark y lo persiste
    python db_persistence.py --report   # solo imprime el historial ya guardado
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path("outputs/backtest_history.duckdb")

RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       BIGINT,
    run_ts       TIMESTAMP,
    data_source  VARCHAR,
    n_test_days  INTEGER,
    best_strategy VARCHAR
)
"""

RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_results (
    run_id             BIGINT,
    strategy           VARCHAR,
    sharpe_gross       DOUBLE,
    sharpe_net         DOUBLE,
    max_drawdown_gross DOUBLE,
    max_drawdown_net   DOUBLE,
    total_return_gross DOUBLE,
    total_return_net   DOUBLE,
    win_rate_net       DOUBLE,
    signal_accuracy    DOUBLE,
    trades_taken       BIGINT,
    friction_drag_pct  DOUBLE
)
"""


def _connect(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(RUNS_SCHEMA)
    con.execute(RESULTS_SCHEMA)
    return con


def _next_run_id(con: duckdb.DuckDBPyConnection) -> int:
    result = con.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM runs").fetchone()
    return int(result[0])


def persist_comparison_run(
    table: pd.DataFrame,
    data_source: str,
    n_test_days: int,
    db_path: Path = DB_PATH,
) -> int:
    """Guarda una corrida de `compare_strategies.run_comparison()` en DuckDB.
    Cada llamada agrega una fila nueva a `runs` y una fila por estrategia a
    `strategy_results`, ligadas por `run_id` -- corridas anteriores nunca se
    sobrescriben, así que el historial completo queda disponible para
    análisis SQL (p. ej. cómo cambió el Sharpe neto de LightGBM entre
    corridas hechas en distintas ventanas de datos)."""
    con = _connect(db_path)
    try:
        run_id = _next_run_id(con)
        best_strategy = table.sort_values("sharpe_net", ascending=False).iloc[0]["strategy"]
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            [run_id, datetime.now(timezone.utc), data_source, int(n_test_days), best_strategy],
        )
        rows = table.copy()
        rows.insert(0, "run_id", run_id)
        con.execute(
            f"INSERT INTO strategy_results SELECT * FROM rows"
        )
        return run_id
    finally:
        con.close()


def load_run_history(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Historial completo, una fila por (corrida, estrategia), ordenado por
    corrida mas reciente primero."""
    con = _connect(db_path)
    try:
        return con.execute(
            """
            SELECT r.run_id, r.run_ts, r.data_source, r.n_test_days, r.best_strategy,
                   s.strategy, s.sharpe_gross, s.sharpe_net, s.max_drawdown_net,
                   s.total_return_net, s.friction_drag_pct
            FROM runs r
            JOIN strategy_results s USING (run_id)
            ORDER BY r.run_id DESC, s.sharpe_net DESC
            """
        ).df()
    finally:
        con.close()


def best_strategy_by_run(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Una fila por corrida: la estrategia ganadora (mayor Sharpe neto) y su
    metrica, util para ver si el ranking de estrategias es estable entre
    corridas o cambia segun la ventana de datos."""
    con = _connect(db_path)
    try:
        return con.execute(
            """
            SELECT run_id, run_ts, data_source, best_strategy, MAX(sharpe_net) AS best_sharpe_net
            FROM runs JOIN strategy_results USING (run_id)
            GROUP BY run_id, run_ts, data_source, best_strategy
            ORDER BY run_id DESC
            """
        ).df()
    finally:
        con.close()


def main() -> None:
    if "--report" in sys.argv:
        history = load_run_history()
        if history.empty:
            print("Sin corridas registradas todavia. Ejecuta 'python db_persistence.py' primero.")
            return
        print(history.to_string(index=False))
        return

    from compare_strategies import run_comparison

    print("Ejecutando benchmark comparativo para persistir en DuckDB...")
    table, _backtests, is_real, n_test = run_comparison()
    source = "real (Binance Vision)" if is_real else "SYNTHETIC (fallback)"

    run_id = persist_comparison_run(table, source, n_test)
    print(f"Corrida #{run_id} persistida en {DB_PATH} ({len(table)} estrategias).")
    print("\nHistorial completo:")
    print(load_run_history().to_string(index=False))


if __name__ == "__main__":
    main()
