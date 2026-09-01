"""Tests unitarios para db_persistence.py, usando un archivo DuckDB temporal
(nunca el `outputs/reports/pairs_trading.duckdb` real del repo)."""

from __future__ import annotations

import pandas as pd
import pytest

import db_persistence


SAMPLE_METRICS = {
    "pair": "BTCUSDT/ETHUSDT",
    "n_days": 100,
    "date_range": ["2023-01-01", "2023-04-10"],
    "engle_granger_pvalue": 0.01,
    "cointegrated_5pct": True,
    "hedge_ratio_beta": 0.55,
    "hedge_ratio_alpha": 1.2,
    "hedge_ratio_r_squared": 0.6,
    "adf_pvalue": 0.02,
    "spread_stationary_5pct": True,
    "total_return": 0.1,
    "total_return_gross": 0.15,
    "total_transaction_cost": 0.05,
    "annualized_sharpe": 1.1,
    "max_drawdown": -0.2,
    "n_trades": 5,
    "win_rate": 0.6,
}

SAMPLE_SCREENING = pd.DataFrame(
    [
        {"pair": "BTC/ETH", "y": "BTCUSDT", "x": "ETHUSDT", "eg_statistic": -4.0, "eg_pvalue": 0.01, "cointegrated_5pct": True},
        {"pair": "BTC/SOL", "y": "BTCUSDT", "x": "SOLUSDT", "eg_statistic": -1.0, "eg_pvalue": 0.4, "cointegrated_5pct": False},
    ]
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_pairs_trading.duckdb"
    monkeypatch.setattr(db_persistence, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(db_persistence, "DB_PATH", db_path)
    return db_path


class TestPersistMetrics:
    def test_persist_and_load_roundtrip(self, temp_db):
        run_id = db_persistence.persist_metrics(SAMPLE_METRICS)
        df = db_persistence.load_backtest_runs()
        assert len(df) == 1
        assert df.iloc[0]["run_id"] == run_id
        assert df.iloc[0]["pair"] == "BTCUSDT/ETHUSDT"
        assert df.iloc[0]["annualized_sharpe"] == pytest.approx(1.1)

    def test_persist_twice_accumulates_rows(self, temp_db):
        db_persistence.persist_metrics(SAMPLE_METRICS)
        db_persistence.persist_metrics(SAMPLE_METRICS, run_id="second-run")
        df = db_persistence.load_backtest_runs()
        assert len(df) == 2


class TestPersistScreening:
    def test_persist_and_load_screening(self, temp_db):
        run_id = db_persistence.persist_screening(SAMPLE_SCREENING)
        df = db_persistence.load_screening_runs(run_id=run_id)
        assert len(df) == 2
        assert set(df["pair"]) == {"BTC/ETH", "BTC/SOL"}
        # ordenado por eg_pvalue ascendente
        assert df.iloc[0]["pair"] == "BTC/ETH"

    def test_load_screening_without_run_id_returns_all(self, temp_db):
        db_persistence.persist_screening(SAMPLE_SCREENING, run_id="run-a")
        db_persistence.persist_screening(SAMPLE_SCREENING, run_id="run-b")
        df = db_persistence.load_screening_runs()
        assert len(df) == 4
