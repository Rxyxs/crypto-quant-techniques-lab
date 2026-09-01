import pandas as pd
import pytest

from db_persistence import load_run_history, persist_comparison_run


@pytest.fixture
def sample_table():
    return pd.DataFrame([
        {
            "strategy": "SMA Crossover (10/50)", "sharpe_gross": 0.1, "sharpe_net": -0.2,
            "max_drawdown_gross": -0.3, "max_drawdown_net": -0.35, "total_return_gross": 0.05,
            "total_return_net": -0.02, "win_rate_net": 0.48, "signal_accuracy": 0.5,
            "trades_taken": 20, "friction_drag_pct": 1.5,
        },
        {
            "strategy": "LightGBM (Gradient Boosting)", "sharpe_gross": 0.4, "sharpe_net": -0.36,
            "max_drawdown_gross": -0.25, "max_drawdown_net": -0.4, "total_return_gross": 0.12,
            "total_return_net": -0.05, "win_rate_net": 0.5, "signal_accuracy": 0.507,
            "trades_taken": 80, "friction_drag_pct": 27.28,
        },
    ])


def test_persist_and_load_run_history_roundtrip(tmp_path, sample_table):
    db_path = tmp_path / "history.duckdb"
    run_id = persist_comparison_run(sample_table, "SYNTHETIC (fallback)", 100, db_path=db_path)
    assert run_id == 1

    history = load_run_history(db_path)
    assert len(history) == len(sample_table)
    assert set(history["run_id"]) == {1}
    assert set(history["strategy"]) == set(sample_table["strategy"])


def test_persist_comparison_run_increments_run_id_across_calls(tmp_path, sample_table):
    db_path = tmp_path / "history.duckdb"
    first_id = persist_comparison_run(sample_table, "SYNTHETIC (fallback)", 100, db_path=db_path)
    second_id = persist_comparison_run(sample_table, "real (Binance Vision)", 150, db_path=db_path)
    assert second_id == first_id + 1

    history = load_run_history(db_path)
    assert set(history["run_id"]) == {1, 2}
    assert len(history) == 2 * len(sample_table)
