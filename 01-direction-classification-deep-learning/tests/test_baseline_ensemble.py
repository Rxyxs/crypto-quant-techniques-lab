"""
Tests unitarios para el modulo de baseline interpretable + ensamble de
arboles (train_baseline_ensemble.py) y para la reutilizacion de la
ingenieria de features de train_classifier.py.

Uso:
    .\\venv\\Scripts\\python.exe -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_classifier import FEATURE_COLUMNS, build_features_and_label, simulate_ohlcv_data
from train_baseline_ensemble import chronological_split, evaluate, persist_to_duckdb


def _small_dataset(n_candles: int = 500, seed: int = 7):
    raw = simulate_ohlcv_data(n_candles, seed)
    return build_features_and_label(raw)


def test_build_features_and_label_no_leakage_columns():
    df = _small_dataset()
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert "label_next_bullish" in df.columns
    # la etiqueta debe ser binaria (0/1)
    labels = set(df["label_next_bullish"].unique().to_list())
    assert labels <= {0, 1}


def test_build_features_and_label_drops_nulls():
    df = _small_dataset()
    assert df.null_count().sum_horizontal().sum() == 0


def test_chronological_split_is_ordered_and_disjoint():
    df = _small_dataset()
    train_df, val_df, test_df = chronological_split(df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    # el split es cronologico: el ultimo timestamp de train <= primero de val <= primero de test
    assert train_df["timestamp"][-1] <= val_df["timestamp"][0]
    assert val_df["timestamp"][-1] <= test_df["timestamp"][0]


def test_evaluate_returns_expected_keys_and_bounds():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200).astype(np.int8)
    y_prob = rng.uniform(0, 1, size=200)

    metrics = evaluate("dummy_model", y_true, y_prob)

    expected_keys = {
        "model",
        "accuracy",
        "roc_auc",
        "precision_bajista",
        "recall_bajista",
        "f1_bajista",
        "precision_alcista",
        "recall_alcista",
        "f1_alcista",
        "n_test",
    }
    assert expected_keys <= set(metrics.keys())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["n_test"] == 200


def test_evaluate_perfect_predictions_score_one():
    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
    y_prob = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

    metrics = evaluate("perfect_model", y_true, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_persist_to_duckdb_roundtrip(tmp_path, monkeypatch):
    import train_baseline_ensemble as mod

    db_path = tmp_path / "model_db_test.duckdb"
    monkeypatch.setattr(mod, "DB_PATH", db_path)
    monkeypatch.setattr(mod, "REPORTS_DIR", tmp_path)

    rows = [
        {
            "model": "unit_test_model",
            "accuracy": 0.55,
            "roc_auc": 0.6,
            "precision_bajista": 0.5,
            "recall_bajista": 0.5,
            "f1_bajista": 0.5,
            "precision_alcista": 0.6,
            "recall_alcista": 0.6,
            "f1_alcista": 0.6,
            "n_test": 100,
        }
    ]

    persist_to_duckdb(rows)

    import duckdb

    con = duckdb.connect(str(db_path))
    result = con.execute("SELECT model, accuracy, roc_auc FROM model_comparison").fetchall()
    con.close()

    assert result == [("unit_test_model", 0.55, 0.6)]
