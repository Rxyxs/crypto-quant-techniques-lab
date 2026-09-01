import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from metrics_store import comparison_table, get_connection, record_metrics, record_predictions


@pytest.fixture()
def temp_db(tmp_path):
    db_path = tmp_path / "test_metrics.duckdb"
    con = get_connection(db_path)
    yield con
    con.close()


def test_record_and_read_metrics(temp_db):
    record_metrics(temp_db, "model_a", "variant_1", 0.9, 0.8, 0.85, 10, 42)
    table = comparison_table(temp_db)
    assert len(table) == 1
    assert table.iloc[0]["model"] == "model_a"
    assert table.iloc[0]["f1"] == pytest.approx(0.85)


def test_record_metrics_upserts_by_model_variant(temp_db):
    record_metrics(temp_db, "model_a", "variant_1", 0.5, 0.5, 0.5, 5, 42)
    record_metrics(temp_db, "model_a", "variant_1", 0.9, 0.9, 0.9, 9, 42)
    table = comparison_table(temp_db)
    assert len(table) == 1
    assert table.iloc[0]["f1"] == pytest.approx(0.9)


def test_record_predictions_roundtrip(temp_db):
    ids = np.array([1, 2, 3])
    scores = np.array([0.1, 0.9, 0.5])
    is_anomaly = np.array([False, True, False])
    true_is_spoof = np.array([False, True, False])
    record_predictions(temp_db, "model_a", ids, scores, is_anomaly, true_is_spoof)
    result = temp_db.execute("SELECT COUNT(*) FROM predictions WHERE model = 'model_a'").fetchone()
    assert result[0] == 3
