import numpy as np

from detect_spoofing import engineer_features, simulate_orderbook
from zscore_baseline import BASELINE_FEATURE, evaluate_baseline, zscore_flag


def _sim(n=600, seed=5):
    rng = np.random.default_rng(seed)
    return engineer_features(simulate_orderbook(n, rng))


def test_zscore_flag_shape_and_type():
    df = _sim()
    is_anomaly, z = zscore_flag(df)
    assert len(is_anomaly) == df.height
    assert len(z) == df.height
    assert is_anomaly.dtype == bool


def test_tighter_threshold_flags_fewer_or_equal():
    df = _sim()
    tight, _ = zscore_flag(df, threshold=4.0)
    loose, _ = zscore_flag(df, threshold=2.0)
    assert tight.sum() <= loose.sum()


def test_evaluate_baseline_returns_bounded_metrics():
    df = _sim()
    metrics = evaluate_baseline(df)
    for key in ("precision", "recall", "f1"):
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["n_flagged"] == int(metrics["is_anomaly"].sum())


def test_baseline_feature_is_order_book_imbalance():
    assert BASELINE_FEATURE == "order_book_imbalance"
