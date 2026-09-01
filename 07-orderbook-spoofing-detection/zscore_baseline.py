"""Simple statistical baseline: a rule-based z-score detector for order book
spoofing, used purely as a reference point for the two learned models
(`detect_spoofing.py`'s Isolation Forest, `autoencoder_spoofing.py`'s
Autoencoder). Flags any snapshot where `order_book_imbalance` -- the single
strongest per-snapshot signal identified in the Isolation Forest README --
is more than `Z_THRESHOLD` standard deviations from its own mean. No
learning, no per-feature interaction, just one feature and a threshold:
the floor a learned model needs to clear to be worth the added complexity.
"""

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_score, recall_score

from detect_spoofing import N_SNAPSHOTS, SEED, engineer_features, simulate_orderbook

Z_THRESHOLD = 3.0
BASELINE_FEATURE = "order_book_imbalance"


def zscore_flag(df, feature=BASELINE_FEATURE, threshold=Z_THRESHOLD):
    values = df[feature].to_numpy()
    mean, std = values.mean(), values.std() or 1.0
    z = (values - mean) / std
    is_anomaly = np.abs(z) >= threshold
    return is_anomaly, z


def evaluate_baseline(df, feature=BASELINE_FEATURE, threshold=Z_THRESHOLD):
    is_anomaly, z = zscore_flag(df, feature=feature, threshold=threshold)
    y_true = df["true_is_spoof"].to_numpy()
    return {
        "precision": precision_score(y_true, is_anomaly, zero_division=0),
        "recall": recall_score(y_true, is_anomaly, zero_division=0),
        "f1": f1_score(y_true, is_anomaly, zero_division=0),
        "n_flagged": int(is_anomaly.sum()),
        "is_anomaly": is_anomaly,
        "z_score": z,
    }


def run_pipeline(feature=BASELINE_FEATURE, threshold=Z_THRESHOLD):
    rng = np.random.default_rng(SEED)
    df = simulate_orderbook(N_SNAPSHOTS, rng)
    df = engineer_features(df)
    is_anomaly, z = zscore_flag(df, feature=feature, threshold=threshold)
    df = df.with_columns([pl.Series("zscore_is_anomaly", is_anomaly), pl.Series("zscore_value", z)])
    return df


def main():
    df = run_pipeline()
    metrics = evaluate_baseline(df)
    print(f"Z-score baseline (|z({BASELINE_FEATURE})| >= {Z_THRESHOLD}) flagged {metrics['n_flagged']} snapshots.")
    print(f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
