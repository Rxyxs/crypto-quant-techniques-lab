"""Interpretable linear baseline for the order-book price-impact regression
task -- a Kyle/Almgren-Chriss-style linear impact model: forward VWAP
return regressed on contemporaneous order-flow imbalance and depth
features, fit with Ridge regression.

This is intentionally the simplest model in the project (§ comparison in
README): it gives a transparent, interpretable coefficient table (which
features move price and by how much per unit) to sit next to the
non-linear XGBoost and PyTorch models trained on the same real BTCUSDT
data (see xgboost_impact.py for data loading / feature engineering).

Requires `data/download_binance_data.py` to have been run first.
Run: venv/Scripts/python.exe baseline_impact.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from xgboost_impact import (
    HORIZONS,
    PRIMARY_HORIZON,
    RAW_DIR,
    SEED,
    TEST_FRAC,
    build_dataset,
    chronological_split,
    engineer_features,
)
from persist_metrics import persist_metrics

FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> tuple[Ridge, StandardScaler]:
    """Fits a standardized Ridge regression -- the linear, interpretable
    counterpart to the tree/neural models: coefficients are directly
    readable as "return per standard deviation of feature"."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha, random_state=SEED)
    model.fit(X_scaled, y_train)
    return model, scaler


def evaluate(model: Ridge, scaler: StandardScaler, X_test: np.ndarray, y_test: np.ndarray,
             y_train_mean: float) -> tuple[dict, np.ndarray]:
    y_pred = model.predict(scaler.transform(X_test))
    baseline_pred = np.full_like(y_test, fill_value=y_train_mean)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        "n_test": int(len(y_test)),
    }
    return metrics, y_pred


def multi_horizon_eval(df, feature_cols: list[str], horizons: list[int], alpha: float) -> dict:
    train_df, test_df = chronological_split(df, TEST_FRAC)
    results = {}
    for h in horizons:
        target_col = f"future_return_{h}"
        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df[target_col].to_numpy()
        X_test = test_df.select(feature_cols).to_numpy()
        y_test = test_df[target_col].to_numpy()

        model, scaler = fit_ridge(X_train, y_train, alpha=alpha)
        metrics, _ = evaluate(model, scaler, X_test, y_test, float(y_train.mean()))
        results[h] = metrics
    return results


def plot_coefficients(model: Ridge, feature_cols: list[str], out_path: Path) -> None:
    order = np.argsort(np.abs(model.coef_))
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_cols)[order], model.coef_[order], color="#7D3BD8")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standardized Ridge coefficient")
    plt.title("Linear Impact Model -- Coefficients (real data, 1-min horizon)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(y_test: np.ndarray, y_pred: np.ndarray, out_path: Path, sample_size: int = 4000) -> None:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_test), size=min(sample_size, len(y_test)), replace=False)
    lo, hi = np.percentile(y_test, [0.5, 99.5])

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test[idx], y_pred[idx], alpha=0.25, s=10, color="#D8B23B")
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect prediction")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual forward return")
    plt.ylabel("Predicted forward return")
    plt.title("Linear Baseline: Predicted vs. Actual (real data, test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=60, color="#D8763B", alpha=0.85)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Residual (actual - predicted forward return)")
    plt.ylabel("Count")
    plt.title("Linear Baseline: Residual Distribution (real data, test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading real BTCUSDT bookDepth + aggTrades data (Binance Vision)...")
    raw = build_dataset(RAW_DIR)
    print(f"  {raw.height:,} joined 1-minute bins")

    print(f"Engineering features (horizons = {HORIZONS} minutes)...")
    df, feature_cols, target_cols = engineer_features(raw, HORIZONS)
    print(f"  {df.height:,} usable rows after feature/target construction")

    primary_target = f"future_return_{PRIMARY_HORIZON}"
    train_df, test_df = chronological_split(df, TEST_FRAC)

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df[primary_target].to_numpy()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df[primary_target].to_numpy()

    print("Fitting Ridge linear impact model (primary horizon)...")
    model, scaler = fit_ridge(X_train, y_train, alpha=1.0)
    metrics, y_pred = evaluate(model, scaler, X_test, y_test, y_train_mean=float(y_train.mean()))
    print(json.dumps(metrics, indent=2))

    print("Running multi-horizon comparison (1 / 5 / 15 minutes ahead)...")
    horizon_results = multi_horizon_eval(df, feature_cols, HORIZONS, alpha=1.0)
    print(json.dumps(horizon_results, indent=2))

    coefficients = dict(zip(feature_cols, [float(c) for c in model.coef_]))
    report = {
        "model": "ridge_linear_baseline",
        "primary_horizon_minutes": PRIMARY_HORIZON,
        "primary_metrics": metrics,
        "coefficients": coefficients,
        "multi_horizon": horizon_results,
        "n_rows_total": df.height,
    }
    with open(REPORTS_DIR / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Persisting metrics to DuckDB...")
    persist_metrics("ridge_linear_baseline", horizon_results)

    print("Rendering figures...")
    plot_coefficients(model, feature_cols, FIGURES_DIR / "baseline_coefficients.png")
    plot_predicted_vs_actual(y_test, y_pred, FIGURES_DIR / "baseline_predicted_vs_actual.png")
    plot_residuals(y_test, y_pred, FIGURES_DIR / "baseline_residuals.png")

    print(f"Done. Figures written to {FIGURES_DIR}/, metrics to {REPORTS_DIR}/metrics_baseline.json")


if __name__ == "__main__":
    main()
