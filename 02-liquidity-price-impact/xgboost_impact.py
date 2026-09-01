"""Order-book price-impact model: XGBoost regression on REAL BTCUSDT
perpetual-futures order-book depth and trade-flow data (Binance Vision),
predicting the forward VWAP price return over the next few minutes.

Requires `data/download_binance_data.py` to have been run first.
Run: venv/Scripts/python.exe xgboost_impact.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from persist_metrics import persist_metrics

SEED = 42
HORIZONS = [1, 5, 15]     # minutes-ahead targets compared in the multi-horizon table
PRIMARY_HORIZON = 1        # headline model / all single-model figures use this one
TEST_FRAC = 0.2            # last 20% held out, chronologically (no shuffling)
RAW_DIR = Path("data/raw")
FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")


def load_real_bookdepth(raw_dir: Path) -> pl.DataFrame:
    """Loads real BTCUSDT bookDepth snapshots (Binance Vision) and pivots
    the long percentage/depth rows into one row per ~30s snapshot.
    """
    files = sorted((raw_dir / "bookDepth").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No bookDepth files in {raw_dir}/bookDepth -- run data/download_binance_data.py first")

    df = pl.concat([pl.read_csv(f, try_parse_dates=True) for f in files])
    df = df.with_columns(
        pl.concat_str([pl.lit("pct_"), pl.col("percentage").cast(pl.Utf8)]).alias("pct_col")
    )
    wide = df.pivot(values="depth", index="timestamp", on="pct_col", aggregate_function="first").sort("timestamp")

    wide = wide.with_columns([
        pl.col("pct_-1").alias("bid_depth_near"),
        pl.col("pct_1").alias("ask_depth_near"),
        pl.col("pct_-5").alias("bid_depth_far"),
        pl.col("pct_5").alias("ask_depth_far"),
    ]).with_columns([
        ((pl.col("bid_depth_near") - pl.col("ask_depth_near")) /
         (pl.col("bid_depth_near") + pl.col("ask_depth_near"))).alias("near_depth_imbalance"),
        ((pl.col("bid_depth_far") - pl.col("ask_depth_far")) /
         (pl.col("bid_depth_far") + pl.col("ask_depth_far"))).alias("far_depth_imbalance"),
        (pl.col("bid_depth_far") + pl.col("ask_depth_far")).alias("total_depth"),
        pl.col("timestamp").dt.truncate("1m").alias("bin"),
    ])

    return (
        wide.group_by("bin")
        .agg([
            pl.col("near_depth_imbalance").last(),
            pl.col("far_depth_imbalance").last(),
            pl.col("total_depth").last(),
        ])
        .sort("bin")
    )


def load_real_aggtrades(raw_dir: Path) -> pl.DataFrame:
    """Loads real BTCUSDT aggregated trades (Binance Vision) and resamples
    to 1-minute bins: VWAP price, total volume, and signed order-flow
    (aggressor side, from `is_buyer_maker`).
    """
    files = sorted((raw_dir / "aggTrades").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No aggTrades files in {raw_dir}/aggTrades -- run data/download_binance_data.py first")

    df = pl.concat([
        pl.scan_csv(f, schema_overrides={"is_buyer_maker": pl.Boolean}) for f in files
    ]).collect()

    df = df.with_columns(
        pl.from_epoch("transact_time", time_unit="ms").alias("ts")
    ).with_columns([
        pl.col("ts").dt.truncate("1m").alias("bin"),
        (pl.col("price") * pl.col("quantity")).alias("notional"),
        # is_buyer_maker=True means the aggressor (taker) was the SELLER.
        pl.when(pl.col("is_buyer_maker")).then(-pl.col("quantity")).otherwise(pl.col("quantity")).alias("signed_qty"),
    ])

    trades_1m = (
        df.group_by("bin")
        .agg([
            (pl.col("notional").sum() / pl.col("quantity").sum()).alias("vwap"),
            pl.col("quantity").sum().alias("trade_volume"),
            pl.col("signed_qty").sum().alias("signed_volume"),
        ])
        .sort("bin")
    )
    return trades_1m.with_columns(
        (pl.col("signed_volume") / pl.col("trade_volume")).alias("trade_flow_imbalance")
    )


def build_dataset(raw_dir: Path) -> pl.DataFrame:
    book = load_real_bookdepth(raw_dir)
    trades = load_real_aggtrades(raw_dir)
    return trades.join(book, on="bin", how="inner").sort("bin")


def engineer_features(df: pl.DataFrame, horizons: list[int]) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Builds model features from information available up to bin t only,
    and forward VWAP-return targets for each horizon -- no lookahead.
    """
    df = df.with_columns(pl.col("vwap").log().diff().alias("log_return"))
    df = df.with_columns([
        pl.col("trade_flow_imbalance").rolling_mean(window_size=5).alias("ofi_roll5"),
        pl.col("trade_flow_imbalance").rolling_mean(window_size=15).alias("ofi_roll15"),
        pl.col("trade_volume").rolling_mean(window_size=5).alias("volume_roll5"),
        pl.col("log_return").rolling_std(window_size=15).alias("realized_vol_roll15"),
        pl.col("log_return").shift(1).alias("lag_return_1"),
        pl.col("log_return").shift(5).alias("lag_return_5"),
    ])

    for h in horizons:
        df = df.with_columns(
            (pl.col("vwap").shift(-h).log() - pl.col("vwap").log()).alias(f"future_return_{h}")
        )

    feature_cols = [
        "trade_flow_imbalance", "ofi_roll5", "ofi_roll15",
        "near_depth_imbalance", "far_depth_imbalance", "total_depth",
        "trade_volume", "volume_roll5", "realized_vol_roll15",
        "lag_return_1", "lag_return_5",
    ]
    target_cols = [f"future_return_{h}" for h in horizons]
    df = df.drop_nulls(subset=feature_cols + target_cols)
    return df, feature_cols, target_cols


def chronological_split(df: pl.DataFrame, test_frac: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    split_idx = int(df.height * (1 - test_frac))
    return df[:split_idx], df[split_idx:]


def tune_model(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[XGBRegressor, dict]:
    """RandomizedSearchCV over a small XGBoost hyperparameter space, scored
    with TimeSeriesSplit (never a random/shuffled K-fold) so every
    validation fold is strictly later in time than its training fold.
    """
    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "n_estimators": [200, 300, 400, 600],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    base = XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=-1, tree_method="hist")
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=20,
        scoring="neg_root_mean_squared_error", cv=TimeSeriesSplit(n_splits=4),
        random_state=seed, n_jobs=1, verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def evaluate(model: XGBRegressor, X_test: np.ndarray, y_test: np.ndarray, y_train_mean: float) -> tuple[dict, np.ndarray]:
    y_pred = model.predict(X_test)
    baseline_pred = np.full_like(y_test, fill_value=y_train_mean)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        "n_test": int(len(y_test)),
    }
    return metrics, y_pred


def multi_horizon_eval(df: pl.DataFrame, feature_cols: list[str], horizons: list[int],
                        best_params: dict, seed: int) -> dict:
    train_df, test_df = chronological_split(df, TEST_FRAC)
    results = {}
    for h in horizons:
        target_col = f"future_return_{h}"
        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df[target_col].to_numpy()
        X_test = test_df.select(feature_cols).to_numpy()
        y_test = test_df[target_col].to_numpy()

        model = XGBRegressor(**best_params, objective="reg:squarederror", random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        metrics, _ = evaluate(model, X_test, y_test, float(y_train.mean()))
        results[h] = metrics
    return results


def plot_price_timeseries(df: pl.DataFrame, out_path: Path) -> None:
    bins = df["bin"].to_list()
    vwap = df["vwap"].to_numpy()
    plt.figure(figsize=(11, 5))
    plt.plot(bins, vwap, linewidth=0.6, color="#3B7DD8")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xlabel("Date (UTC)")
    plt.ylabel("BTCUSDT VWAP (USDT)")
    plt.title("Real BTCUSDT Perpetual-Futures Price -- Binance Vision, 1-min VWAP")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def animate_price_timeseries(df: pl.DataFrame, out_path: Path, n_frames: int = 60) -> None:
    """Racing line-chart GIF of the same real VWAP series used in
    plot_price_timeseries -- reveals the line progressively across frames
    with a floating annotated label at the advancing tip."""
    import matplotlib.animation as animation

    bins = df["bin"].to_list()
    vwap = df["vwap"].to_numpy()

    # Subsample the real, already-computed series down to n_frames points.
    n_points = len(vwap)
    frame_count = min(n_frames, n_points)
    idx = np.linspace(0, n_points - 1, frame_count).astype(int)
    bins_s = [bins[i] for i in idx]
    vwap_s = vwap[idx]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        line, = ax.plot([], [], linewidth=1.2, color="#3B7DD8")
        ax.set_xlim(bins_s[0], bins_s[-1])
        pad = (vwap_s.max() - vwap_s.min()) * 0.1 or 1.0
        ax.set_ylim(vwap_s.min() - pad, vwap_s.max() + pad)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("BTCUSDT VWAP (USDT)")
        ax.set_title("Real BTCUSDT Perpetual-Futures Price -- Binance Vision, 1-min VWAP")

        label = ax.annotate(
            "", xy=(bins_s[0], vwap_s[0]), xytext=(15, 15), textcoords="offset points",
            fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.4", fc="#3B7DD8", ec="none", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#3B7DD8"),
        )

        def update(frame):
            line.set_data(bins_s[: frame + 1], vwap_s[: frame + 1])
            x, y = bins_s[frame], vwap_s[frame]
            label.set_text(f"VWAP: {y:,.2f}")
            label.xy = (x, y)
            return line, label

        ani = animation.FuncAnimation(fig, update, frames=frame_count, interval=100, blit=False)
        ani.save(out_path, writer="pillow")
        plt.close(fig)


def plot_feature_importance(model: XGBRegressor, feature_cols: list[str], out_path: Path) -> None:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    scores = np.array([gain.get(f"f{i}", 0.0) for i in range(len(feature_cols))])
    order = np.argsort(scores)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=scores[order], y=np.array(feature_cols)[order], color="#3B7DD8")
    plt.xlabel("Gain-based importance")
    plt.title("XGBoost Feature Importance -- Price Impact Model (real data)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_volume_vs_price(df: pl.DataFrame, out_path: Path, sample_size: int = 4000) -> None:
    sample = df.sample(n=min(sample_size, df.height), seed=SEED)
    plt.figure(figsize=(8, 6))
    plt.scatter(sample["trade_volume"], sample["vwap"], alpha=0.25, s=10, color="#D8763B")
    plt.xlabel("Trade volume (BTC, per 1-min bin)")
    plt.ylabel("VWAP price (USDT)")
    plt.title("Trade Volume vs. Price (real BTCUSDT data)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_correlation_matrix(df: pl.DataFrame, feature_cols: list[str], target_col: str, out_path: Path) -> None:
    cols = feature_cols + [target_col]
    corr = np.corrcoef(df.select(cols).to_numpy(), rowvar=False)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                xticklabels=cols, yticklabels=cols, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Matrix (real data)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(y_test: np.ndarray, y_pred: np.ndarray, out_path: Path, sample_size: int = 4000) -> None:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_test), size=min(sample_size, len(y_test)), replace=False)
    lo, hi = np.percentile(y_test, [0.5, 99.5])

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test[idx], y_pred[idx], alpha=0.25, s=10, color="#3B9E7D")
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect prediction")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual forward return")
    plt.ylabel("Predicted forward return")
    plt.title("Predicted vs. Actual Forward Return (real data, test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading real BTCUSDT bookDepth + aggTrades data (Binance Vision)...")
    raw = build_dataset(RAW_DIR)
    print(f"  {raw.height:,} joined 1-minute bins, {raw['bin'].min()} .. {raw['bin'].max()}")

    print(f"Engineering features (horizons = {HORIZONS} minutes)...")
    df, feature_cols, target_cols = engineer_features(raw, HORIZONS)
    print(f"  {df.height:,} usable rows after feature/target construction, {len(feature_cols)} features")

    primary_target = f"future_return_{PRIMARY_HORIZON}"
    train_df, test_df = chronological_split(df, TEST_FRAC)
    print(f"  Train: {train_df.height:,} rows | Test: {test_df.height:,} rows (chronological split, no shuffling)")

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df[primary_target].to_numpy()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df[primary_target].to_numpy()

    print("Tuning XGBoost (RandomizedSearchCV, TimeSeriesSplit, 20 iterations)...")
    model, best_params = tune_model(X_train, y_train, SEED)
    print(f"  Best params: {best_params}")

    print("Evaluating primary model on the held-out (chronologically later) test set...")
    metrics, y_pred = evaluate(model, X_test, y_test, y_train_mean=float(y_train.mean()))
    print(json.dumps(metrics, indent=2))

    print("Running multi-horizon comparison (1 / 5 / 15 minutes ahead)...")
    horizon_results = multi_horizon_eval(df, feature_cols, HORIZONS, best_params, SEED)
    print(json.dumps(horizon_results, indent=2))

    report = {
        "primary_horizon_minutes": PRIMARY_HORIZON,
        "primary_metrics": metrics,
        "best_params": best_params,
        "multi_horizon": horizon_results,
        "n_rows_total": df.height,
        "date_range": [str(df["bin"].min()), str(df["bin"].max())],
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Persisting metrics to DuckDB...")
    persist_metrics("xgboost", horizon_results)

    print("Rendering figures...")
    plot_price_timeseries(df, FIGURES_DIR / "price_timeseries.png")
    animate_price_timeseries(df, FIGURES_DIR / "price_timeseries_animated.gif")
    plot_feature_importance(model, feature_cols, FIGURES_DIR / "feature_importance.png")
    plot_volume_vs_price(df, FIGURES_DIR / "volume_vs_price.png")
    plot_correlation_matrix(df, feature_cols, primary_target, FIGURES_DIR / "correlation_matrix.png")
    plot_predicted_vs_actual(y_test, y_pred, FIGURES_DIR / "predicted_vs_actual.png")

    print(f"Done. Figures written to {FIGURES_DIR}/, metrics to {REPORTS_DIR}/metrics.json")


if __name__ == "__main__":
    main()
