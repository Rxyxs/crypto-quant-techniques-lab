from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SNAPSHOTS = 5000
N_SPOOF_EVENTS = 25
SPOOF_BURST_RANGE = (2, 5)
CONTAMINATION = 0.02

# Full L2 depth: how many price levels per side the book carries, and how
# much larger (on average) each successive level is than the one above it
# -- a standard, realistic order-book depth profile (more resting size the
# further you get from the touch).
N_LEVELS = 5
LEVEL_SIZE_GROWTH = 1.35
SPOOF_LEVELS_RANGE = (1, 3)  # a spoof burst layers 1-3 contiguous levels, not just the touch

VELOCITY_WINDOW = 10  # snapshots used to normalize a level's size-delta into a "velocity"

FEATURE_COLS = [
    "order_book_imbalance",
    "cancel_to_trade_ratio",
    "order_lifetime_ms",
    "price_impact",
    "top_level_size_ratio",
]

# Every per-level cancel-to-trade ratio and size-velocity column, computed
# by `engineer_features` across the full L2 depth -- available for
# diagnostics (e.g. "which level actually triggered this alert") and for
# the raw-vs-aggregated ablation in the README/notebook, but NOT the
# default model input (see L2_FEATURE_COLS below and §5 of the README for
# why).
RAW_L2_LEVEL_COLS = [
    f"{side}_cancel_ratio_l{lvl}" for side in ("bid", "ask") for lvl in range(N_LEVELS)
] + [
    f"{side}_size_velocity_l{lvl}" for side in ("bid", "ask") for lvl in range(N_LEVELS)
]

# Cross-level aggregates: the single worst level's cancel ratio / size
# velocity, on either side, for this snapshot -- collapses the full L2
# depth into two features that isolate "the most suspicious level" without
# diluting IsolationForest's random split sampling across ~20 mostly-noise
# raw columns (validated empirically, see README §5 -- feeding the raw
# per-level columns directly measurably *hurts* precision/recall here).
L2_AGGREGATE_COLS = ["max_cancel_ratio_across_levels", "max_size_velocity_across_levels"]

# Default model input. Built from an empirical ablation (README §5), not
# just "old features + new features": `cancel_to_trade_ratio` sums
# canceled/executed counts across all 10 level-slots (5 levels x 2 sides),
# so a spoof burst touching only 1-3 of those slots gets diluted into a
# whole-book average and *loses* discriminative power once the book has
# real L2 depth -- `max_cancel_ratio_across_levels` (the worst single
# level, not the book-wide average) replaces it. `order_book_imbalance`
# is kept: it is computed from total depth by construction, and remains
# the single strongest feature (median 0.60 on spoofed snapshots vs. 0.00
# on normal ones) precisely because a spoof burst's extra size pushes the
# *whole* book lopsided, unlike the diluted cancel ratio.
L2_FEATURE_COLS = [
    "order_book_imbalance",
    "order_lifetime_ms",
    "price_impact",
    "top_level_size_ratio",
] + L2_AGGREGATE_COLS

# For the ablation comparison only: every raw per-level column included too.
L2_FULL_RAW_FEATURE_COLS = FEATURE_COLS + RAW_L2_LEVEL_COLS + L2_AGGREGATE_COLS

OUTPUT_DIR = Path("outputs")


def simulate_orderbook(n_snapshots, rng, n_levels=N_LEVELS):
    """Simulates a full L2 order book: `n_levels` price levels per side,
    each with its own resting size and its own placed/canceled/executed
    order-flow counts per snapshot.

    A spoofing burst layers a contiguous *band* of levels (1-3 levels deep,
    not just the touch) on one randomly chosen side with a large size
    multiplier, cancels essentially all of it, and executes essentially
    none of it -- levels outside that band, and the whole opposite side,
    behave exactly like a normal snapshot. This is what makes true
    per-level features meaningful: an aggregate, whole-book feature would
    blur *which* levels were actually manipulated.
    """
    spoof_starts = rng.choice(np.arange(50, n_snapshots - 50), size=N_SPOOF_EVENTS, replace=False)
    burst_lengths = rng.integers(SPOOF_BURST_RANGE[0], SPOOF_BURST_RANGE[1] + 1, size=N_SPOOF_EVENTS)

    spoof_at = {}
    for start, length in zip(spoof_starts, burst_lengths):
        side = "bid" if rng.random() < 0.5 else "ask"
        depth = int(rng.integers(SPOOF_LEVELS_RANGE[0], SPOOF_LEVELS_RANGE[1] + 1))
        start_level = int(rng.integers(0, n_levels - depth + 1))
        levels = list(range(start_level, start_level + depth))
        for offset in range(int(length)):
            spoof_at[int(start) + offset] = (side, levels)

    mid_price = 50_000.0
    rows = []
    for t in range(n_snapshots):
        mid_price += rng.normal(0, 2.5)
        spread = abs(rng.normal(5.0, 1.5)) + 1.0

        is_spoof = t in spoof_at
        spoof_side, spoof_levels = spoof_at.get(t, (None, []))

        row = {"snapshot_id": t, "mid_price": mid_price, "spread": spread, "true_is_spoof": is_spoof}

        total_placed = total_canceled = total_executed = 0
        top_bid_size = top_ask_size = 0.0
        order_lifetime_samples = []
        price_impact_samples = []

        for side in ("bid", "ask"):
            for lvl in range(n_levels):
                base_size = rng.lognormal(mean=2.0, sigma=0.5) * (LEVEL_SIZE_GROWTH**lvl)
                num_placed = int(rng.integers(6, 20))
                # deeper levels naturally see relatively more cancellation and
                # less execution than the touch, even with no spoofing at all --
                # a genuine baseline profile the model has to learn is normal.
                base_cancel_frac = min(0.95, 0.5 + 0.06 * lvl)
                num_canceled = int(rng.integers(int(num_placed * base_cancel_frac), num_placed + 1))
                num_executed = max(0, num_placed - num_canceled)
                lifetime_ms = abs(rng.normal(800 - 15 * lvl, 250)) + 50
                impact = abs(rng.normal(0.0, 1.2)) / (1 + 0.4 * lvl)

                size = base_size
                if is_spoof and side == spoof_side and lvl in spoof_levels:
                    spoof_multiplier = rng.uniform(12, 30)
                    size = base_size + base_size * spoof_multiplier
                    num_canceled = num_placed
                    num_executed = 0
                    lifetime_ms = abs(rng.normal(120, 40)) + 10
                    impact = abs(rng.normal(0.3, 0.15))

                row[f"{side}_size_l{lvl}"] = size
                row[f"{side}_placed_l{lvl}"] = num_placed
                row[f"{side}_canceled_l{lvl}"] = num_canceled
                row[f"{side}_executed_l{lvl}"] = num_executed

                total_placed += num_placed
                total_canceled += num_canceled
                total_executed += num_executed
                order_lifetime_samples.append(lifetime_ms)
                price_impact_samples.append(impact)
                if lvl == 0:
                    if side == "bid":
                        top_bid_size = size
                    else:
                        top_ask_size = size

        total_bid_size = sum(row[f"bid_size_l{lvl}"] for lvl in range(n_levels))
        total_ask_size = sum(row[f"ask_size_l{lvl}"] for lvl in range(n_levels))
        total_size = total_bid_size + total_ask_size

        row.update(
            {
                "bid_size": top_bid_size,
                "ask_size": top_ask_size,
                "total_bid_depth": total_bid_size,
                "total_ask_depth": total_ask_size,
                "order_book_imbalance": (total_bid_size - total_ask_size) / total_size,
                "num_orders_placed": total_placed,
                "num_orders_canceled": total_canceled,
                "num_orders_executed": total_executed,
                "cancel_to_trade_ratio": total_canceled / (total_executed + 1),
                "order_lifetime_ms": float(np.mean(order_lifetime_samples)),
                "price_impact": float(np.mean(price_impact_samples)),
                # Diagnostic-only column for post-hoc analysis of *which*
                # levels a burst affected -- NOT a detection feature (it is
                # derived directly from the ground-truth label, so it must
                # never appear in `L2_FEATURE_COLS`).
                "spoofed_level_depth": len(spoof_levels) if is_spoof else 0,
            }
        )
        rows.append(row)

    return pl.DataFrame(rows)


def engineer_features(df, window=50, velocity_window=VELOCITY_WINDOW, n_levels=N_LEVELS):
    """Adds the top-of-book rolling-ratio feature (unchanged from the
    original single-level version) plus, per price level and per side:

        - `{side}_cancel_ratio_l{lvl}`: canceled / (executed + 1) at that
          level, in that snapshot -- a spoof concentrated at level 2 shows
          up at level 2's ratio, not diluted into a whole-book average.
        - `{side}_size_velocity_l{lvl}`: that level's snapshot-to-snapshot
          size change, z-scored against its own trailing rolling
          volatility -- "how fast is size appearing/disappearing here,
          relative to how that level normally moves", not just the raw
          delta (which would be dominated by naturally larger deep levels).

    All rolling/diff windows are trailing, so every feature at row t uses
    only information from row t and earlier -- required for both the
    lookahead-safety of a single batch fit and for the streaming
    simulation in `streaming_alerts.py` to be valid (a trailing rolling
    window computed once over the whole frame is equivalent to computing
    it snapshot-by-snapshot as a stream would).
    """
    df = df.with_columns(pl.max_horizontal(["bid_size", "ask_size"]).alias("top_size"))
    df = df.with_columns(
        pl.col("top_size").rolling_mean(window_size=window, min_samples=10).alias("rolling_avg_size")
    )
    global_mean_size = df["top_size"].mean()
    df = df.with_columns(pl.col("rolling_avg_size").fill_null(global_mean_size))
    df = df.with_columns((pl.col("top_size") / pl.col("rolling_avg_size")).alias("top_level_size_ratio"))

    level_exprs = []
    for side in ("bid", "ask"):
        for lvl in range(n_levels):
            level_exprs.append(
                (pl.col(f"{side}_canceled_l{lvl}") / (pl.col(f"{side}_executed_l{lvl}") + 1)).alias(
                    f"{side}_cancel_ratio_l{lvl}"
                )
            )
    df = df.with_columns(level_exprs)

    velocity_exprs = []
    for side in ("bid", "ask"):
        for lvl in range(n_levels):
            size_col = f"{side}_size_l{lvl}"
            delta_col = f"_{side}_size_delta_l{lvl}"
            df = df.with_columns(pl.col(size_col).diff().fill_null(0.0).alias(delta_col))
            rolling_std_col = f"_{side}_delta_std_l{lvl}"
            df = df.with_columns(
                pl.col(delta_col)
                .rolling_std(window_size=velocity_window, min_samples=5)
                .alias(rolling_std_col)
            )
            global_std = df[delta_col].std() or 1.0
            df = df.with_columns(pl.col(rolling_std_col).fill_null(global_std).clip(lower_bound=1e-6))
            velocity_exprs.append((pl.col(delta_col) / pl.col(rolling_std_col)).alias(f"{side}_size_velocity_l{lvl}"))
    df = df.with_columns(velocity_exprs)
    df = df.drop([c for c in df.columns if c.startswith("_bid_") or c.startswith("_ask_")])

    cancel_ratio_cols = [f"{side}_cancel_ratio_l{lvl}" for side in ("bid", "ask") for lvl in range(n_levels)]
    velocity_cols = [f"{side}_size_velocity_l{lvl}" for side in ("bid", "ask") for lvl in range(n_levels)]
    df = df.with_columns(
        [
            pl.max_horizontal(cancel_ratio_cols).alias("max_cancel_ratio_across_levels"),
            pl.max_horizontal([pl.col(c).abs() for c in velocity_cols]).alias("max_size_velocity_across_levels"),
        ]
    )
    return df


def fit_isolation_forest(X, contamination=CONTAMINATION, seed=SEED):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    model.fit(X_scaled)
    predictions = model.predict(X_scaled)  # 1 = inlier, -1 = outlier
    anomaly_score = -model.score_samples(X_scaled)  # higher = more anomalous
    is_anomaly = predictions == -1
    return model, scaler, is_anomaly, anomaly_score


def evaluate(true_is_spoof, is_anomaly):
    precision = precision_score(true_is_spoof, is_anomaly, zero_division=0)
    recall = recall_score(true_is_spoof, is_anomaly, zero_division=0)
    f1 = f1_score(true_is_spoof, is_anomaly, zero_division=0)
    cm = confusion_matrix(true_is_spoof, is_anomaly, labels=[False, True])
    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


def plot_anomaly_scatter(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    normal = df.filter(~pl.col("is_anomaly"))
    anomalies = df.filter(pl.col("is_anomaly"))

    ax.scatter(
        normal["top_level_size_ratio"], normal["price_impact"],
        s=18, alpha=0.5, color="#4C72B0", label=f"Normal (n={normal.height})",
    )
    ax.scatter(
        anomalies["top_level_size_ratio"], anomalies["price_impact"],
        s=45, alpha=0.9, color="red", marker="x", linewidths=2,
        label=f"Anomaly flagged (n={anomalies.height})",
    )
    ax.set_xlabel("Top-of-book size ratio (vs. 50-snapshot rolling average)")
    ax.set_ylabel("Realized price impact")
    ax.set_title("Isolation Forest anomalies: displayed size vs. realized price impact")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_imbalance_histogram(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        df.filter(~pl.col("is_anomaly"))["order_book_imbalance"].to_numpy(),
        bins=40, color="#4C72B0", label="Normal snapshots", ax=ax, stat="count",
    )
    sns.histplot(
        df.filter(pl.col("is_anomaly"))["order_book_imbalance"].to_numpy(),
        bins=40, color="red", label="Flagged anomalies", ax=ax, stat="count",
    )
    ax.set_xlabel("Order book imbalance (full L2 depth, (bid - ask) / total)")
    ax.set_ylabel("Number of snapshots")
    ax.set_title("Order book imbalance distribution: normal vs. flagged snapshots")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_pipeline(feature_cols=L2_FEATURE_COLS):
    rng = np.random.default_rng(SEED)
    df = simulate_orderbook(N_SNAPSHOTS, rng)
    df = engineer_features(df)

    X = df.select(feature_cols).to_numpy()
    _, _, is_anomaly, anomaly_score = fit_isolation_forest(X)
    df = df.with_columns(
        [pl.Series("is_anomaly", is_anomaly), pl.Series("anomaly_score", anomaly_score)]
    )
    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = run_pipeline()

    metrics = evaluate(df["true_is_spoof"].to_numpy(), df["is_anomaly"].to_numpy())
    n_true_spoof = int(df["true_is_spoof"].sum())
    n_flagged = int(df["is_anomaly"].sum())

    print(f"Simulated {df.height} order book snapshots ({N_LEVELS} L2 levels/side), "
          f"{n_true_spoof} true spoofing events injected.")
    print(f"Isolation Forest flagged {n_flagged} snapshots as anomalous (contamination={CONTAMINATION}).")
    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print("Confusion matrix (rows=true [normal, spoof], cols=predicted [normal, anomaly]):")
    print(metrics["confusion_matrix"])

    df.write_csv(OUTPUT_DIR / "orderbook_snapshots_scored.csv")
    print(f"\nSaved {OUTPUT_DIR / 'orderbook_snapshots_scored.csv'}")

    plot_anomaly_scatter(df, OUTPUT_DIR / "anomaly_scatter.png")
    plot_imbalance_histogram(df, OUTPUT_DIR / "imbalance_histogram.png")
    print(f"Saved figures to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
