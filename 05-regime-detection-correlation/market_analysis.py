import datetime as dt
import sys
from collections import Counter
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SEED = 42
ASSETS = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "AVAX", "MATIC"]
N_DAYS = 1095
WINDOW = 20
K_RANGE = range(2, 7)
OUTPUT_DIR = Path("outputs")

# Equicorrelation one-factor model per regime: return_i,t = drift + vol * (sqrt(corr)*common_t + sqrt(1-corr)*idio_i,t).
# `corr` rises sharply in Bear Crash, reflecting the real correlation-breakdown/contagion
# effect seen across crypto assets during broad market stress.
REGIMES = [
    {"name": "Bull Quiet", "drift": 0.0015, "vol": 0.015, "corr": 0.35},
    {"name": "Bull Volatile", "drift": 0.0025, "vol": 0.045, "corr": 0.45},
    {"name": "Bear Crash", "drift": -0.0040, "vol": 0.065, "corr": 0.80},
    {"name": "Sideways", "drift": 0.0001, "vol": 0.020, "corr": 0.25},
]

# Rows sum to 1. High diagonal persistence (0.94-0.97) keeps regimes contiguous for
# multi-day/week stretches, matching how real market regimes actually behave.
TRANSITION = np.array(
    [
        [0.97, 0.01, 0.01, 0.01],
        [0.02, 0.96, 0.01, 0.01],
        [0.03, 0.01, 0.94, 0.02],
        [0.02, 0.01, 0.01, 0.96],
    ]
)


def simulate_regime_path(n_days, transition, start_state, rng):
    states = np.empty(n_days, dtype=int)
    states[0] = start_state
    n_states = transition.shape[0]
    for t in range(1, n_days):
        states[t] = rng.choice(n_states, p=transition[states[t - 1]])
    return states


def simulate_returns(states, regimes, n_assets, rng):
    n_days = len(states)
    returns = np.empty((n_days, n_assets))
    for t, s in enumerate(states):
        regime = regimes[s]
        rho = regime["corr"]
        common = rng.standard_normal()
        idio = rng.standard_normal(n_assets)
        shock = np.sqrt(rho) * common + np.sqrt(1 - rho) * idio
        returns[t] = regime["drift"] + regime["vol"] * shock
    return returns


def build_date_index(n_days, start=dt.date(2021, 1, 1)):
    end = start + dt.timedelta(days=n_days - 1)
    return pl.date_range(start, end, interval="1d", eager=True)


def build_returns_frame(dates, returns, assets, true_states, regime_names):
    data = {"date": dates}
    data.update({asset: returns[:, i] for i, asset in enumerate(assets)})
    data["true_regime"] = [regime_names[s] for s in true_states]
    return pl.DataFrame(data)


def add_rolling_return_and_volatility(df, assets, window):
    df = df.with_columns(
        [pl.col(a).rolling_mean(window).alias(f"__{a}_mean") for a in assets]
        + [pl.col(a).rolling_std(window).alias(f"__{a}_std") for a in assets]
    )
    df = df.with_columns(
        [
            pl.mean_horizontal([f"__{a}_mean" for a in assets]).alias("avg_return"),
            pl.mean_horizontal([f"__{a}_std" for a in assets]).alias("avg_volatility"),
        ]
    )
    return df.drop([f"__{a}_mean" for a in assets] + [f"__{a}_std" for a in assets])


def rolling_correlation_dispersion_drawdown(returns, window):
    n_days, n_assets = returns.shape
    avg_pairwise_corr = np.full(n_days, np.nan)
    dispersion = np.full(n_days, np.nan)
    drawdown = np.full(n_days, np.nan)
    off_diag_mask = ~np.eye(n_assets, dtype=bool)

    for t in range(window - 1, n_days):
        w = returns[t - window + 1 : t + 1]
        corr = np.corrcoef(w, rowvar=False)
        avg_pairwise_corr[t] = corr[off_diag_mask].mean()
        dispersion[t] = w.std(axis=1).mean()

        index_level = np.cumprod(1 + w.mean(axis=1))
        running_max = np.maximum.accumulate(index_level)
        drawdown[t] = ((index_level - running_max) / running_max).min()

    return avg_pairwise_corr, dispersion, drawdown


def label_regimes(centroids_original_scale):
    returns_col = centroids_original_scale[:, 0]
    vol_col = centroids_original_scale[:, 1]
    median_return = np.median(returns_col)
    median_vol = np.median(vol_col)

    labels = []
    for r, v in zip(returns_col, vol_col):
        if r >= median_return and v < median_vol:
            labels.append("Bull Quiet")
        elif r >= median_return and v >= median_vol:
            labels.append("Bull Volatile")
        elif r < median_return and v >= median_vol:
            labels.append("Bear Crash")
        else:
            labels.append("Sideways / Consolidation")

    counts = Counter(labels)
    seen = Counter()
    disambiguated = []
    for label in labels:
        if counts[label] > 1:
            seen[label] += 1
            disambiguated.append(f"{label} ({seen[label]})")
        else:
            disambiguated.append(label)
    return disambiguated


def select_k_by_silhouette(X_scaled, k_range):
    scores = {}
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X_scaled)
        scores[k] = silhouette_score(X_scaled, model.labels_)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def plot_correlation_heatmap_overall(returns, assets, out_path):
    corr = np.corrcoef(returns, rowvar=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        xticklabels=assets, yticklabels=assets, ax=ax,
    )
    ax.set_title("Full-period asset correlation matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap_by_regime(returns, valid_idx, cluster_labels, regime_names, assets, best_k, out_path):
    n_cols = min(best_k, 3)
    n_rows = -(-best_k // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for c in range(best_k):
        days_in_cluster = valid_idx[cluster_labels == c]
        subset_returns = returns[days_in_cluster]
        corr_c = np.corrcoef(subset_returns, rowvar=False)
        ax = axes_flat[c]
        sns.heatmap(
            corr_c, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            xticklabels=assets, yticklabels=assets, ax=ax, cbar=(c == 0),
        )
        ax.set_title(f"{regime_names[c]}\n(n={len(days_in_cluster)} days)")

    for c in range(best_k, len(axes_flat)):
        axes_flat[c].axis("off")

    fig.suptitle("Asset correlation structure by detected regime")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cluster_scatter(X_scaled, cluster_labels, regime_names, best_k, model_used, silhouette, out_path):
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_scaled)
    palette = sns.color_palette("tab10", best_k)

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(best_k):
        mask = cluster_labels == c
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1], s=18, alpha=0.7, color=palette[c],
            label=f"{regime_names[c]} (n={mask.sum()})",
        )

    centroids_2d = pca.transform(
        np.array([X_scaled[cluster_labels == c].mean(axis=0) for c in range(best_k)])
    )
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", s=200, color="black", label="Centroids")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title(f"Market regimes detected via {model_used} (k={best_k}, silhouette={silhouette:.3f})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regime_timeline(dates_list, returns, valid_idx, cluster_labels, regime_names, best_k, out_path):
    index_level = 100 * np.cumprod(1 + returns.mean(axis=1))
    palette = sns.color_palette("tab10", best_k)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates_list, index_level, color="black", linewidth=1)

    seg_start = valid_idx[0]
    prev_c = cluster_labels[0]
    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] != prev_c:
            ax.axvspan(dates_list[seg_start], dates_list[valid_idx[i]], color=palette[prev_c], alpha=0.15)
            seg_start = valid_idx[i]
            prev_c = cluster_labels[i]
    ax.axvspan(dates_list[seg_start], dates_list[valid_idx[-1]], color=palette[prev_c], alpha=0.15)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[c], alpha=0.3, label=regime_names[c])
        for c in range(best_k)
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title("Equal-weight simulated market index with detected regime overlay")
    ax.set_ylabel("Index level (start = 100)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regime_timeline_animated(dates_list, returns, out_path, n_frames=48):
    """Racing line-chart version of the regime timeline: the equal-weight
    simulated index progressively draws across frames, with a floating
    label at the advancing tip showing the live index level."""
    index_level = 100 * np.cumprod(1 + returns.mean(axis=1))
    n_days = len(index_level)

    # Subsample real, already-computed data down to ~n_frames points.
    frame_idx = np.unique(np.linspace(0, n_days - 1, min(n_frames, n_days)).astype(int))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(dates_list[0], dates_list[-1])
    ax.set_ylim(index_level.min() * 0.97, index_level.max() * 1.03)
    ax.set_title("Equal-weight simulated market index (animated)")
    ax.set_ylabel("Index level (start = 100)")
    ax.grid(alpha=0.2)

    (line,) = ax.plot([], [], color="#00d3ff", linewidth=1.6)
    label = ax.annotate(
        "",
        xy=(dates_list[0], index_level[0]),
        xytext=(15, 15),
        textcoords="offset points",
        color="black",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#00d3ff", ec="none"),
    )

    def update(frame_num):
        i = frame_idx[frame_num]
        xs = [dates_list[j] for j in range(0, i + 1)]
        ys = index_level[: i + 1]
        line.set_data(xs, ys)
        label.xy = (dates_list[i], index_level[i])
        label.set_position((15, 15))
        label.set_text(f"Sim Index: {index_level[i]:.1f}")
        return line, label

    ani = FuncAnimation(fig, update, frames=len(frame_idx), interval=120, blit=False)
    ani.save(out_path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")


def save_results_to_duckdb(regime_summary, result_df, db_path):
    """Persist the per-regime summary and the full daily regime assignment
    table to a local DuckDB file, so downstream tools (BI, notebooks, a
    position-sizing rule) can query results with SQL instead of re-running
    the pipeline or parsing the CSV/plots."""
    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS regime_summary")
    con.execute("CREATE TABLE regime_summary AS SELECT * FROM regime_summary")
    con.execute("DROP TABLE IF EXISTS regime_days")
    con.execute("CREATE TABLE regime_days AS SELECT * FROM result_df")
    con.close()


def main():
    # Windows terminals default stdout to cp1252, which can't render the box-drawing
    # characters Polars uses to print DataFrames; force UTF-8 so `print(df)` never crashes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    regime_names_true = [r["name"] for r in REGIMES]

    true_states = simulate_regime_path(N_DAYS, TRANSITION, start_state=0, rng=rng)
    returns = simulate_returns(true_states, REGIMES, len(ASSETS), rng)
    dates = build_date_index(N_DAYS)
    dates_list = dates.to_list()

    df = build_returns_frame(dates, returns, ASSETS, true_states, regime_names_true)
    df = add_rolling_return_and_volatility(df, ASSETS, WINDOW)

    avg_corr, dispersion, drawdown = rolling_correlation_dispersion_drawdown(returns, WINDOW)
    df = df.with_columns(
        [
            pl.Series("avg_pairwise_corr", avg_corr),
            pl.Series("dispersion", dispersion),
            pl.Series("drawdown", drawdown),
        ]
    )

    valid = df.filter(pl.col("avg_pairwise_corr").is_not_nan())
    valid_idx = np.arange(WINDOW - 1, N_DAYS)
    assert valid.height == len(valid_idx)

    feature_cols = ["avg_return", "avg_volatility", "avg_pairwise_corr", "dispersion", "drawdown"]
    X = valid.select(feature_cols).to_numpy()
    true_regime_valid = valid["true_regime"].to_list()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, silhouette_by_k = select_k_by_silhouette(X_scaled, K_RANGE)
    print("Silhouette score by k (KMeans):")
    for k, score in silhouette_by_k.items():
        marker = " <- selected" if k == best_k else ""
        print(f"  k={k}: {score:.4f}{marker}")

    kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(X_scaled)
    kmeans_sil = silhouette_score(X_scaled, kmeans.labels_)

    gmm = GaussianMixture(n_components=best_k, random_state=SEED, n_init=5).fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)
    gmm_sil = silhouette_score(X_scaled, gmm_labels)

    if gmm_sil > kmeans_sil:
        final_labels, model_used, final_sil = gmm_labels, "GMM", gmm_sil
    else:
        final_labels, model_used, final_sil = kmeans.labels_, "KMeans", kmeans_sil

    print(f"\nKMeans silhouette: {kmeans_sil:.4f} | GMM silhouette: {gmm_sil:.4f}")
    print(f"Selected model: {model_used} (k={best_k}, silhouette={final_sil:.4f})")

    true_numeric = np.array([regime_names_true.index(r) for r in true_regime_valid])
    ari = adjusted_rand_score(true_numeric, final_labels)
    print(f"Adjusted Rand Index vs. simulated ground-truth regime: {ari:.4f}")

    centroids_scaled = np.array([X_scaled[final_labels == c].mean(axis=0) for c in range(best_k)])
    centroids_original = scaler.inverse_transform(centroids_scaled)
    regime_names = label_regimes(centroids_original)
    print("\nDetected regimes:")
    for c, name in enumerate(regime_names):
        print(f"  cluster {c}: {name} (n={int((final_labels == c).sum())} days)")

    result_df = valid.with_columns(pl.Series("cluster", final_labels))
    regime_summary = (
        result_df.group_by("cluster")
        .agg(
            [
                pl.len().alias("n_days"),
                pl.col("avg_return").mean().alias("avg_return_mean"),
                pl.col("avg_volatility").mean().alias("avg_volatility_mean"),
                pl.col("avg_pairwise_corr").mean().alias("avg_pairwise_corr_mean"),
                pl.col("dispersion").mean().alias("dispersion_mean"),
                pl.col("drawdown").mean().alias("drawdown_mean"),
            ]
        )
        .sort("cluster")
        .with_columns((pl.col("n_days") / pl.col("n_days").sum() * 100).round(1).alias("pct_days"))
    )
    regime_summary = regime_summary.with_columns(
        pl.Series("regime_label", [regime_names[c] for c in regime_summary["cluster"].to_list()])
    )
    regime_summary.write_csv(OUTPUT_DIR / "regime_summary.csv")
    print(f"\nSaved {OUTPUT_DIR / 'regime_summary.csv'}")
    print(regime_summary)

    db_path = OUTPUT_DIR / "market_regimes.duckdb"
    save_results_to_duckdb(regime_summary, result_df, db_path)
    print(f"Saved {db_path} (tables: regime_summary, regime_days)")

    plot_correlation_heatmap_overall(returns, ASSETS, OUTPUT_DIR / "correlation_heatmap_overall.png")
    plot_correlation_heatmap_by_regime(
        returns, valid_idx, final_labels, regime_names, ASSETS, best_k,
        OUTPUT_DIR / "correlation_heatmap_by_regime.png",
    )
    plot_cluster_scatter(
        X_scaled, final_labels, regime_names, best_k, model_used, final_sil,
        OUTPUT_DIR / "cluster_scatter.png",
    )
    plot_regime_timeline(
        dates_list, returns, valid_idx, final_labels, regime_names, best_k,
        OUTPUT_DIR / "regime_timeline.png",
    )
    plot_regime_timeline_animated(
        dates_list, returns, OUTPUT_DIR / "regime_timeline_animated.gif",
    )
    print(f"Saved figures to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
