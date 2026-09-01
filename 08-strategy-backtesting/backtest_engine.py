"""
Statistical backtesting engine for a crypto trading signal.

Trains a lookahead-safe LightGBM classifier to predict next-period price
direction, backtests the resulting long/flat signal against realized
returns net of a realistic friction model (maker/taker fees + dynamic
volatility-scaled slippage, see `frictions.py`), and renders a single
dashboard (gross vs. net equity curve, drawdown, confusion matrix,
performance metrics) from one real run.

Data: real daily BTCUSDT OHLCV from Binance Vision (`download_binance_data.py`),
downloaded automatically on first run. Falls back to a clearly-labeled
SYNTHETIC series (GBM with GARCH(1,1)-style volatility clustering) only if
the download is unavailable (e.g. no network in the current environment),
so the pipeline always has something to run against end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, accuracy_score

from download_binance_data import OUTPUT_PATH as BINANCE_CSV_PATH
from download_binance_data import download_binance_klines
from frictions import apply_frictions, total_friction_cost_summary
from ml_models import train_lightgbm

SEED = 42
N_DAYS = 1500
TRAIN_FRACTION = 0.7
PERIODS_PER_YEAR = 365
OUTPUT_PATH = "outputs/dashboard.png"


def generate_synthetic_crypto_prices(n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """Simulate a daily crypto-like price series with GARCH(1,1) volatility
    clustering. SYNTHETIC DATA — a stand-in for a real exchange feed, used
    only when `load_price_data` can't reach Binance Vision."""
    rng = np.random.default_rng(seed)

    omega, alpha, beta = 5e-5, 0.10, 0.85
    mu = 0.0005

    variances = np.empty(n_days)
    returns = np.empty(n_days)
    variances[0] = omega / (1 - alpha - beta)
    returns[0] = mu + np.sqrt(variances[0]) * rng.standard_normal()

    for t in range(1, n_days):
        variances[t] = omega + alpha * returns[t - 1] ** 2 + beta * variances[t - 1]
        returns[t] = mu + np.sqrt(variances[t]) * rng.standard_normal()

    price0 = 30_000.0
    prices = price0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "price": prices})


def load_price_data(csv_path: Path = BINANCE_CSV_PATH) -> tuple[pd.DataFrame, bool]:
    """Carga precios reales de Binance Vision: usa el CSV local si ya existe,
    lo descarga automáticamente si no, y solo cae a la serie sintética si la
    descarga falla (sin red disponible). Devuelve `(df, es_real)`."""
    if csv_path.exists():
        raw = pd.read_csv(csv_path, parse_dates=["date"])
        return raw[["date", "close"]].rename(columns={"close": "price"}), True

    try:
        print("No hay datos locales de Binance; descargando historial real...")
        raw = download_binance_klines()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(csv_path, index=False)
        return raw[["date", "close"]].rename(columns={"close": "price"}), True
    except Exception as exc:
        print(f"No se pudo descargar de Binance Vision ({exc}); usando serie sintetica de respaldo.")
        return generate_synthetic_crypto_prices(), False


def build_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Lookahead-safe features: every predictor for day t uses only returns
    known strictly before day t (shifted by at least 1). The label is the
    direction of day t's own return, which the model has never seen."""
    df = df.copy()
    df["return"] = np.log(df["price"]).diff()

    past_return = df["return"].shift(1)
    for w in (3, 5, 10, 20):
        df[f"mom_{w}d"] = past_return.rolling(w).mean()
        df[f"vol_{w}d"] = past_return.rolling(w).std()
    for lag in (1, 2, 3):
        df[f"lag_return_{lag}"] = df["return"].shift(lag)

    df["label_up"] = (df["return"] > 0).astype(int)

    feature_cols = [c for c in df.columns if c.startswith(("mom_", "vol_", "lag_return_"))]
    df = df.dropna(subset=feature_cols + ["label_up"]).reset_index(drop=True)
    return df, feature_cols


def chronological_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_backtest(test_df: pd.DataFrame, signal: np.ndarray) -> pd.DataFrame:
    """Position is 1 (long) when the model predicts an up day, else flat (0).
    The position for day t is decided before day t's return is realized, and
    earns exactly that day's realized return."""
    out = test_df.copy()
    out["signal"] = signal
    out["strategy_return"] = out["signal"] * out["return"]
    out["equity_curve"] = (1 + out["strategy_return"]).cumprod()
    out["buy_hold_curve"] = (1 + out["return"]).cumprod()
    running_max = out["equity_curve"].cummax()
    out["drawdown"] = out["equity_curve"] / running_max - 1
    return out


def compute_metrics(
    bt: pd.DataFrame,
    return_col: str = "strategy_return",
    equity_col: str = "equity_curve",
    drawdown_col: str = "drawdown",
) -> dict[str, float]:
    strat_returns = bt[return_col]
    sharpe = (
        strat_returns.mean() / strat_returns.std() * np.sqrt(PERIODS_PER_YEAR)
        if strat_returns.std() > 0
        else 0.0
    )
    trades = bt[bt["signal"] == 1]
    win_rate = (trades[return_col] > 0).mean() if len(trades) > 0 else 0.0
    max_drawdown = bt[drawdown_col].min()
    total_return = bt[equity_col].iloc[-1] - 1
    accuracy = accuracy_score(bt["label_up"], bt["signal"])

    return {
        "Sharpe Ratio (annualized)": round(float(sharpe), 3),
        "Win Rate": round(float(win_rate), 3),
        "Max Drawdown": round(float(max_drawdown), 3),
        "Total Return": round(float(total_return), 3),
        "Signal Accuracy": round(float(accuracy), 3),
        "Trades Taken": int(len(trades)),
    }


def render_dashboard(
    bt: pd.DataFrame, metrics_gross: dict[str, float], metrics_net: dict[str, float],
    friction_summary: dict[str, float], output_path: str = OUTPUT_PATH,
) -> None:
    cm = confusion_matrix(bt["label_up"], bt["signal"])

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], hspace=0.45, wspace=0.25)

    ax_equity = fig.add_subplot(gs[0, 0])
    ax_equity.plot(bt["date"], bt["equity_curve"], label="Strategy (gross, sin comisiones)",
                    color="#1f77b4", linewidth=1.8)
    ax_equity.plot(bt["date"], bt["equity_curve_net_taker"], label="Strategy (neto de comisiones + slippage)",
                    color="#d62728", linewidth=1.8)
    ax_equity.plot(bt["date"], bt["buy_hold_curve"], label="Buy & Hold", color="#888888", linestyle="--")
    ax_equity.set_title("Equity Curve (test period): con y sin fricciones")
    ax_equity.set_ylabel("Growth of $1")
    ax_equity.legend(fontsize=8)
    ax_equity.grid(alpha=0.3)
    ax_equity.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax_equity.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    ax_dd = fig.add_subplot(gs[0, 1])
    ax_dd.fill_between(bt["date"], bt["drawdown_net_taker"], 0, color="#d62728", alpha=0.6)
    ax_dd.set_title(f"Drawdown neto (max = {metrics_net['Max Drawdown']:.1%})")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(alpha=0.3)
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax_dd.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    ax_cm = fig.add_subplot(gs[1, 0])
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title("Signal vs. Actual Direction")
    ax_cm.set_xlabel("Predicted signal")
    ax_cm.set_ylabel("Actual direction")
    ax_cm.set_xticks([0, 1], ["Down/Flat", "Up"])
    ax_cm.set_yticks([0, 1], ["Down", "Up"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")
    rows = [
        ["Sharpe (gross)", f"{metrics_gross['Sharpe Ratio (annualized)']}"],
        ["Sharpe (net)", f"{metrics_net['Sharpe Ratio (annualized)']}"],
        ["Max Drawdown (gross)", f"{metrics_gross['Max Drawdown']:.1%}"],
        ["Max Drawdown (net)", f"{metrics_net['Max Drawdown']:.1%}"],
        ["Total Return (gross)", f"{metrics_gross['Total Return']:.1%}"],
        ["Total Return (net)", f"{metrics_net['Total Return']:.1%}"],
        ["Signal Accuracy", f"{metrics_gross['Signal Accuracy']}"],
        ["Position changes", f"{friction_summary['n_position_changes']}"],
        ["Friction drag (total)", f"{friction_summary['total_friction_drag_pct']:.2f}%"],
    ]
    table = ax_table.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    ax_table.set_title("Performance Metrics: Gross vs. Net of Frictions", pad=12, y=1.05)

    fig.suptitle("Crypto Strategy Backtest Dashboard (LightGBM signal, real Binance data)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_equity_curve_animation(
    bt: pd.DataFrame, output_path: str = "outputs/dashboard_animated.gif",
) -> None:
    """Racing-line animation of the same equity curves rendered in
    `render_dashboard` (gross, net of frictions, and Buy & Hold), subsampled
    from the real backtest -- no synthetic values."""
    import os

    series = [
        ("Strategy (gross)", bt["equity_curve"].to_numpy(), "#1f77b4"),
        ("Strategy (net)", bt["equity_curve_net_taker"].to_numpy(), "#d62728"),
        ("Buy & Hold", bt["buy_hold_curve"].to_numpy(), "#888888"),
    ]
    dates = bt["date"].to_numpy()

    n_points = len(dates)
    n_frames = min(50, n_points)
    idx = np.unique(np.linspace(0, n_points - 1, n_frames).astype(int))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(dates[0], dates[-1])
    y_all = np.concatenate([s[1] for s in series])
    pad = (y_all.max() - y_all.min()) * 0.08
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax.set_title("Equity Curve (animada): con y sin fricciones vs. Buy & Hold")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    lines = []
    labels = []
    for name, values, color in series:
        (line,) = ax.plot([], [], label=name, color=color, linewidth=1.8)
        lines.append((line, values))
        label = ax.annotate(
            "", xy=(dates[0], values[0]), xytext=(10, 0), textcoords="offset points",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.9),
            va="center",
        )
        labels.append(label)
    ax.legend(loc="upper left", fontsize=8)

    def update(frame_i):
        cut = idx[frame_i]
        for (line, values), label in zip(lines, labels):
            line.set_data(dates[: cut + 1], values[: cut + 1])
            label.xy = (dates[cut], values[cut])
            label.set_text(f"{line.get_label()}: {values[cut]:.3f}")
        return [l for line in lines for l in line[:1]] + labels

    ani = FuncAnimation(fig, update, frames=len(idx), interval=120, blit=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")


def main() -> None:
    prices, is_real = load_price_data()
    source = "real (Binance Vision)" if is_real else "SYNTHETIC (fallback, sin conexion)"
    print(f"Fuente de datos: {source} -- {len(prices)} filas, "
          f"{prices['date'].min().date()} -> {prices['date'].max().date()}")

    print("Building lookahead-safe features and labels...")
    df, feature_cols = build_features_and_labels(prices)
    print(f"Rows after feature construction: {len(df)}")

    train_df, test_df = chronological_split(df)
    print(f"Chronological split: train={len(train_df)} test={len(test_df)}")

    print("Training LightGBM signal classifier...")
    signal, proba = train_lightgbm(train_df, test_df, feature_cols)

    print("Running backtest on held-out test period...")
    bt = run_backtest(test_df, signal)
    bt = apply_frictions(bt)

    metrics_gross = compute_metrics(bt)
    metrics_net = compute_metrics(
        bt, return_col="strategy_return_net_taker",
        equity_col="equity_curve_net_taker", drawdown_col="drawdown_net_taker",
    )
    friction_summary = total_friction_cost_summary(bt)

    print("\n--- Performance Metrics (GROSS, sin comisiones) ---")
    for k, v in metrics_gross.items():
        print(f"{k}: {v}")

    print("\n--- Performance Metrics (NET, con comisiones + slippage) ---")
    for k, v in metrics_net.items():
        print(f"{k}: {v}")

    print("\n--- Impacto de fricciones ---")
    for k, v in friction_summary.items():
        print(f"{k}: {v}")

    print(f"\nRendering dashboard to {OUTPUT_PATH}...")
    render_dashboard(bt, metrics_gross, metrics_net, friction_summary)

    print("Rendering animated equity curve to outputs/dashboard_animated.gif...")
    render_equity_curve_animation(bt)
    print("Done.")


if __name__ == "__main__":
    main()
