"""Benchmark comparativo: señales técnicas tradicionales vs. modelos de
Machine Learning (regresión logística y Gradient Boosting/LightGBM), las
cinco evaluadas sobre exactamente el mismo test set, con el mismo motor de
backtest y el mismo modelo de fricciones (`frictions.py`) aplicado por igual
a todas -- ninguna estrategia recibe una ventaja de evaluación que las
demás no tengan.

Uso:
    python compare_strategies.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

from backtest_engine import build_features_and_labels, chronological_split, compute_metrics, load_price_data, run_backtest
from frictions import apply_frictions, total_friction_cost_summary
from ml_models import train_lightgbm, train_logistic_regression
from technical_signals import compute_all_technical_signals

OUTPUT_TABLE_PATH = "outputs/comparison_table.csv"
OUTPUT_CHART_PATH = "outputs/comparison_dashboard.png"
OUTPUT_SUMMARY_JSON = "outputs/comparison_summary.json"

STRATEGY_LABELS = {
    "signal_sma_crossover": "SMA Crossover (10/50)",
    "signal_rsi_mean_reversion": "RSI Mean-Reversion (14, <30)",
    "signal_momentum": "Momentum Sign (10d)",
    "signal_logistic_regression": "Logistic Regression",
    "signal_lightgbm": "LightGBM (Gradient Boosting)",
}


def _row_for_strategy(name: str, test_df: pd.DataFrame, signal) -> dict:
    bt = run_backtest(test_df, signal)
    bt = apply_frictions(bt)
    gross = compute_metrics(bt)
    net = compute_metrics(
        bt, return_col="strategy_return_net_taker",
        equity_col="equity_curve_net_taker", drawdown_col="drawdown_net_taker",
    )
    friction = total_friction_cost_summary(bt)
    return {
        "strategy": STRATEGY_LABELS[name],
        "sharpe_gross": gross["Sharpe Ratio (annualized)"],
        "sharpe_net": net["Sharpe Ratio (annualized)"],
        "max_drawdown_gross": gross["Max Drawdown"],
        "max_drawdown_net": net["Max Drawdown"],
        "total_return_gross": gross["Total Return"],
        "total_return_net": net["Total Return"],
        "win_rate_net": net["Win Rate"],
        "signal_accuracy": gross["Signal Accuracy"],
        "trades_taken": gross["Trades Taken"],
        "friction_drag_pct": friction["total_friction_drag_pct"],
    }, bt


def run_comparison() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    prices, is_real = load_price_data()
    df, feature_cols = build_features_and_labels(prices)
    df = compute_all_technical_signals(df)
    train_df, test_df = chronological_split(df)

    proba_lr = None
    signals = {}
    signal_lr, proba_lr = train_logistic_regression(train_df, test_df, feature_cols)
    signal_gbm, proba_gbm = train_lightgbm(train_df, test_df, feature_cols)

    signals["signal_sma_crossover"] = test_df["signal_sma_crossover"].to_numpy()
    signals["signal_rsi_mean_reversion"] = test_df["signal_rsi_mean_reversion"].to_numpy()
    signals["signal_momentum"] = test_df["signal_momentum"].to_numpy()
    signals["signal_logistic_regression"] = signal_lr
    signals["signal_lightgbm"] = signal_gbm

    rows = []
    backtests = {}
    for name, signal in signals.items():
        row, bt = _row_for_strategy(name, test_df, signal)
        rows.append(row)
        backtests[name] = bt

    table = pd.DataFrame(rows).sort_values("sharpe_net", ascending=False).reset_index(drop=True)
    return table, backtests, is_real, len(test_df)


def render_comparison_chart(table: pd.DataFrame, backtests: dict[str, pd.DataFrame], output_path: str = OUTPUT_CHART_PATH) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.tab10.colors

    order = table["strategy"].tolist()
    name_by_label = {v: k for k, v in STRATEGY_LABELS.items()}

    axes[0].barh(table["strategy"], table["sharpe_net"], color=[colors[i % 10] for i in range(len(table))])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Sharpe Ratio anualizado (neto de fricciones)")
    axes[0].set_title("Sharpe neto por estrategia")
    axes[0].grid(alpha=0.3, axis="x")

    for i, label in enumerate(order):
        key = name_by_label[label]
        bt = backtests[key]
        axes[1].plot(bt["date"], bt["equity_curve_net_taker"], label=label, color=colors[i % 10], linewidth=1.6)
    axes[1].set_title("Equity curves netas de fricciones (test period)")
    axes[1].set_ylabel("Growth of $1")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Benchmark: Señales Técnicas vs. Gradient Boosting (net de comisiones + slippage)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_comparison_animation(
    table: pd.DataFrame, backtests: dict[str, pd.DataFrame],
    output_path: str = "outputs/comparison_dashboard_animated.gif",
) -> None:
    """Racing-line animation of the net-of-friction equity curves for all
    five strategies, subsampled from the same real backtests used in
    `render_comparison_chart` -- no synthetic values."""
    import os

    colors = plt.cm.tab10.colors
    order = table["strategy"].tolist()
    name_by_label = {v: k for k, v in STRATEGY_LABELS.items()}

    series = []
    dates = None
    for i, label in enumerate(order):
        key = name_by_label[label]
        bt = backtests[key]
        if dates is None:
            dates = bt["date"].to_numpy()
        series.append((label, bt["equity_curve_net_taker"].to_numpy(), colors[i % 10]))

    n_points = len(dates)
    n_frames = min(50, n_points)
    idx = np.unique(np.linspace(0, n_points - 1, n_frames).astype(int))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(dates[0], dates[-1])
    y_all = np.concatenate([s[1] for s in series])
    pad = (y_all.max() - y_all.min()) * 0.08
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
    ax.set_title("Equity curves netas de fricciones (animada, test period)")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.2)
    ax.tick_params(axis="x", rotation=45)

    lines = []
    labels = []
    for name, values, color in series:
        (line,) = ax.plot([], [], label=name, color=color, linewidth=1.6)
        lines.append((line, values))
        label = ax.annotate(
            "", xy=(dates[0], values[0]), xytext=(10, 0), textcoords="offset points",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.9),
            va="center",
        )
        labels.append(label)
    ax.legend(loc="upper left", fontsize=7)

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
    print("Ejecutando benchmark comparativo: tecnicas tradicionales vs. ML...")
    table, backtests, is_real, n_test = run_comparison()

    source = "real (Binance Vision)" if is_real else "SYNTHETIC (fallback)"
    print(f"Fuente de datos: {source} | Test set: {n_test} dias\n")

    print(table.to_string(index=False))

    import os
    os.makedirs("outputs", exist_ok=True)
    table.to_csv(OUTPUT_TABLE_PATH, index=False)
    print(f"\nTabla guardada en {OUTPUT_TABLE_PATH}")

    render_comparison_chart(table, backtests)
    print(f"Grafico guardado en {OUTPUT_CHART_PATH}")

    render_comparison_animation(table, backtests)
    print("Animacion guardada en outputs/comparison_dashboard_animated.gif")

    summary = {
        "data_source": source,
        "n_test_days": n_test,
        "best_strategy_by_net_sharpe": table.iloc[0]["strategy"],
        "table": table.to_dict(orient="records"),
    }
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Resumen guardado en {OUTPUT_SUMMARY_JSON}")

    try:
        from db_persistence import persist_comparison_run

        run_id = persist_comparison_run(table, source, n_test)
        print(f"Corrida #{run_id} persistida en outputs/backtest_history.duckdb")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
