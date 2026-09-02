"""Builds a self-contained interactive Plotly equity-curve comparison
across all 5 strategies (3 technical, 2 ML), net of commissions +
slippage, by calling compare_strategies.run_comparison() directly -- the
exact same function that produced outputs/comparison_summary.json in
this update's fresh run (606-day real BTCUSDT test set, Binance Vision).
"""
import plotly.graph_objects as go
from compare_strategies import run_comparison, STRATEGY_LABELS

table, backtests, is_real, n_test = run_comparison()
name_by_label = {v: k for k, v in STRATEGY_LABELS.items()}
order = table["strategy"].tolist()

palette = ["#2E86AB", "#E63946", "#2A9D8F", "#F4A261", "#8E44AD"]
fig = go.Figure()
for i, label in enumerate(order):
    key = name_by_label[label]
    bt = backtests[key]
    sharpe = table.loc[table["strategy"] == label, "sharpe_net"].iloc[0]
    fig.add_trace(go.Scatter(
        x=bt["date"], y=bt["equity_curve_net_taker"], mode="lines",
        name=f"{label} (Sharpe={sharpe:.2f})",
        line=dict(color=palette[i % len(palette)], width=1.8),
    ))
fig.add_hline(y=1.0, line_dash="dot", line_color="gray")

fig.update_layout(
    title=f"Strategy comparison: net-of-friction equity curves, {n_test}-day real BTCUSDT test set (Binance Vision)",
    xaxis_title="Date", yaxis_title="Growth of $1 (net of commissions + slippage)",
    template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=1050, height=600,
)

import os
os.makedirs("outputs/interactive", exist_ok=True)
out = "outputs/interactive/equity_curves_comparison.html"
fig.write_html(out, include_plotlyjs="inline", full_html=True)
print("wrote", out)
