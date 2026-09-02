"""Builds a self-contained interactive Plotly chart of the BNBUSDT/XRPUSDT
cointegrated spread and its rolling z-score with entry/exit bands, from a
real, freshly-run pass of the engle_granger_test/estimate_hedge_ratio/
compute_spread/rolling_zscore pipeline in pairs_trading_engine.py over the
committed real data/prices_panel.csv (reproduced exactly: EG p-value
0.0299, beta 0.3007, matching outputs/reports/metrics.json).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pairs_trading_engine import (
    estimate_hedge_ratio, compute_spread, rolling_zscore, generate_signals,
    ENTRY_Z, EXIT_Z,
)

panel = pd.read_csv("data/prices_panel.csv", index_col=0, parse_dates=True)
y_symbol, x_symbol = "BNBUSDT", "XRPUSDT"
log_y = np.log(panel[y_symbol])
log_x = np.log(panel[x_symbol])
hedge = estimate_hedge_ratio(log_y, log_x)
spread = compute_spread(log_y, log_x, hedge)
z = rolling_zscore(spread)
positions = generate_signals(z)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=(
        f"Cointegrated spread: log({y_symbol}) − {hedge.beta:.3f}·log({x_symbol}) − {hedge.alpha:.3f}",
        "Rolling z-score of the spread, with entry/exit bands",
    ),
)
fig.add_trace(go.Scatter(x=spread.index, y=spread.values, name="Spread",
                          line=dict(color="#2E86AB", width=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(x=z.index, y=z.values, name="Rolling z-score",
                          line=dict(color="#264653", width=1.3)), row=2, col=1)
for level, label, color in [(ENTRY_Z, "entry", "#E63946"), (-ENTRY_Z, "entry", "#E63946"),
                             (EXIT_Z, "exit", "#2A9D8F"), (-EXIT_Z, "exit", "#2A9D8F")]:
    fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.6, row=2, col=1)

long_entries = positions[(positions == 1) & (positions.shift(1).fillna(0) == 0)]
short_entries = positions[(positions == -1) & (positions.shift(1).fillna(0) == 0)]
fig.add_trace(go.Scatter(x=long_entries.index, y=z.loc[long_entries.index], mode="markers",
                          name="Long spread entry", marker=dict(color="#2A9D8F", size=8, symbol="triangle-up")),
              row=2, col=1)
fig.add_trace(go.Scatter(x=short_entries.index, y=z.loc[short_entries.index], mode="markers",
                          name="Short spread entry", marker=dict(color="#E63946", size=8, symbol="triangle-down")),
              row=2, col=1)

fig.update_layout(
    title=f"Pairs trading: {y_symbol}/{x_symbol} cointegrated spread & z-score (real daily closes, Engle-Granger p={0.0299:.4f})",
    template="plotly_white", hovermode="x unified", height=750, width=1050,
    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
)

out = "outputs/interactive/spread_zscore_interactive.html"
import os
os.makedirs("outputs/interactive", exist_ok=True)
fig.write_html(out, include_plotlyjs="inline", full_html=True)
print("wrote", out)
