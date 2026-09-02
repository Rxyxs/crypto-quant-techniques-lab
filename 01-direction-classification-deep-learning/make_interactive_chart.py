"""Builds a self-contained interactive Plotly chart of the walk-forward
gross vs. net (cost-adjusted) cumulative PnL curve from the real,
already-committed outputs/reports/walkforward_equity_curve.csv (9,978
out-of-sample test rows across 5 walk-forward folds). No numbers are
invented here -- this script only re-renders the existing CSV.
"""
import polars as pl
import plotly.graph_objects as go

df = pl.read_csv("outputs/reports/walkforward_equity_curve.csv")
x = list(range(df.height))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=df["gross_cum"].to_list(),
    mode="lines", name="Gross cumulative log-return (no costs)",
    line=dict(color="#2E86AB", width=1.5),
))
fig.add_trace(go.Scatter(
    x=x, y=df["net_cum"].to_list(),
    mode="lines", name="Net cumulative log-return (10 bps cost+slippage/trade)",
    line=dict(color="#E63946", width=1.5),
))
fig.add_hline(y=0, line_dash="dot", line_color="gray")

fig.update_layout(
    title="Conv1D + Attention walk-forward: gross vs. cost-adjusted PnL (5 folds, 9,978 OOS candles, real BTCUSDT)",
    xaxis_title="Out-of-sample candle index (concatenated across 5 walk-forward folds)",
    yaxis_title="Cumulative log-return",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=1000, height=550,
)

out = "outputs/interactive/walkforward_pnl_interactive.html"
fig.write_html(out, include_plotlyjs="inline", full_html=True)
print("wrote", out)
