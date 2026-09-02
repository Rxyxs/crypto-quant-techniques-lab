"""Builds a self-contained interactive Plotly chart of real BTCUSDT
1-minute price (VWAP) vs. order-book depth imbalance and trade-flow
imbalance, from real Binance Vision bookDepth + aggTrades data
(data/raw/, downloaded directly from data.binance.vision). This is a
genuine smoke-run of the same build_dataset()/engineer_features()
pipeline used by baseline_impact.py / xgboost_impact.py -- not
synthetic data.
"""
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost_impact import build_dataset, engineer_features

raw = build_dataset(Path("data/raw"))
df, feature_cols, target_cols = engineer_features(raw, [1, 5, 15])
df = df.with_columns(df["trade_flow_imbalance"].rolling_mean(window_size=15).alias("tfi_roll15"))

x = df["bin"].to_list()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(
    x=x, y=df["vwap"].to_list(), name="BTCUSDT VWAP (1-min, real)",
    line=dict(color="#2E86AB", width=1.5),
), secondary_y=False)
fig.add_trace(go.Scatter(
    x=x, y=df["near_depth_imbalance"].to_list(), name="Near-book depth imbalance",
    line=dict(color="#E63946", width=1, dash="dot"), opacity=0.7,
), secondary_y=True)
fig.add_trace(go.Scatter(
    x=x, y=df["tfi_roll15"].to_list(),
    name="Trade-flow imbalance (15-min rolling mean)",
    line=dict(color="#2A9D8F", width=1), opacity=0.7,
), secondary_y=True)

fig.update_layout(
    title="Real BTCUSDT price vs. order-book / trade-flow imbalance (2024-05-13, Binance Vision, 1-min bins)",
    template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=1050, height=550,
)
fig.update_yaxes(title_text="VWAP (USDT)", secondary_y=False)
fig.update_yaxes(title_text="Imbalance (signed, -1..1)", secondary_y=True)
fig.update_xaxes(title_text="Time (UTC)")

out_dir = Path("outputs/interactive")
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "price_vs_liquidity_imbalance.html"
fig.write_html(str(out), include_plotlyjs="inline", full_html=True)
print("wrote", out, "rows:", pdf.shape)
