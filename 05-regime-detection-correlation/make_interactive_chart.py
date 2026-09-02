"""Builds a self-contained interactive Plotly chart of a regime-colored
cumulative price series (BTC leg of the simulated multi-asset universe),
using the actual cluster labels persisted by a fresh run of
market_analysis.py to outputs/market_regimes.duckdb (KMeans, k=2,
silhouette=0.4926 -- see README §7 for the full real run log). Regimes
are unsupervised-detected, not hand-labeled.
"""
import duckdb
import numpy as np
import plotly.graph_objects as go

con = duckdb.connect("outputs/market_regimes.duckdb")
df = con.execute("select date, BTC, cluster from regime_days order by date").fetchdf()
summary = con.execute("select cluster, regime_label from regime_summary").fetchdf()
label_map = dict(zip(summary["cluster"], summary["regime_label"]))

df["cum_btc"] = (1 + df["BTC"]).cumprod()

colors = {0: "#E63946", 1: "#2A9D8F", 2: "#457B9D", 3: "#F4A261", 4: "#8E44AD"}
fig = go.Figure()

# Draw one continuous line, but color-code by adding a scatter segment per
# contiguous regime run so the legend groups by regime label.
df["regime_change"] = (df["cluster"] != df["cluster"].shift()).cumsum()
seen_labels = set()
for _, seg in df.groupby("regime_change"):
    c = seg["cluster"].iloc[0]
    label = label_map.get(c, f"cluster {c}")
    show = label not in seen_labels
    seen_labels.add(label)
    fig.add_trace(go.Scatter(
        x=seg["date"], y=seg["cum_btc"], mode="lines",
        line=dict(color=colors.get(c, "#333"), width=2),
        name=label, legendgroup=label, showlegend=show,
        hovertemplate="%{x}<br>cum. BTC leg=%{y:.2f}<extra>" + label + "</extra>",
    ))

fig.update_layout(
    title="Regime-colored cumulative price path (BTC leg), unsupervised KMeans regime detection (k=2, silhouette=0.493)",
    xaxis_title="Date", yaxis_title="Cumulative return (growth of 1)",
    template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=1050, height=550,
)

import os
os.makedirs("outputs/interactive", exist_ok=True)
out = "outputs/interactive/regime_colored_price.html"
fig.write_html(out, include_plotlyjs="inline", full_html=True)
print("wrote", out, "rows:", len(df))
