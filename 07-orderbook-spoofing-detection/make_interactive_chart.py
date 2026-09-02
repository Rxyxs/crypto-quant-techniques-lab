"""Builds a self-contained interactive Plotly alert timeline from a fresh
run of detect_spoofing.py's Isolation Forest pipeline (outputs/
orderbook_snapshots_scored.csv, regenerated in this update: 5,000
snapshots, 85 true spoofing events, Isolation Forest precision=0.64,
recall=0.75, F1=0.69 -- see terminal output / README for the exact run).
"""
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("outputs/orderbook_snapshots_scored.csv")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["snapshot_id"], y=df["anomaly_score"], mode="lines",
    name="Isolation Forest anomaly score", line=dict(color="#457B9D", width=1),
    opacity=0.7,
))

true_spoof = df[df["true_is_spoof"]]
fig.add_trace(go.Scatter(
    x=true_spoof["snapshot_id"], y=true_spoof["anomaly_score"], mode="markers",
    name="True spoofing event", marker=dict(color="#E63946", size=7, symbol="x"),
))

flagged = df[df["is_anomaly"] & ~df["true_is_spoof"]]
fig.add_trace(go.Scatter(
    x=flagged["snapshot_id"], y=flagged["anomaly_score"], mode="markers",
    name="False-positive alert", marker=dict(color="#F4A261", size=7, symbol="circle-open"),
))

caught = df[df["is_anomaly"] & df["true_is_spoof"]]
fig.add_trace(go.Scatter(
    x=caught["snapshot_id"], y=caught["anomaly_score"], mode="markers",
    name="True positive alert", marker=dict(color="#2A9D8F", size=9, symbol="star"),
))

fig.update_layout(
    title="Order-book spoofing alert timeline: Isolation Forest anomaly score per snapshot (5,000 snapshots, 85 true events, precision=0.64, recall=0.75)",
    xaxis_title="Snapshot index (time-ordered)", yaxis_title="Anomaly score",
    template="plotly_white", hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=1100, height=550,
)

import os
os.makedirs("outputs/interactive", exist_ok=True)
out = "outputs/interactive/spoofing_alert_timeline.html"
fig.write_html(out, include_plotlyjs="inline", full_html=True)
print("wrote", out, "rows:", len(df))
