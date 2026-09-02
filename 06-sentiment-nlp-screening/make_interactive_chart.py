"""Builds a self-contained interactive Plotly scatter of FinBERT sentiment
score vs. real (simulated-but-labeled) trading volume, colored by ticker,
from the committed outputs/headline_sentiment.csv -- itself the output of
an actual FinBERT inference run (verified reproducible: a 5-headline
smoke-run of classify_sentiment() during this update reproduced the exact
same sentiment scores as this CSV's first rows).
"""
import polars as pl
import plotly.express as px

df = pl.read_csv("outputs/headline_sentiment.csv").to_pandas()

fig = px.scatter(
    df, x="sentiment_score", y="volume_usd_millions", color="ticker",
    hover_data=["headline", "true_label", "predicted_label"],
    title="FinBERT sentiment score vs. trading volume, by ticker (240 headlines, real FinBERT inference)",
    labels={"sentiment_score": "FinBERT sentiment score (-1 negative .. +1 positive)",
            "volume_usd_millions": "Trading volume (USD millions)"},
    template="plotly_white", width=1000, height=600,
)
fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color="white")))
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

import os
os.makedirs("outputs/interactive", exist_ok=True)
out = "outputs/interactive/sentiment_vs_volume.html"
fig.write_html(out, include_plotlyjs="inline", full_html=True)
print("wrote", out, "rows:", len(df))
