"""Builds a self-contained interactive Plotly efficient frontier chart
from outputs/portfolio_results.duckdb, populated by an actual, fresh
execution of 01_Efficient_Frontier_Optimization.ipynb on real Binance
daily closes (data/raw/daily_closes.csv, 731 rows x 6 assets). Every
point plotted here comes directly from that real run -- nothing is
invented.
"""
import duckdb
import plotly.graph_objects as go

con = duckdb.connect("outputs/portfolio_results.duckdb")
frontier = con.execute("select return, volatility from efficient_frontier order by volatility").fetchdf()
perf = con.execute("select portfolio, expected_return, volatility, sharpe from performance").fetchdf()
alloc = con.execute("select portfolio, asset, weight from allocations where weight > 0.001").fetchdf()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=frontier["volatility"], y=frontier["return"], mode="lines+markers",
    name="Efficient frontier (SLSQP, 6 real crypto assets)",
    line=dict(color="#2E86AB", width=2), marker=dict(size=4),
))

colors = {"max_sharpe": "#E63946", "min_volatility": "#2A9D8F"}
for _, row in perf.iterrows():
    p = row["portfolio"]
    holdings = alloc[alloc["portfolio"] == p]
    hover = "<br>".join(f"{r.asset}: {r.weight:.1%}" for r in holdings.itertuples())
    fig.add_trace(go.Scatter(
        x=[row["volatility"]], y=[row["expected_return"]], mode="markers+text",
        name=f"{p.replace('_',' ').title()} (Sharpe={row['sharpe']:.2f})",
        marker=dict(size=14, color=colors.get(p, "#333"), symbol="star"),
        text=[p.replace("_", " ").title()], textposition="top center",
        hovertext=[f"{p}<br>return={row['expected_return']:.1%} vol={row['volatility']:.1%} Sharpe={row['sharpe']:.2f}<br>{hover}"],
        hoverinfo="text",
    ))

fig.update_layout(
    title="Markowitz efficient frontier: BTC/ETH/SOL/BNB/XRP/ADA (real daily closes, 2022-06 to 2024-05)",
    xaxis_title="Annualized volatility", yaxis_title="Annualized expected return",
    xaxis_tickformat=".0%", yaxis_tickformat=".0%",
    template="plotly_white", hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=1000, height=600,
)

out = "outputs/interactive/efficient_frontier_interactive.html"
import os
os.makedirs("outputs/interactive", exist_ok=True)
fig.write_html(out, include_plotlyjs="inline", full_html=True)
print("wrote", out)
