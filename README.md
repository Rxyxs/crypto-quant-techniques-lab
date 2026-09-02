[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Crypto Quant Techniques Lab

A single lab, eight standalone techniques applied to crypto market data — signal detection, market microstructure, portfolio construction, and NLP. Each technique lives in its own numbered folder with its own README, requirements, and tests, and can be run independently. This repo replaces eight separate single-technique repos that used to live on this profile; consolidating them here makes the actual point clearer: these are variations on a shared toolkit (statistical arbitrage, anomaly detection, supervised/unsupervised ML), not eight unrelated projects.

## Techniques

| # | Technique | Folder | What it does |
|---|---|---|---|
| 01 | Direction classification (deep learning) | [`01-direction-classification-deep-learning`](01-direction-classification-deep-learning) | Dense neural nets (ReLU vs. Tanh) classify next-candle direction on simulated AR(1)/GARCH OHLCV data and real Binance data. |
| 02 | Liquidity & price impact | [`02-liquidity-price-impact`](02-liquidity-price-impact) | XGBoost model of order-book price impact from synthetic depth/imbalance features. |
| 03 | Pairs trading (cointegration) | [`03-pairs-trading-cointegration`](03-pairs-trading-cointegration) | Statistical cointegration screening + Kalman-filtered dynamic hedge ratio for pairs trading. |
| 04 | Portfolio optimization (Markowitz) | [`04-portfolio-markowitz-optimization`](04-portfolio-markowitz-optimization) | Efficient-frontier portfolio construction across a crypto asset universe. |
| 05 | Regime detection & correlation | [`05-regime-detection-correlation`](05-regime-detection-correlation) | Markov regime-switching detection and its effect on cross-asset correlation. |
| 06 | Sentiment screening (NLP) | [`06-sentiment-nlp-screening`](06-sentiment-nlp-screening) | FinBERT sentiment on financial headlines vs. trading volume. |
| 07 | Order-book spoofing detection | [`07-orderbook-spoofing-detection`](07-orderbook-spoofing-detection) | Isolation Forest + autoencoder over streaming L2 order-book data, with alert-budget calibration. |
| 08 | Strategy backtesting | [`08-strategy-backtesting`](08-strategy-backtesting) | Statistical backtesting engine comparing trading signals under realistic frictions. |

## Sample results

One representative interactive chart per technique, generated from each folder's own real data/run (see each folder's README for full results and methodology):

| # | Technique | Interactive chart |
|---|---|---|
| 01 | Direction classification | [Walk-forward gross vs. cost-adjusted PnL](https://htmlpreview.github.io/?https://github.com/Rxyxs/crypto-quant-techniques-lab/blob/main/01-direction-classification-deep-learning/outputs/interactive/walkforward_pnl_interactive.html) |
| 02 | Liquidity & price impact | [Real BTCUSDT price vs. order-book/trade-flow imbalance](https://htmlpreview.github.io/?https://github.com/Rxyxs/crypto-quant-techniques-lab/blob/main/02-liquidity-price-impact/outputs/interactive/price_vs_liquidity_imbalance.html) |
| 03 | Pairs trading (cointegration) | [Cointegrated spread & rolling z-score](https://htmlpreview.github.io/?https://github.com/Rxyxs/crypto-quant-techniques-lab/blob/main/03-pairs-trading-cointegration/outputs/interactive/spread_zscore_interactive.html) |
| 04 | Portfolio optimization (Markowitz) | [Efficient frontier with max-Sharpe / min-vol portfolios](https://htmlpreview.github.io/?https://github.com/Rxyxs/crypto-quant-techniques-lab/blob/main/04-portfolio-markowitz-optimization/outputs/interactive/efficient_frontier_interactive.html) |
| 05 | Regime detection & correlation | [Regime-colored cumulative price path](https://htmlpreview.github.io/?https://github.com/Rxyxs/crypto-quant-techniques-lab/blob/main/05-regime-detection-correlation/outputs/interactive/regime_colored_price.html) |
| 06 | Sentiment screening (NLP) | [FinBERT sentiment score vs. trading volume](https://htmlpreview.github.io/?https://github.com/Rxyxs/crypto-quant-techniques-lab/blob/main/06-sentiment-nlp-screening/outputs/interactive/sentiment_vs_volume.html) |
| 07 | Order-book spoofing detection | [Spoofing alert timeline](https://htmlpreview.github.io/?https://github.com/Rxyxs/crypto-quant-techniques-lab/blob/main/07-orderbook-spoofing-detection/outputs/interactive/spoofing_alert_timeline.html) |
| 08 | Strategy backtesting | [Net-of-friction equity curves, 5 strategies](https://htmlpreview.github.io/?https://github.com/Rxyxs/crypto-quant-techniques-lab/blob/main/08-strategy-backtesting/outputs/interactive/equity_curves_comparison.html) |

## Why one repo instead of eight

Each technique is real, runnable, and independently tested (see each folder's own README for setup and results) — this isn't about hiding scope, it's about representing it accurately. Eight repos with the `crypto-` prefix read as eight unrelated projects; one lab folder with eight techniques reads as what it actually is: a systematic exploration of the same toolkit — anomaly detection, supervised/unsupervised ML, statistical time series — applied across different problems in the same market.

## Running a technique

Each folder is self-contained:

```bash
cd 0N-technique-name
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # Windows
python <entry_point>.py
```

See the folder's own README for the exact entry point, real results from an actual run, and any honest negative findings.

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Code: MIT — see [LICENSE](LICENSE)
