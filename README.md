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
