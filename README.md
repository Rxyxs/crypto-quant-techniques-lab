[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Crypto Quant Techniques Lab

[![CI](https://github.com/Rxyxs/crypto-quant-techniques-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Rxyxs/crypto-quant-techniques-lab/actions/workflows/ci.yml)

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

## Methodology: temporal integrity, risk-adjusted metrics, and frictions

Two techniques — 03 (pairs trading) and 08 (strategy backtesting) — went through a dedicated audit that the rest of this lab hasn't (yet). Each folder's own README has the full write-up; this is the summary that matters at the repo level:

- **Look-ahead bias, found and removed in 03.** The pipeline used to fit the pairs-trading hedge ratio (OLS) once on the *entire* price history and reuse that single beta to score the whole backtest — including days that, in wall-clock time, happened before the data behind that beta existed. Fixed by switching the live signal path (`run_full_pipeline`) to the Kalman filter's online beta (`kalman_pairs.py`), which re-estimates itself day by day from data available up to that day only — the static OLS fit is still computed and printed, but only as a labeled reference, never fed into the backtest. Verified with a truncation-invariance test, not just a code review: appending future rows to the price series cannot change a beta, spread, z-score, or position already computed for an earlier day (`03-pairs-trading-cointegration/tests/test_lookahead_audit.py`).
- **Sortino Ratio, added to both 03 and 08.** Downside-deviation-only risk adjustment (`sqrt(mean(min(return − MAR, 0)²))`, MAR = minimum acceptable return), tested to confirm it actually ignores upside dispersion the way Sharpe doesn't — two return series with identical downside and identical mean but different upside variance produce the same Sortino and different Sharpe — and to return `NaN`, never a division by zero or a misleading `0.0`, when a sample has no returns below the MAR.
- **Frictions, explicit in both, not identical.** Both charge the same 10 bps taker fee by default and neither ever fills at the close for free. `08` scales its slippage with realized volatility (`base_bps + vol_multiplier × recent_volatility`); `03` uses a static entry/exit slippage split instead — a real difference between the two engines, not full parity, documented as such rather than glossed over.

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

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs **7 jobs on every push and pull request** to `main` (`ubuntu-latest`, Python 3.10, pip cached per folder) — one job per technique that has a `tests/` directory. `06-sentiment-nlp-screening` doesn't have one yet, so it isn't in the matrix.

| Technique | Tests | Technique | Tests |
|---|---|---|---|
| 01 — Direction classification | 6 | 05 — Regime detection | 11 |
| 02 — Liquidity & price impact | 10 | 07 — Order-book spoofing | 21 |
| 03 — Pairs trading | 35 | 08 — Strategy backtesting | 29 |
| 04 — Portfolio optimization | 8 | **Total** | **120** |

`03` and `08` carry more than half that total (64 of 120) because they're the two techniques with the dedicated audit above — every truncation-invariance, Kalman-filter, and Sortino test from it runs here on every push, not just once locally.

## Production readiness checklist

What's actually verified as of this commit, closing this lab's third week of work — each row links to where it's checked, not just asserted:

| | Item | Evidence |
|---|---|---|
| ✅ | Multi-module automated CI (7/7 jobs on GitHub Actions, Python 3.10) | [Continuous integration](#continuous-integration) above; badge at the top of this page |
| ✅ | No look-ahead bias in `03` and `08` (verified via truncation-invariance tests) | `03-pairs-trading-cointegration/tests/test_lookahead_audit.py`, `08-strategy-backtesting/tests/test_lookahead_and_metrics_audit.py` — scoped to these two; the other six techniques haven't had this specific audit yet |
| ✅ | Risk-adjusted metrics: Sharpe (`03`, `04`, `08`) and Sortino (`03`, `08`) implemented | `pairs_trading_engine.py::sortino_ratio`, `backtest_engine.py::sortino_ratio`, `04`'s own Sharpe on its efficient frontier |
| ✅ | Real market frictions in `03` and `08` (taker/maker fees + slippage) | `frictions.py` (dynamic, volatility-scaled) and `pairs_trading_engine.py::compute_transaction_cost_returns` (static entry/exit) — see the methodology note above on the real difference between the two |
| ✅ | Dynamic pairs-trading pipeline (online Kalman filter) | `run_full_pipeline` in `03` now drives its backtest from `kalman_pairs.py`'s beta, not a full-sample static OLS fit |
| ✅ | DuckDB persistence and real Binance data, across most of the lab | DuckDB in all 8 technique folders; real Binance downloads in `01`, `02`, `03`, `04`, `08` — no production serving layer exists here, this is persistence and real market data, not a deployed service |

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Code: MIT — see [LICENSE](LICENSE)
