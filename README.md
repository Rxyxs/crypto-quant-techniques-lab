[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Crypto Quant Techniques Lab

[![CI](https://github.com/Rxyxs/crypto-quant-techniques-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Rxyxs/crypto-quant-techniques-lab/actions/workflows/ci.yml)

A single lab, eight standalone techniques applied to crypto market data — signal detection, market microstructure, portfolio construction, and NLP. Each technique lives in its own numbered folder with its own README, requirements, and tests, and can be run independently. This repo replaces eight separate single-technique repos that used to live on this profile; consolidating them here makes the actual point clearer: these are variations on a shared toolkit (statistical arbitrage, anomaly detection, supervised/unsupervised ML), not eight unrelated projects.

## Techniques

| # | Technique | Folder | What it does | Entry point |
|---|---|---|---|---|
| 01 | Direction classification (deep learning) | [`01-direction-classification-deep-learning`](01-direction-classification-deep-learning) | Dense neural nets (ReLU vs. Tanh) classify next-candle direction on simulated AR(1)/GARCH OHLCV data and real Binance data. | `python train_classifier.py` |
| 02 | Liquidity & price impact | [`02-liquidity-price-impact`](02-liquidity-price-impact) | XGBoost model of order-book price impact from synthetic depth/imbalance features. | `python xgboost_impact.py` |
| 03 | Pairs trading (cointegration) | [`03-pairs-trading-cointegration`](03-pairs-trading-cointegration) | Statistical cointegration screening + Kalman-filtered dynamic hedge ratio for pairs trading. | `python pairs_trading_engine.py` |
| 04 | Portfolio optimization (Markowitz) | [`04-portfolio-markowitz-optimization`](04-portfolio-markowitz-optimization) | Efficient-frontier portfolio construction across a crypto asset universe. | notebook-driven — `portfolio_optimizer.py` is imported, not run standalone; see the folder's README |
| 05 | Regime detection & correlation | [`05-regime-detection-correlation`](05-regime-detection-correlation) | Markov regime-switching detection and its effect on cross-asset correlation. | `python market_analysis.py` |
| 06 | Sentiment screening (NLP) | [`06-sentiment-nlp-screening`](06-sentiment-nlp-screening) | FinBERT sentiment on financial headlines vs. trading volume. | `python sentiment_screener.py` |
| 07 | Order-book spoofing detection | [`07-orderbook-spoofing-detection`](07-orderbook-spoofing-detection) | Isolation Forest + autoencoder over streaming L2 order-book data, with alert-budget calibration. | `python detect_spoofing.py` |
| 08 | Strategy backtesting | [`08-strategy-backtesting`](08-strategy-backtesting) | Statistical backtesting engine comparing trading signals under realistic frictions. | `python backtest_engine.py` |

## What the eight techniques found

Every number comes from an actual run of that folder's pipeline. Read the right-hand column before the middle one: **most of these results are negative, and they stayed in.**

| # | Technique | Headline number | What it actually says |
|---|---|---|---|
| **01** | Direction classification | Real BTCUSDT accuracy **0.519**, ROC-AUC 0.536 | Barely above the 50% coin flip. On synthetic data with an injected AR(1) signal it reaches 0.596 — the gap between the two *is* the finding. The project treats **>90% accuracy as a leakage red flag, not a discovery** |
| **02** | Liquidity & price impact | R² **0.100** at a 1-minute horizon, 5.2% RMSE reduction | Genuine but small skill, and it evaporates fast: at 5 minutes R² is −0.008 and at 15 minutes −0.068 — *worse than a naive forecast*. The horizon, not the model, decides whether there is anything to predict |
| **03** | Pairs trading (cointegration) | Kalman **−12.2%** vs. static OLS **−43.7%** net | Both lose money. The dynamic hedge ratio cuts the loss 3.6x and the max drawdown from −51.6% to −21.4%, on 13 trades instead of 21. An improvement that is still a loss is reported as exactly that |
| **04** | Portfolio optimization | Max-Sharpe **0.720**, allocating 74.6% BTC / 23.2% SOL / 2.2% BNB | A real, lopsided allocation from 24 months of actual Binance history — not the tidy diversified pie a textbook example produces |
| **05** | Regime detection | Pairwise correlation **0.42 → 0.58** pooled (1.37x), **0.29 → 0.56** rolling (1.90x) | Diversification degrades exactly when it is needed. Two statistics, two magnitudes — [§6.3](05-regime-detection-correlation#63-correlation-structure-by-regime) explains why they differ instead of quoting whichever is larger |
| **06** | Sentiment screening (NLP) | FinBERT recall: **100%** negative, 78.6% positive, **32.5% neutral** | The model pushes neutral headlines into the polar classes. A blind spot surfaced by measuring per-class recall rather than reporting one headline accuracy number |
| **07** | Order-book spoofing | Precision **0.92** at a 0.5% alert budget vs. **0.64** at default 2% contamination | The one unambiguously positive result. Precision more than doubles by calibrating to what one analyst can actually review — 25 alerts a day, not a library default |
| **08** | Strategy backtesting | Best net Sharpe: **SMA Crossover at −0.358**; LightGBM worst at −0.93 | All five strategies lose money net of frictions, and **the plainest rule beats gradient boosting**. LightGBM's higher turnover burns 27.28% of its return on frictions |

**A signal accuracy of 0.507 still produced a net-negative Sharpe** (08). That single line is the lab's thesis: accuracy does not imply profitability once fees and slippage are charged.

---

## Evidence

### Five strategies, five negative Sharpes, and the simplest one wins

![Net Sharpe and equity curves, five strategies](08-strategy-backtesting/outputs/comparison_dashboard.png)

**How to read it.** Left: annualized Sharpe **net of fees and slippage**, one bar per strategy — note that the axis runs from −0.8 to 0, so every bar is a loss and a *shorter* bar is better. Right: the equity curves behind those bars, all starting at $1.

The ordering is the finding. **LightGBM, the only machine-learning model in the comparison, is the worst of the five** (−0.93, ending near $0.53), while a 10/50 simple moving-average crossover is the least bad (−0.358, ending near $0.84). The mechanism is in the frictions: LightGBM trades more, and a higher turnover rate burns 27.28% of its gross return before the signal quality ever gets to matter.

### An improvement that is still a loss

![Kalman vs. static OLS equity curve, net of costs](03-pairs-trading-cointegration/outputs/figures/kalman_vs_ols_equity_curve.png)

**How to read it.** Both curves are net of transaction costs and start at 1.0. Red is the static OLS hedge ratio fitted once over the whole sample; green is the Kalman filter's online beta, re-estimated day by day from data available up to that day only.

Red collapses to ~0.5 within two months and never recovers, ending at −43.7%. Green tracks near 1.0 for most of the window and ends at −12.2%. The dynamic estimator is unambiguously better — 3.6x smaller loss, drawdown cut from −51.6% to −21.4% — **and it still loses money.** Presenting it as a win because it beat the alternative would be the easy read; the honest one is that the strategy does not work in this window and the hedge ratio was never the binding problem.

This figure exists because of an audit: the original pipeline fitted OLS on the *entire* price history and reused that beta to score days that happened before the data behind it existed. That look-ahead bias is now caught by a truncation-invariance test that runs in CI.

### Diversification collapses when it is needed

![Correlation by regime](05-regime-detection-correlation/outputs/correlation_heatmap_by_regime.png)

**How to read it.** One correlation matrix per discovered regime, over the same eight assets. The regimes are **unsupervised** — no "this day was a crash" label was used; k=2 was chosen by silhouette score over k=2..6.

Every pair is redder on the left. Pooled across each regime's days, the off-diagonal mean rises from **0.419 in Bull Quiet to 0.576 in Bear Crash**. A portfolio sized on a single static correlation matrix is carrying more concentrated risk than its own risk model assumes, precisely in the 389 days when that matters.

---

## The pattern across all eight

Eight techniques, built independently, on the same market. They converge on one uncomfortable conclusion:

> **The edge mostly isn't there — and the honest way to show that is to keep the negative results.**

- **Frictions decide the ranking, not model quality.** In 08 the gradient-boosting model finishes last and the simplest technical rule finishes first, entirely because of turnover. In 03 the gross loss of −5.9% becomes −12.2% net.
- **Accuracy is not profitability.** 0.507 signal accuracy in 08, net-negative Sharpe. 0.519 accuracy on real BTCUSDT in 01, against a 0.596 on synthetic data with a signal deliberately injected.
- **The horizon decides whether anything is predictable at all.** In 02, R² is 0.100 at one minute and *negative* at five and fifteen — the same model, the same features.
- **Calibration beats defaults.** In 07, moving from a library's default 2% contamination to a 0.5% budget matched to one analyst's actual review capacity takes precision from 0.64 to 0.92.
- **A high number is treated as a symptom.** 01 states outright that >90% accuracy on this problem would be read as a leakage red flag rather than a result.

The two techniques that went through a dedicated temporal-integrity audit (03 and 08) carry 64 of the lab's 120 tests, because every truncation-invariance and Sortino check from that audit runs in CI on every push.

---

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
python <entry_point>.py                        # exact command per technique in the table above
```

See the folder's own README for real results from an actual run and any honest negative findings.

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
