import datetime as dt
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import pipeline

SEED = 42
TICKERS = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "AVAX", "MATIC"]
N_HEADLINES = 240
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = Path("outputs")

# Each template's bucket is the intended ground-truth polarity, used only to
# validate FinBERT's own predictions against -- the model never sees these
# labels, they exist purely to check whether a real financial-sentiment
# model agrees with the polarity the headline was written to carry.
HEADLINE_TEMPLATES = {
    "positive": [
        "{ticker} surges {pct}% as institutional adoption accelerates",
        "{ticker} rallies after major exchange announces listing expansion",
        "Analysts raise {ticker} price targets following strong network growth",
        "{ticker} climbs on news of favorable regulatory clarity",
        "{ticker} hits new yearly high amid growing ETF inflows",
        "Whale accumulation lifts {ticker} sentiment as trading volume builds",
        "{ticker} breaks key resistance level as bulls take control",
        "Partnership announcement sends {ticker} sharply higher",
    ],
    "negative": [
        "{ticker} plunges {pct}% after exchange hack triggers panic selling",
        "Regulators announce crackdown as {ticker} sinks on compliance fears",
        "{ticker} tumbles after major holder liquidates position",
        "Network outage sparks {ticker} sell-off across major exchanges",
        "{ticker} slides after downgrade from institutional research desk",
        "Liquidity concerns weigh on {ticker}, price drops sharply",
        "{ticker} falls to multi-month low amid broader risk-off sentiment",
        "Security vulnerability disclosure sends {ticker} lower",
    ],
    "neutral": [
        "{ticker} trades sideways ahead of upcoming network upgrade",
        "Analysts maintain hold rating on {ticker} amid mixed signals",
        "{ticker} price stable as market awaits Federal Reserve decision",
        "Trading volume for {ticker} remains within recent average range",
        "{ticker} developers release routine software maintenance update",
        "{ticker} consolidates near current levels in a low-volatility session",
        "Market participants adopt wait-and-see approach on {ticker}",
        "{ticker} little changed as broader crypto market lacks clear direction",
    ],
}


def simulate_headlines(n, py_rng, np_rng):
    start = dt.date(2024, 1, 1)
    true_labels = list(HEADLINE_TEMPLATES.keys())
    rows = []
    for _ in range(n):
        true_label = py_rng.choice(true_labels)
        template = py_rng.choice(HEADLINE_TEMPLATES[true_label])
        ticker = py_rng.choice(TICKERS)
        pct = py_rng.randint(3, 22)
        headline = template.format(ticker=ticker, pct=pct)
        date = start + dt.timedelta(days=int(np_rng.integers(0, 300)))
        volume = simulate_volume(true_label, pct, np_rng)
        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "headline": headline,
                "true_label": true_label,
                "volume_usd_millions": volume,
            }
        )
    return rows


def simulate_volume(true_label, pct, np_rng):
    base = float(np_rng.lognormal(mean=15.5, sigma=0.35))
    if true_label == "neutral":
        return base
    intensity_boost = 1.0 + (pct / 100.0) * 3.0
    return base * intensity_boost


def classify_sentiment(headlines):
    classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None)
    return classifier(headlines, batch_size=16, truncation=True)


def scores_to_row(result):
    by_label = {entry["label"].lower(): entry["score"] for entry in result}
    predicted_label = max(by_label, key=by_label.get)
    sentiment_score = by_label.get("positive", 0.0) - by_label.get("negative", 0.0)
    return predicted_label, sentiment_score, by_label


def plot_sentiment_histogram(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["sentiment_score"].to_numpy(), bins=20, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Sentiment score (P(positive) - P(negative))")
    ax.set_ylabel("Number of headlines")
    ax.set_title(f"FinBERT sentiment distribution across {df.height} simulated crypto headlines")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_volume_by_sentiment_category(df, out_path):
    category_order = ["negative", "neutral", "positive"]
    colors = {"negative": "#C44E52", "neutral": "#8C8C8C", "positive": "#55A868"}

    summary = (
        df.group_by("predicted_label")
        .agg(
            [
                pl.col("volume_usd_millions").mean().alias("avg_volume"),
                pl.col("sentiment_score").mean().alias("avg_sentiment"),
                pl.len().alias("n_headlines"),
            ]
        )
    )
    summary = summary.filter(pl.col("predicted_label").is_in(category_order))
    order_map = {c: i for i, c in enumerate(category_order)}
    summary = summary.sort(pl.col("predicted_label").replace_strict(order_map))

    categories = summary["predicted_label"].to_list()
    avg_volume = summary["avg_volume"].to_numpy()
    n_headlines = summary["n_headlines"].to_list()

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        categories, avg_volume, color=[colors[c] for c in categories], alpha=0.85,
    )
    for bar, n in zip(bars, n_headlines):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"n={n}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_xlabel("FinBERT-predicted sentiment category")
    ax.set_ylabel("Avg. simulated trading volume (USD millions)")
    ax.set_title("Average trading volume by predicted sentiment category")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary


def summarize_by_ticker(df):
    return (
        df.group_by("ticker")
        .agg(
            [
                pl.col("sentiment_score").mean().alias("avg_sentiment"),
                pl.col("volume_usd_millions").mean().alias("avg_volume"),
                pl.len().alias("n_headlines"),
            ]
        )
        .sort("avg_sentiment", descending=True)
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    py_rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    rows = simulate_headlines(N_HEADLINES, py_rng, np_rng)
    headlines = [r["headline"] for r in rows]

    print(f"Simulated {len(rows)} financial news headlines across {len(TICKERS)} tickers.")
    print(f"Loading {MODEL_NAME} and classifying sentiment...")
    raw_results = classify_sentiment(headlines)

    predicted_labels, sentiment_scores, prob_rows = [], [], []
    for result in raw_results:
        predicted_label, sentiment_score, by_label = scores_to_row(result)
        predicted_labels.append(predicted_label)
        sentiment_scores.append(sentiment_score)
        prob_rows.append(by_label)

    df = pl.DataFrame(rows).with_columns(
        [
            pl.Series("predicted_label", predicted_labels),
            pl.Series("sentiment_score", sentiment_scores),
            pl.Series("prob_positive", [p.get("positive", 0.0) for p in prob_rows]),
            pl.Series("prob_negative", [p.get("negative", 0.0) for p in prob_rows]),
            pl.Series("prob_neutral", [p.get("neutral", 0.0) for p in prob_rows]),
        ]
    )

    true_labels = df["true_label"].to_list()
    acc = accuracy_score(true_labels, predicted_labels)
    labels_order = ["negative", "neutral", "positive"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels_order)

    print(f"\nFinBERT vs. template-intended polarity accuracy: {acc:.4f}")
    print(f"Confusion matrix (rows=true {labels_order}, cols=pred {labels_order}):")
    print(cm)

    print("\nPredicted sentiment label distribution:")
    print(df.group_by("predicted_label").agg(pl.len().alias("n")).sort("predicted_label"))

    df.write_csv(OUTPUT_DIR / "headline_sentiment.csv")
    print(f"\nSaved {OUTPUT_DIR / 'headline_sentiment.csv'}")

    plot_sentiment_histogram(df, OUTPUT_DIR / "sentiment_histogram.png")
    category_summary = plot_volume_by_sentiment_category(df, OUTPUT_DIR / "volume_by_sentiment_category.png")
    category_summary.write_csv(OUTPUT_DIR / "volume_by_sentiment_category.csv")

    ticker_summary = summarize_by_ticker(df)
    ticker_summary.write_csv(OUTPUT_DIR / "sentiment_volume_by_ticker.csv")

    print("\nAverage trading volume by predicted sentiment category:")
    print(category_summary)
    print("\nPer-ticker sentiment screener ranking:")
    print(ticker_summary)
    print(f"\nSaved figures and tables to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
