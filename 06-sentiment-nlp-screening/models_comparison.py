"""Complementary sentiment-classification approaches, benchmarked against
the FinBERT predictions produced by ``sentiment_screener.py``.

FinBERT (see ``sentiment_screener.py``) is a pretrained transformer used
purely for inference -- it is never trained on this dataset. This module
adds three approaches that *are* trained directly on the simulated
headlines (against the same ``true_label`` ground truth FinBERT is
checked against), so the four can be compared on equal footing:

1. ``train_baseline``   -- TF-IDF + Logistic Regression (interpretable).
2. ``train_ensemble``   -- TF-IDF + Random Forest (tree ensemble).
3. ``train_mlp``        -- PyTorch MLP over TF-IDF features, with a
   custom focal-style loss and a comparison across ReLU / GELU / Swish
   (SiLU) activations.

Outputs:
- ``outputs/model_comparison.png``       -- accuracy bar chart, 4 models.
- ``outputs/confusion_matrices.png``     -- confusion matrix grid.
- ``outputs/activation_loss_curves.png`` -- MLP training loss per activation.
- ``outputs/model_comparison.duckdb``    -- metrics + predictions tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
LABELS = ["negative", "neutral", "positive"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
OUTPUT_DIR = Path("outputs")
DB_PATH = OUTPUT_DIR / "model_comparison.duckdb"
ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "Swish": nn.SiLU,  # SiLU == Swish (x * sigmoid(x))
}


def load_headline_data(csv_path=OUTPUT_DIR / "headline_sentiment.csv"):
    """Load the headlines + FinBERT predictions produced by sentiment_screener.py."""
    df = pl.read_csv(csv_path)
    return df


def split_data(df, test_size=0.25, seed=SEED):
    headlines = df["headline"].to_list()
    true_labels = df["true_label"].to_list()
    idx = np.arange(len(headlines))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=true_labels
    )
    return idx_train, idx_test, headlines, true_labels


def vectorize(headlines, idx_train, idx_test):
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=1)
    train_texts = [headlines[i] for i in idx_train]
    test_texts = [headlines[i] for i in idx_test]
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {"accuracy": float(acc), "f1_macro": float(f1), "confusion_matrix": cm}


def train_baseline(X_train, y_train, X_test, y_test):
    """Interpretable baseline: TF-IDF + multinomial Logistic Regression."""
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return clf, preds, evaluate(y_test, preds)


def train_ensemble(X_train, y_train, X_test, y_test):
    """Tree ensemble over the same TF-IDF text features."""
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return clf, preds, evaluate(y_test, preds)


class SentimentMLP(nn.Module):
    """Small feed-forward classification head over dense TF-IDF features."""

    def __init__(self, input_dim, hidden_dim=64, n_classes=3, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    """Custom loss: focal loss (Lin et al. 2017), down-weights easy examples
    so the MLP spends more gradient on headlines it is unsure about --
    relevant here since the neutral class is easiest and can dominate a
    plain cross-entropy loss."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


def train_mlp(
    X_train, y_train, X_test, y_test, activation_name="ReLU", epochs=60, lr=0.01, seed=SEED,
):
    """Train the MLP with the custom focal loss for one activation choice."""
    torch.manual_seed(seed)
    activation = ACTIVATIONS[activation_name]

    X_train_dense = torch.tensor(np.asarray(X_train.todense()), dtype=torch.float32)
    X_test_dense = torch.tensor(np.asarray(X_test.todense()), dtype=torch.float32)
    y_train_idx = torch.tensor([LABEL_TO_IDX[y] for y in y_train], dtype=torch.long)
    y_test_idx = torch.tensor([LABEL_TO_IDX[y] for y in y_test], dtype=torch.long)

    model = SentimentMLP(input_dim=X_train_dense.shape[1], activation=activation)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_dense)
        loss = criterion(logits, y_train_idx)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_dense)
        pred_idx = test_logits.argmax(dim=1).numpy()
    preds = [LABELS[i] for i in pred_idx]

    metrics = evaluate(y_test, preds)
    metrics["loss_history"] = loss_history
    return model, preds, metrics


def compare_activations(X_train, y_train, X_test, y_test, epochs=60):
    """Train the MLP once per activation function, return metrics + curves for each."""
    results = {}
    for name in ACTIVATIONS:
        _, preds, metrics = train_mlp(
            X_train, y_train, X_test, y_test, activation_name=name, epochs=epochs
        )
        results[name] = {"preds": preds, "metrics": metrics}
    return results


def plot_model_comparison(comparison, out_path):
    names = list(comparison.keys())
    accuracies = [comparison[n]["accuracy"] for n in names]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(names, accuracies, color=colors[: len(names)], alpha=0.88)
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{acc:.3f}", ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Accuracy vs. template-intended polarity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sentiment classifier comparison on held-out headlines")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(cms_by_model, out_path):
    n = len(cms_by_model)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms_by_model.items()):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_yticklabels(LABELS)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_activation_loss_curves(activation_results, out_path):
    colors = {"ReLU": "#4C72B0", "GELU": "#55A868", "Swish": "#C44E52"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, result in activation_results.items():
        ax.plot(
            result["metrics"]["loss_history"], label=name,
            color=colors.get(name, None), linewidth=1.8,
        )
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Focal loss (train)")
    ax.set_title("MLP training loss by activation function (ReLU / GELU / Swish)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def animate_activation_loss_curves(activation_results, out_path):
    """Racing line-chart GIF of the same real MLP training loss curves used
    in ``plot_activation_loss_curves`` -- subsampled to at most 60 frames,
    never fabricated. Uses matplotlib.animation.FuncAnimation + Pillow."""
    import matplotlib.animation as animation

    colors = {"ReLU": "#00d4ff", "GELU": "#7cff6b", "Swish": "#ff6ec7"}
    names = list(activation_results.keys())
    series = {name: activation_results[name]["metrics"]["loss_history"] for name in names}
    n_points = len(next(iter(series.values())))

    max_frames = 60
    if n_points > max_frames:
        frame_idx = np.linspace(0, n_points - 1, max_frames).astype(int)
        frame_idx = sorted(set(frame_idx.tolist()))
    else:
        frame_idx = list(range(n_points))
    n_frames = len(frame_idx)

    x_full = np.arange(n_points)
    y_max = max(max(vals) for vals in series.values())
    y_min = min(min(vals) for vals in series.values())

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        lines = {name: ax.plot([], [], color=colors.get(name), linewidth=2.2, label=name)[0] for name in names}
        labels = {
            name: ax.annotate(
                "", xy=(0, 0), xytext=(12, 0), textcoords="offset points",
                fontsize=10, color="white", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc=colors.get(name), ec="none", alpha=0.85),
            )
            for name in names
        }

        ax.set_xlim(0, max(n_points - 1, 1))
        pad = (y_max - y_min) * 0.1 + 1e-6
        ax.set_ylim(max(y_min - pad, 0), y_max + pad)
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Focal loss (train)")
        ax.set_title("MLP training loss by activation function (ReLU / GELU / Swish)")
        ax.legend(loc="upper right")
        fig.tight_layout()

        def update(frame):
            cutoff = frame_idx[frame]
            for name in names:
                y_vals = series[name][: cutoff + 1]
                lines[name].set_data(x_full[: cutoff + 1], y_vals)
                current_y = y_vals[-1]
                labels[name].xy = (cutoff, current_y)
                labels[name].set_text(f"{name}: {current_y:.3f}")
            return list(lines.values()) + list(labels.values())

        ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=120, blit=False)
        ani.save(out_path, writer="pillow")
        plt.close(fig)


def persist_to_duckdb(comparison, predictions_df, db_path=DB_PATH):
    """Persist per-model metrics and per-headline predictions to a local DuckDB file."""
    con = duckdb.connect(str(db_path))
    metrics_rows = [
        {"model": name, "accuracy": vals["accuracy"], "f1_macro": vals["f1_macro"]}
        for name, vals in comparison.items()
    ]
    metrics_df = pl.DataFrame(metrics_rows)
    metrics_records = metrics_df.to_dicts()
    con.execute("CREATE OR REPLACE TABLE model_metrics (model VARCHAR, accuracy DOUBLE, f1_macro DOUBLE)")
    con.executemany(
        "INSERT INTO model_metrics VALUES (?, ?, ?)",
        [(r["model"], r["accuracy"], r["f1_macro"]) for r in metrics_records],
    )

    pred_records = predictions_df.to_dicts()
    pred_cols = predictions_df.columns
    con.execute(
        f"CREATE OR REPLACE TABLE model_predictions ({', '.join(c + ' VARCHAR' for c in pred_cols)})"
    )
    con.executemany(
        f"INSERT INTO model_predictions VALUES ({', '.join(['?'] * len(pred_cols))})",
        [tuple(r[c] for c in pred_cols) for r in pred_records],
    )
    con.close()
    return metrics_df


def run_comparison(csv_path=OUTPUT_DIR / "headline_sentiment.csv", epochs=60):
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_headline_data(csv_path)
    idx_train, idx_test, headlines, true_labels = split_data(df)
    vectorizer, X_train, X_test = vectorize(headlines, idx_train, idx_test)
    y_train = [true_labels[i] for i in idx_train]
    y_test = [true_labels[i] for i in idx_test]

    _, baseline_preds, baseline_metrics = train_baseline(X_train, y_train, X_test, y_test)
    _, ensemble_preds, ensemble_metrics = train_ensemble(X_train, y_train, X_test, y_test)
    activation_results = compare_activations(X_train, y_train, X_test, y_test, epochs=epochs)
    best_activation = max(activation_results, key=lambda k: activation_results[k]["metrics"]["accuracy"])
    mlp_preds = activation_results[best_activation]["preds"]
    mlp_metrics = activation_results[best_activation]["metrics"]

    finbert_test_labels = df["predicted_label"].to_list()
    finbert_preds = [finbert_test_labels[i] for i in idx_test]
    finbert_metrics = evaluate(y_test, finbert_preds)

    comparison = {
        "FinBERT (pretrained)": finbert_metrics,
        "Logistic Regression (TF-IDF)": baseline_metrics,
        "Random Forest (TF-IDF)": ensemble_metrics,
        f"PyTorch MLP ({best_activation})": mlp_metrics,
    }

    plot_model_comparison(comparison, OUTPUT_DIR / "model_comparison.png")
    plot_confusion_matrices(
        {name: vals["confusion_matrix"] for name, vals in comparison.items()},
        OUTPUT_DIR / "confusion_matrices.png",
    )
    plot_activation_loss_curves(activation_results, OUTPUT_DIR / "activation_loss_curves.png")
    animate_activation_loss_curves(activation_results, OUTPUT_DIR / "activation_loss_curves_animated.gif")

    predictions_df = pl.DataFrame(
        {
            "headline": [headlines[i] for i in idx_test],
            "true_label": y_test,
            "finbert_pred": finbert_preds,
            "logreg_pred": baseline_preds,
            "rf_pred": ensemble_preds,
            f"mlp_{best_activation.lower()}_pred": mlp_preds,
        }
    )
    metrics_df = persist_to_duckdb(comparison, predictions_df)

    metrics_summary = {
        name: {"accuracy": vals["accuracy"], "f1_macro": vals["f1_macro"]}
        for name, vals in comparison.items()
    }
    with open(OUTPUT_DIR / "model_comparison_metrics.json", "w") as f:
        json.dump(
            {"best_mlp_activation": best_activation, "metrics": metrics_summary},
            f, indent=2,
        )

    print("Model comparison (accuracy / macro-F1):")
    for name, vals in comparison.items():
        print(f"  {name:32s} acc={vals['accuracy']:.4f}  f1_macro={vals['f1_macro']:.4f}")
    print(f"\nBest MLP activation: {best_activation}")
    print(f"Saved plots, metrics JSON, and {DB_PATH} to {OUTPUT_DIR}/")
    return comparison, metrics_df


if __name__ == "__main__":
    run_comparison()
