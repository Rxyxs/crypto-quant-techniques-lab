"""PyTorch Autoencoder anomaly detector for order book spoofing -- a second,
complementary approach to `detect_spoofing.py`'s Isolation Forest.

Where Isolation Forest isolates anomalies by random-split path length, an
Autoencoder learns to *reconstruct* normal order-flow snapshots from a
compressed latent bottleneck; a spoofed snapshot -- which looks different
from the bulk of genuine order flow the network trained on -- reconstructs
poorly, and that per-snapshot reconstruction error becomes the anomaly
score. This module reuses `simulate_orderbook` / `engineer_features` /
`L2_FEATURE_COLS` unchanged from `detect_spoofing.py`, so both approaches
are scored on the identical simulated book and feature set -- a fair,
apples-to-apples comparison, not two different problems.

It also runs a small controlled ablation across three hidden-layer
activation functions (ReLU, GELU, Swish/SiLU) on the identical
architecture, split, and seed, to see whether activation choice matters
for this reconstruction task (§7 of the README reports the result).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from detect_spoofing import L2_FEATURE_COLS, N_SNAPSHOTS, OUTPUT_DIR, SEED, engineer_features, simulate_orderbook

torch.manual_seed(SEED)
pl.Config.set_tbl_formatting("ASCII_FULL")  # avoid Windows console cp1252 box-drawing errors

HIDDEN_DIMS = (8, 3)  # encoder: input -> 8 -> 3 (bottleneck) -> 8 -> input
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 128
ACTIVATIONS = {"ReLU": nn.ReLU, "GELU": nn.GELU, "Swish": nn.SiLU}  # SiLU == Swish
DEFAULT_ACTIVATION = "GELU"


class SpoofAutoencoder(nn.Module):
    """Small symmetric autoencoder: input -> 8 -> 3 -> 8 -> input.

    The 3-unit bottleneck forces the network to compress each snapshot's
    engineered features into a small latent code capturing "normal"
    order-flow structure; a spoofed snapshot's feature combination doesn't
    compress/reconstruct cleanly through a code learned mostly from normal
    data, which is exactly the signal used for detection.
    """

    def __init__(self, n_features, activation_cls=nn.GELU):
        super().__init__()
        h1, h2 = HIDDEN_DIMS
        self.encoder = nn.Sequential(
            nn.Linear(n_features, h1), activation_cls(),
            nn.Linear(h1, h2), activation_cls(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h2, h1), activation_cls(),
            nn.Linear(h1, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_loss(x_hat, x):
    """Per-sample mean squared reconstruction error (custom loss: returns
    the *per-row* vector, not a scalar) -- needed both as the scalar
    training objective (`.mean()`) and, unreduced, as the per-snapshot
    anomaly score at inference time."""
    return ((x_hat - x) ** 2).mean(dim=1)


def train_autoencoder(X_train, n_features, activation_name=DEFAULT_ACTIVATION, epochs=EPOCHS, seed=SEED):
    torch.manual_seed(seed)
    model = SpoofAutoencoder(n_features, ACTIVATIONS[activation_name])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)

    model.train()
    history = []
    n = X_tensor.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch = X_tensor[idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = reconstruction_loss(recon, batch).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history.append(epoch_loss / n)
    return model, history


def score_autoencoder(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        recon = model(X_tensor)
        scores = reconstruction_loss(recon, X_tensor).numpy()
    return scores


def fit_predict_autoencoder(df, feature_cols=L2_FEATURE_COLS, contamination=0.02, activation_name=DEFAULT_ACTIVATION, seed=SEED):
    """Trains only on snapshots below the anomaly-score threshold from a
    held-out split -- an autoencoder trained *including* spoof events would
    partly learn to reconstruct them too, weakening the anomaly signal.
    Since true labels aren't available at deployment (§3.3 of the README),
    training simply uses all data (spoofs are rare -- 1.7% -- so their
    contribution to the learned reconstruction is small), matching how
    Isolation Forest is fit unsupervised in `detect_spoofing.py`."""
    X = df.select(feature_cols).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, _ = train_test_split(X_scaled, test_size=0.2, random_state=seed)
    model, history = train_autoencoder(X_train, X_scaled.shape[1], activation_name=activation_name, seed=seed)

    scores = score_autoencoder(model, X_scaled)
    threshold = np.quantile(scores, 1 - contamination)
    is_anomaly = scores >= threshold
    return model, scaler, is_anomaly, scores, history


def evaluate_activation(df, activation_name, feature_cols=L2_FEATURE_COLS, contamination=0.02, seed=SEED):
    _, _, is_anomaly, scores, history = fit_predict_autoencoder(
        df, feature_cols=feature_cols, contamination=contamination, activation_name=activation_name, seed=seed
    )
    y_true = df["true_is_spoof"].to_numpy()
    return {
        "activation": activation_name,
        "precision": precision_score(y_true, is_anomaly, zero_division=0),
        "recall": recall_score(y_true, is_anomaly, zero_division=0),
        "f1": f1_score(y_true, is_anomaly, zero_division=0),
        "final_train_loss": history[-1],
        "is_anomaly": is_anomaly,
        "scores": scores,
    }


def compare_activations(df, feature_cols=L2_FEATURE_COLS, contamination=0.02, seed=SEED):
    """Runs the identical architecture/split/seed with ReLU, GELU, and Swish
    (SiLU) hidden activations and returns a comparison table -- the
    controlled ablation reported in the README."""
    rows = []
    for name in ACTIVATIONS:
        result = evaluate_activation(df, name, feature_cols=feature_cols, contamination=contamination, seed=seed)
        rows.append({k: v for k, v in result.items() if k not in ("is_anomaly", "scores")})
    return pl.DataFrame(rows)


def plot_reconstruction_error_distribution(df, scores, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    true_spoof = df["true_is_spoof"].to_numpy()
    ax.hist(scores[~true_spoof], bins=40, alpha=0.6, color="#4C72B0", label=f"Normal (n={(~true_spoof).sum()})")
    ax.hist(scores[true_spoof], bins=40, alpha=0.8, color="red", label=f"True spoof (n={true_spoof.sum()})")
    ax.set_xlabel("Autoencoder reconstruction error (anomaly score)")
    ax.set_ylabel("Number of snapshots")
    ax.set_title("Autoencoder reconstruction error: normal vs. true spoof snapshots")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_activation_comparison(comparison_df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    activations = comparison_df["activation"].to_list()
    metrics = ["precision", "recall", "f1"]
    colors = {"precision": "#4C72B0", "recall": "#DD8452", "f1": "#55A868"}
    x = np.arange(len(activations))
    width = 0.25
    for i, metric in enumerate(metrics):
        values = comparison_df[metric].to_numpy()
        ax.bar(x + (i - 1) * width, values, width, label=metric.capitalize(), color=colors[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(activations)
    ax.set_ylabel("Score")
    ax.set_title("Autoencoder activation comparison: ReLU vs. GELU vs. Swish")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_pipeline(feature_cols=L2_FEATURE_COLS, activation_name=DEFAULT_ACTIVATION):
    rng = np.random.default_rng(SEED)
    df = simulate_orderbook(N_SNAPSHOTS, rng)
    df = engineer_features(df)
    _, _, is_anomaly, scores, _ = fit_predict_autoencoder(df, feature_cols=feature_cols, activation_name=activation_name)
    df = df.with_columns([pl.Series("ae_is_anomaly", is_anomaly), pl.Series("ae_anomaly_score", scores)])
    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = run_pipeline()

    y_true = df["true_is_spoof"].to_numpy()
    is_anomaly = df["ae_is_anomaly"].to_numpy()
    scores = df["ae_anomaly_score"].to_numpy()

    precision = precision_score(y_true, is_anomaly, zero_division=0)
    recall = recall_score(y_true, is_anomaly, zero_division=0)
    f1 = f1_score(y_true, is_anomaly, zero_division=0)

    print(f"Autoencoder ({DEFAULT_ACTIVATION} activation) flagged {int(is_anomaly.sum())} snapshots as anomalous.")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    plot_reconstruction_error_distribution(df, scores, OUTPUT_DIR / "ae_reconstruction_error.png")
    print(f"Saved {OUTPUT_DIR / 'ae_reconstruction_error.png'}")

    print("\nRunning activation ablation (ReLU / GELU / Swish)...")
    rng = np.random.default_rng(SEED)
    full_df = engineer_features(simulate_orderbook(N_SNAPSHOTS, rng))
    comparison = compare_activations(full_df)
    print(comparison)
    comparison.write_csv(OUTPUT_DIR / "ae_activation_comparison.csv")
    plot_activation_comparison(comparison, OUTPUT_DIR / "ae_activation_comparison.png")
    print(f"Saved {OUTPUT_DIR / 'ae_activation_comparison.csv'} and ae_activation_comparison.png")


if __name__ == "__main__":
    main()
