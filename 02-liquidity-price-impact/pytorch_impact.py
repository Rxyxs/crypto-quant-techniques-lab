"""PyTorch MLP for the order-book price-impact regression task, trained
with a custom Huber loss (robust to the fat-tailed return distribution)
and compared across three activation functions -- ReLU, GELU, and Swish
(SiLU) -- on the same real BTCUSDT data and train/test split used by
xgboost_impact.py and baseline_impact.py.

Requires `data/download_binance_data.py` to have been run first.
Run: venv/Scripts/python.exe pytorch_impact.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from xgboost_impact import (
    HORIZONS,
    PRIMARY_HORIZON,
    RAW_DIR,
    SEED,
    TEST_FRAC,
    build_dataset,
    chronological_split,
    engineer_features,
)
from persist_metrics import persist_metrics

FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 256
ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,  # SiLU == Swish (x * sigmoid(x))
}


def huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Custom Huber loss: quadratic for small errors, linear beyond
    `delta` -- robust to the fat-tailed forward-return distribution
    compared to plain MSE, without discarding gradient information for
    outliers the way plain MAE does."""
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)


class ImpactMLP(nn.Module):
    def __init__(self, n_features: int, activation: type[nn.Module], hidden: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            activation(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(X_train: np.ndarray, y_train: np.ndarray, activation: type[nn.Module],
                 seed: int, epochs: int = EPOCHS) -> tuple[ImpactMLP, list[float]]:
    torch.manual_seed(seed)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                          generator=torch.Generator().manual_seed(seed))

    model = ImpactMLP(n_features=X_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loss_history = []
    model.train()
    for _ in range(epochs):
        epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = huber_loss(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        loss_history.append(float(np.mean(epoch_losses)))
    return model, loss_history


def evaluate(model: ImpactMLP, X_test: np.ndarray, y_test: np.ndarray, y_train_mean: float) -> tuple[dict, np.ndarray]:
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    baseline_pred = np.full_like(y_test, fill_value=y_train_mean)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        "n_test": int(len(y_test)),
    }
    return metrics, y_pred


def compare_activations(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                         seed: int) -> tuple[dict, dict[str, list[float]], dict[str, np.ndarray]]:
    """Trains the same MLP architecture with ReLU, GELU, and Swish
    activations on the identical data/split and returns comparable
    metrics + loss curves for each."""
    results = {}
    loss_curves = {}
    predictions = {}
    y_train_mean = float(y_train.mean())
    for name, activation in ACTIVATIONS.items():
        model, loss_history = train_model(X_train, y_train, activation, seed)
        metrics, y_pred = evaluate(model, X_test, y_test, y_train_mean)
        results[name] = metrics
        loss_curves[name] = loss_history
        predictions[name] = y_pred
    return results, loss_curves, predictions


def multi_horizon_eval(df, feature_cols: list[str], horizons: list[int], activation: type[nn.Module],
                        seed: int) -> dict:
    train_df, test_df = chronological_split(df, TEST_FRAC)
    results = {}
    for h in horizons:
        target_col = f"future_return_{h}"
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df.select(feature_cols).to_numpy())
        y_train = train_df[target_col].to_numpy()
        X_test = scaler.transform(test_df.select(feature_cols).to_numpy())
        y_test = test_df[target_col].to_numpy()

        model, _ = train_model(X_train, y_train, activation, seed)
        metrics, _ = evaluate(model, X_test, y_test, float(y_train.mean()))
        results[h] = metrics
    return results


def plot_loss_curves(loss_curves: dict[str, list[float]], out_path: Path) -> None:
    colors = {"relu": "#3B7DD8", "gelu": "#D8763B", "swish": "#3B9E7D"}
    plt.figure(figsize=(8, 6))
    for name, history in loss_curves.items():
        plt.plot(history, label=name.upper(), color=colors.get(name), linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training Huber loss")
    plt.title("PyTorch MLP: Loss per Epoch by Activation (real data)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def animate_loss_curves(loss_curves: dict[str, list[float]], out_path: Path) -> None:
    """Racing line-chart GIF of the same real per-epoch Huber training loss
    used in plot_loss_curves -- one frame per epoch (already <= 60), with a
    floating annotated label at the advancing tip of each activation's line."""
    import matplotlib.animation as animation

    colors = {"relu": "#3B7DD8", "gelu": "#D8763B", "swish": "#3B9E7D"}
    names = list(loss_curves.keys())
    n_frames = max(len(h) for h in loss_curves.values())
    all_losses = np.concatenate([np.array(h) for h in loss_curves.values()])

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        lines = {name: ax.plot([], [], label=name.upper(), color=colors.get(name), linewidth=1.8)[0]
                 for name in names}
        labels = {name: ax.annotate(
            "", xy=(0, loss_curves[name][0]), xytext=(15, 0), textcoords="offset points",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc=colors.get(name), ec="none", alpha=0.9),
        ) for name in names}

        ax.set_xlim(0, n_frames - 1)
        ax.set_ylim(all_losses.min() * 0.95, all_losses.max() * 1.05)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Huber loss")
        ax.set_title("PyTorch MLP: Loss per Epoch by Activation (real data)")
        ax.legend(loc="upper right")

        def update(frame):
            artists = []
            for name, history in loss_curves.items():
                i = min(frame, len(history) - 1)
                xs = list(range(i + 1))
                ys = history[: i + 1]
                lines[name].set_data(xs, ys)
                labels[name].xy = (i, history[i])
                labels[name].set_text(f"{name.upper()}: {history[i]:.4f}")
                artists.extend([lines[name], labels[name]])
            return artists

        ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=150, blit=False)
        ani.save(out_path, writer="pillow")
        plt.close(fig)


def plot_predicted_vs_actual(y_test: np.ndarray, y_pred: np.ndarray, out_path: Path, sample_size: int = 4000) -> None:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_test), size=min(sample_size, len(y_test)), replace=False)
    lo, hi = np.percentile(y_test, [0.5, 99.5])

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test[idx], y_pred[idx], alpha=0.25, s=10, color="#3B7DD8")
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect prediction")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual forward return")
    plt.ylabel("Predicted forward return")
    plt.title("PyTorch MLP (Swish): Predicted vs. Actual (real data, test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading real BTCUSDT bookDepth + aggTrades data (Binance Vision)...")
    raw = build_dataset(RAW_DIR)
    print(f"  {raw.height:,} joined 1-minute bins")

    print(f"Engineering features (horizons = {HORIZONS} minutes)...")
    df, feature_cols, target_cols = engineer_features(raw, HORIZONS)
    print(f"  {df.height:,} usable rows after feature/target construction")

    primary_target = f"future_return_{PRIMARY_HORIZON}"
    train_df, test_df = chronological_split(df, TEST_FRAC)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.select(feature_cols).to_numpy())
    y_train = train_df[primary_target].to_numpy()
    X_test = scaler.transform(test_df.select(feature_cols).to_numpy())
    y_test = test_df[primary_target].to_numpy()

    print(f"Training MLP with {list(ACTIVATIONS)} activations (custom Huber loss, {EPOCHS} epochs each)...")
    activation_results, loss_curves, predictions = compare_activations(X_train, y_train, X_test, y_test, SEED)
    print(json.dumps(activation_results, indent=2))

    best_activation = min(activation_results, key=lambda k: activation_results[k]["rmse"])
    print(f"Best activation on primary horizon: {best_activation}")

    print("Running multi-horizon comparison with best activation (1 / 5 / 15 minutes ahead)...")
    horizon_results = multi_horizon_eval(df, feature_cols, HORIZONS, ACTIVATIONS[best_activation], SEED)
    print(json.dumps(horizon_results, indent=2))

    report = {
        "model": "pytorch_mlp_huber",
        "primary_horizon_minutes": PRIMARY_HORIZON,
        "activation_comparison": activation_results,
        "best_activation": best_activation,
        "multi_horizon_best_activation": horizon_results,
        "n_rows_total": df.height,
    }
    with open(REPORTS_DIR / "metrics_pytorch.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Persisting metrics to DuckDB...")
    persist_metrics("pytorch_mlp", horizon_results)

    print("Rendering figures...")
    plot_loss_curves(loss_curves, FIGURES_DIR / "pytorch_loss_curves.png")
    animate_loss_curves(loss_curves, FIGURES_DIR / "pytorch_loss_curves_animated.gif")
    plot_predicted_vs_actual(y_test, predictions[best_activation], FIGURES_DIR / "pytorch_predicted_vs_actual.png")

    print(f"Done. Figures written to {FIGURES_DIR}/, metrics to {REPORTS_DIR}/metrics_pytorch.json")


if __name__ == "__main__":
    main()
