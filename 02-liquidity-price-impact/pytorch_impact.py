"""PyTorch MLP for the order-book price-impact regression task, trained
with a custom Huber loss (robust to the fat-tailed return distribution)
and compared across three activation functions -- ReLU, GELU, and Swish
(SiLU) -- on the same real BTCUSDT data and train/test split used by
xgboost_impact.py and baseline_impact.py.

Requires `data/download_binance_data.py` to have been run first.
Run: venv/Scripts/python.exe pytorch_impact.py
"""
from __future__ import annotations

import copy
import hashlib
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
MIN_EPOCHS = 100  # ninguna red se detiene antes de 100 epocas
MAX_EPOCHS = 300
PATIENCE = 15
VAL_FRACTION = 0.15  # cola cronologica del train usada para early stopping
CHECKPOINT_DIR = Path("outputs/models/checkpoints")
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


def _fingerprint(*arrays, **config) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def train_model(X_train: np.ndarray, y_train: np.ndarray, activation: type[nn.Module],
                 seed: int, epochs: int = MAX_EPOCHS, min_epochs: int = MIN_EPOCHS, patience: int = PATIENCE,
                 checkpoint_path: Path | None = None, history_out: dict | None = None) -> tuple[ImpactMLP, list[float]]:
    """Entrena con early stopping. La validacion es el ultimo ``VAL_FRACTION`` del
    train en orden cronologico (anterior al test). Al menos ``min_epochs`` epocas;
    desde ahi corta tras ``patience`` sin mejora y restaura la mejor epoca.
    Con ``checkpoint_path`` el estado completo se guarda cada epoca y una corrida
    interrumpida retoma donde quedo. Devuelve (modelo, loss de train por epoca);
    ``history_out`` recibe ademas la loss de validacion y la mejor epoca."""
    torch.manual_seed(seed)
    n_val = max(1, int(len(X_train) * VAL_FRACTION))
    Xt = torch.tensor(X_train[:-n_val], dtype=torch.float32)
    yt = torch.tensor(y_train[:-n_val], dtype=torch.float32)
    Xv = torch.tensor(X_train[-n_val:], dtype=torch.float32)
    yv = torch.tensor(y_train[-n_val:], dtype=torch.float32)
    generator = torch.Generator()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                                         shuffle=True, generator=generator)

    model = ImpactMLP(n_features=X_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    fp = _fingerprint(X_train, y_train, act=activation.__name__, seed=seed, epochs=epochs,
                      min_epochs=min_epochs, patience=patience, lr=LR, wd=WEIGHT_DECAY, batch=BATCH_SIZE)
    state = {"epoch": 0, "train": [], "val": [], "best_val": float("inf"), "best_epoch": 0,
             "best_state": None, "bad_epochs": 0, "done": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            state = ckpt["state"]

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state}, tmp)
        tmp.replace(path)

    while not state["done"] and state["epoch"] < epochs:
        generator.manual_seed(seed + state["epoch"])  # mismo orden de batches al retomar
        torch.manual_seed(seed + 10_000 + state["epoch"])
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = huber_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        state["epoch"] += 1
        state["train"].append(float(np.mean(epoch_losses)))
        model.eval()
        with torch.no_grad():
            val_loss = huber_loss(model(Xv), yv).item()
        state["val"].append(val_loss)
        if val_loss < state["best_val"] - 1e-9:
            state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if state["epoch"] >= min(min_epochs, epochs) and state["bad_epochs"] >= patience:
            state["done"] = True
        save_checkpoint()

    if state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()
    if history_out is not None:
        history_out.update(val_loss=state["val"], best_epoch=state["best_epoch"], epochs_run=state["epoch"])
    return model, state["train"]


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
        info: dict = {}
        model, loss_history = train_model(X_train, y_train, activation, seed,
                                          checkpoint_path=CHECKPOINT_DIR / f"impact_{name}.ckpt", history_out=info)
        print(f"  {name}: mejor epoca {info['best_epoch']} de {info['epochs_run']}")
        metrics, y_pred = evaluate(model, X_test, y_test, y_train_mean)
        metrics["best_epoch"] = info["best_epoch"]
        metrics["val_huber_loss"] = float(min(info["val_loss"]))
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

        model, _ = train_model(X_train, y_train, activation, seed,
                               checkpoint_path=CHECKPOINT_DIR / f"impact_h{h}_{activation.__name__}.ckpt")
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

    print(f"Training MLP with {list(ACTIVATIONS)} activations (custom Huber loss, {MIN_EPOCHS}-{MAX_EPOCHS} epochs each, early stopping)...")
    activation_results, loss_curves, predictions = compare_activations(X_train, y_train, X_test, y_test, SEED)
    print(json.dumps(activation_results, indent=2))

    # Elegida con la loss de validacion (cola cronologica del train), no con el RMSE de test.
    best_activation = min(activation_results, key=lambda k: activation_results[k]["val_huber_loss"])
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
