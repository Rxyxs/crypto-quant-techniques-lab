"""
Clasificador de direccion de velas (alcista/bajista) sobre datos OHLCV
simulados estilo Binance, usando una red neuronal densa en PyTorch.

Entrena dos variantes identicas del mismo clasificador -- una con
activaciones ReLU y otra con Tanh -- para comparar su comportamiento de
entrenamiento, y evalua la variante ganadora en un set de test cronologico
(confusion matrix + tabla de precision/recall).

Uso:
    .\\venv\\Scripts\\python.exe train_classifier.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
N_CANDLES = 20_000
SYMBOL = "BTCUSDT"
INTERVAL_HOURS = 1

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# el resto (0.15) es test

BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 12
LEARNING_RATE = 1e-3
HIDDEN_DIMS = (64, 32, 16)
DROPOUT = 0.2

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"

FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "vol_10",
    "vol_20",
    "rsi_14",
    "sma_ratio_10",
    "sma_ratio_20",
    "ema_ratio_12",
    "volume_zscore",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "momentum_streak",
]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def simulate_ohlcv_data(n_candles: int, seed: int) -> pl.DataFrame:
    """Simula velas OHLCV estilo Binance con clustering de volatilidad
    (GARCH(1,1)) y un componente de momentum de corto plazo AR(1) en los
    retornos -- sin este ultimo, la direccion de la siguiente vela seria
    indistinguible de un lanzamiento de moneda y no habria nada que un
    clasificador pudiera aprender genuinamente."""
    rng = np.random.default_rng(seed)

    phi = 0.30  # momentum AR(1) en el retorno log
    omega, alpha, beta = 1e-7, 0.08, 0.88  # GARCH(1,1)

    log_returns = np.zeros(n_candles)
    sigma2 = np.full(n_candles, omega / (1 - alpha - beta))
    eps = rng.standard_normal(n_candles)

    for t in range(1, n_candles):
        sigma2[t] = omega + alpha * (log_returns[t - 1] ** 2) + beta * sigma2[t - 1]
        log_returns[t] = phi * log_returns[t - 1] + np.sqrt(sigma2[t]) * eps[t]

    close = 40_000.0 * np.exp(np.cumsum(log_returns))
    open_ = np.empty(n_candles)
    open_[0] = close[0]
    gap_noise = rng.normal(0, 0.0005, size=n_candles)
    open_[1:] = close[:-1] * np.exp(gap_noise[1:])

    sigma = np.sqrt(sigma2)
    intrabar_range = close * sigma * rng.uniform(0.6, 2.2, size=n_candles)
    upper_wick = intrabar_range * rng.uniform(0.1, 1.0, size=n_candles)
    lower_wick = intrabar_range * rng.uniform(0.1, 1.0, size=n_candles)
    high = np.maximum(open_, close) + upper_wick
    low = np.minimum(open_, close) - lower_wick

    volume = 500 * np.exp(rng.normal(0, 0.4, size=n_candles)) * (1 + 8 * np.abs(log_returns) / sigma.mean())

    start = np.datetime64("2024-01-01T00:00:00", "us")
    timestamp = start + (np.arange(n_candles) * INTERVAL_HOURS * 3600).astype("timedelta64[s]")

    return pl.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": [SYMBOL] * n_candles,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def build_features_and_label(df: pl.DataFrame) -> pl.DataFrame:
    """Todas las features usan solo informacion disponible al cierre de la
    vela t (o antes); la etiqueta es la direccion de la vela t+1, obtenida
    con shift(-1) para que quede alineada con las features de t sin fuga
    de informacion hacia adelante."""
    df = df.with_columns(
        pl.col("close").log().diff().alias("_log_ret"),
    )
    gains = pl.when(pl.col("_log_ret") > 0).then(pl.col("_log_ret")).otherwise(0.0)
    losses = pl.when(pl.col("_log_ret") < 0).then(-pl.col("_log_ret")).otherwise(0.0)
    direction = pl.when(pl.col("_log_ret") > 0).then(1).when(pl.col("_log_ret") < 0).then(-1).otherwise(0)

    df = df.with_columns(
        ret_1=pl.col("_log_ret"),
        ret_3=(pl.col("close").log() - pl.col("close").log().shift(3)),
        ret_6=(pl.col("close").log() - pl.col("close").log().shift(6)),
        vol_10=pl.col("_log_ret").rolling_std(window_size=10),
        vol_20=pl.col("_log_ret").rolling_std(window_size=20),
        rsi_14=100
        - 100
        / (1 + gains.rolling_mean(window_size=14) / (losses.rolling_mean(window_size=14) + 1e-12)),
        sma_ratio_10=(pl.col("close") / pl.col("close").rolling_mean(window_size=10) - 1),
        sma_ratio_20=(pl.col("close") / pl.col("close").rolling_mean(window_size=20) - 1),
        ema_ratio_12=(pl.col("close") / pl.col("close").ewm_mean(span=12) - 1),
        volume_zscore=(
            (pl.col("volume") - pl.col("volume").rolling_mean(window_size=20))
            / (pl.col("volume").rolling_std(window_size=20) + 1e-12)
        ),
        body_ratio=((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-12)),
        upper_wick_ratio=(
            (pl.col("high") - pl.max_horizontal("open", "close")) / (pl.col("high") - pl.col("low") + 1e-12)
        ),
        lower_wick_ratio=(
            (pl.min_horizontal("open", "close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-12)
        ),
        momentum_streak=direction.rolling_sum(window_size=5),
        label_next_bullish=(pl.col("close").shift(-1) > pl.col("open").shift(-1)).cast(pl.Int8),
    )

    return df.drop("_log_ret").drop_nulls()


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_auc: float = 0.0


class DenseClassifier(nn.Module):
    def __init__(self, input_dim: int, activation: str, hidden_dims=HIDDEN_DIMS, dropout: float = DROPOUT):
        super().__init__()
        act_layer = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), act_layer(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_variant(
    activation: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> tuple[DenseClassifier, TrainHistory]:
    model = DenseClassifier(input_dim=X_train.shape[1], activation=activation).to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    history = TrainHistory()
    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_loss = epoch_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_acc = float(((val_probs >= 0.5).astype(np.int8) == y_val).mean())
            val_auc = roc_auc_score(y_val, val_probs)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            history.best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  [{activation:>4}] epoch {epoch:3d}/{MAX_EPOCHS}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.3f}  val_auc={val_auc:.3f}"
            )

        if patience_counter >= PATIENCE:
            print(f"  [{activation:>4}] early stopping en epoch {epoch} (mejor epoch: {history.best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def plot_activation_functions() -> None:
    x = np.linspace(-5, 5, 400)
    relu = np.maximum(0, x)
    tanh = np.tanh(x)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(x, relu, color="#4C72B0", linewidth=2)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].axvline(0, color="gray", linewidth=0.8)
    axes[0].set_title("ReLU: f(x) = max(0, x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")

    axes[1].plot(x, tanh, color="#DD8452", linewidth=2)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].axvline(0, color="gray", linewidth=0.8)
    axes[1].set_title("Tanh: f(x) = tanh(x)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f(x)")

    fig.suptitle("Funciones de activacion comparadas")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "activation_functions.png", dpi=150)
    plt.close(fig)


def plot_training_comparison(history_relu: TrainHistory, history_tanh: TrainHistory, suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history_relu.val_loss, label="ReLU", color="#4C72B0")
    axes[0].plot(history_tanh.val_loss, label="Tanh", color="#DD8452")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val loss (BCE)")
    axes[0].set_title("Perdida de validacion por epoch")
    axes[0].legend()

    axes[1].plot(history_relu.val_accuracy, label="ReLU", color="#4C72B0")
    axes[1].plot(history_tanh.val_accuracy, label="Tanh", color="#DD8452")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val accuracy")
    axes[1].set_title("Accuracy de validacion por epoch")
    axes[1].legend()

    fig.suptitle("Comparacion de entrenamiento: ReLU vs Tanh")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"training_curves_relu_vs_tanh{suffix}.png", dpi=150)
    plt.close(fig)

    animate_training_comparison(history_relu, history_tanh, suffix)


def animate_training_comparison(history_relu: TrainHistory, history_tanh: TrainHistory, suffix: str = "") -> None:
    """Version animada (GIF) de la curva de perdida de validacion: un
    'racing line chart' que dibuja progresivamente ReLU vs Tanh epoch a
    epoch, con una etiqueta flotante que muestra el valor actual en la
    punta de cada linea -- usa exactamente los mismos val_loss ya
    calculados en plot_training_comparison, sin inventar datos."""
    import matplotlib.animation as animation

    relu_loss = history_relu.val_loss
    tanh_loss = history_tanh.val_loss
    n_epochs = min(len(relu_loss), len(tanh_loss))
    if n_epochs < 2:
        return

    n_frames = min(60, max(30, n_epochs))
    n_frames = min(n_frames, n_epochs)
    frame_epochs = sorted(set(np.linspace(1, n_epochs, n_frames, dtype=int)))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(1, n_epochs + 1)
        ax.set_xlim(1, n_epochs)
        y_all = relu_loss[:n_epochs] + tanh_loss[:n_epochs]
        pad = (max(y_all) - min(y_all)) * 0.15 + 1e-6
        ax.set_ylim(min(y_all) - pad, max(y_all) + pad)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val loss (BCE)")
        ax.set_title("Perdida de validacion por epoch -- ReLU vs Tanh (animado)")

        line_relu, = ax.plot([], [], color="#4C72B0", linewidth=2, label="ReLU")
        line_tanh, = ax.plot([], [], color="#DD8452", linewidth=2, label="Tanh")
        ax.legend(loc="upper right")

        label_relu = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#4C72B0", ec="white", alpha=0.85),
        )
        label_tanh = ax.annotate(
            "", xy=(0, 0), xytext=(10, -20), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#DD8452", ec="white", alpha=0.85),
        )

        def update(frame_idx):
            k = frame_epochs[frame_idx]
            xs = x[:k]
            line_relu.set_data(xs, relu_loss[:k])
            line_tanh.set_data(xs, tanh_loss[:k])
            label_relu.xy = (xs[-1], relu_loss[k - 1])
            label_relu.set_text(f"ReLU: {relu_loss[k - 1]:.4f}")
            label_tanh.xy = (xs[-1], tanh_loss[k - 1])
            label_tanh.set_text(f"Tanh: {tanh_loss[k - 1]:.4f}")
            return line_relu, line_tanh, label_relu, label_tanh

        ani = animation.FuncAnimation(fig, update, frames=len(frame_epochs), interval=120, blit=False)
        out_path = FIGURES_DIR / f"training_curves_relu_vs_tanh{suffix}_animated.gif"
        ani.save(out_path, writer="pillow")
        plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, activation: str, suffix: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bajista", "Alcista"],
        yticklabels=["Bajista", "Alcista"],
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion (test) -- modelo {activation.upper()}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix{suffix}.png", dpi=150)
    plt.close(fig)


def plot_precision_recall_table(y_true: np.ndarray, y_pred: np.ndarray, activation: str, suffix: str = "") -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    accuracy = float((y_true == y_pred).mean())

    rows = [
        ["Bajista (0)", f"{precision[0]:.3f}", f"{recall[0]:.3f}", f"{f1[0]:.3f}", f"{support[0]}"],
        ["Alcista (1)", f"{precision[1]:.3f}", f"{recall[1]:.3f}", f"{f1[1]:.3f}", f"{support[1]}"],
        ["Accuracy", "", "", f"{accuracy:.3f}", f"{support.sum()}"],
    ]
    col_labels = ["Clase", "Precision", "Recall", "F1-score", "Support"]

    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    ax.set_title(f"Precision / Recall (test) -- modelo {activation.upper()}", pad=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"precision_recall_table{suffix}.png", dpi=150)
    plt.close(fig)

    return {
        "accuracy": accuracy,
        "precision_bajista": float(precision[0]),
        "recall_bajista": float(recall[0]),
        "f1_bajista": float(f1[0]),
        "precision_alcista": float(precision[1]),
        "recall_alcista": float(recall[1]),
        "f1_alcista": float(f1[1]),
    }


def main() -> None:
    set_seeds(SEED)
    for d in (DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print(f"\nSimulando {N_CANDLES:,} velas OHLCV de {SYMBOL} (semilla={SEED})...")
    raw = simulate_ohlcv_data(N_CANDLES, SEED)
    raw.write_csv(DATA_DIR / "ohlcv_simulated.csv")

    df = build_features_and_label(raw)
    print(f"  {len(df):,} velas utilizables tras calcular features (se pierden las primeras por ventanas rolling)")
    bullish_rate = df["label_next_bullish"].mean()
    print(f"  balance de clases: {bullish_rate:.1%} alcistas / {1 - bullish_rate:.1%} bajistas")

    n = len(df)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)
    train_df = df[:n_train]
    val_df = df[n_train : n_train + n_val]
    test_df = df[n_train + n_val :]
    print(f"  split cronologico -> train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    X_train = train_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    X_val = val_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    X_test = test_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    y_train = train_df["label_next_bullish"].to_numpy().astype(np.float32)
    y_val = val_df["label_next_bullish"].to_numpy().astype(np.float32)
    y_test = test_df["label_next_bullish"].to_numpy().astype(np.float32)

    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(a).astype(np.float32) for a in (X_train, X_val, X_test))

    print("\nGraficando funciones de activacion...")
    plot_activation_functions()

    print("\nEntrenando variante ReLU...")
    model_relu, history_relu = train_variant("relu", X_train, y_train, X_val, y_val, device)
    print("\nEntrenando variante Tanh...")
    model_tanh, history_tanh = train_variant("tanh", X_train, y_train, X_val, y_val, device)

    plot_training_comparison(history_relu, history_tanh)

    if history_relu.best_val_auc >= history_tanh.best_val_auc:
        winner_name, winner_model, winner_history = "relu", model_relu, history_relu
    else:
        winner_name, winner_model, winner_history = "tanh", model_tanh, history_tanh

    print(
        f"\nActivacion ganadora por AUC de validacion: {winner_name.upper()} "
        f"(ReLU AUC={history_relu.best_val_auc:.3f} vs Tanh AUC={history_tanh.best_val_auc:.3f})"
    )

    torch.save(model_relu.state_dict(), MODELS_DIR / "classifier_relu.pt")
    torch.save(model_tanh.state_dict(), MODELS_DIR / "classifier_tanh.pt")

    winner_model.eval()
    with torch.no_grad():
        test_logits = winner_model(torch.from_numpy(X_test).to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    y_pred = (test_probs >= 0.5).astype(np.int8)
    y_true = y_test.astype(np.int8)

    test_auc = roc_auc_score(y_true, test_probs)
    print(f"\nEvaluacion en test (modelo {winner_name.upper()}): AUC={test_auc:.3f}")

    plot_confusion_matrix(y_true, y_pred, winner_name)
    metrics = plot_precision_recall_table(y_true, y_pred, winner_name)
    metrics["test_auc"] = float(test_auc)
    metrics["winner_activation"] = winner_name
    metrics["relu_best_val_auc"] = history_relu.best_val_auc
    metrics["tanh_best_val_auc"] = history_tanh.best_val_auc
    metrics["n_train"] = len(train_df)
    metrics["n_val"] = len(val_df)
    metrics["n_test"] = len(test_df)

    import json

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAccuracy test: {metrics['accuracy']:.3f}")
    print(f"Artefactos guardados en {DATA_DIR}, {FIGURES_DIR}, {MODELS_DIR}, {REPORTS_DIR}")


if __name__ == "__main__":
    main()
