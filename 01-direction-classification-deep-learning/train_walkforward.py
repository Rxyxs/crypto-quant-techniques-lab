"""
Evolucion del clasificador a 3 clases (Bajista / Neutro / Alcista) con la
arquitectura hibrida Conv1D + Multi-Head Attention de `model_conv_attention.py`,
evaluada con un esquema riguroso de Walk-Forward Cross-Validation (rolling-
origin) y una metrica final que incorpora costos de transaccion y slippage
-- no solo accuracy/F1, que ignoran por completo si la senal del modelo es
economicamente utilizable una vez pagados los costos reales de operar.

Corre sobre las velas REALES de BTCUSDT (`fetch_binance_data.py`): un costo
de transaccion solo tiene sentido economico sobre un mercado real, no sobre
precios simulados.

Walk-forward (rolling-origin) en vez de un unico split 70/15/15:
    Fold 1: train [0 .. t1)          val [ultimo tramo de train]   test [t1 .. t2)
    Fold 2: train [0 .. t2)          val [ultimo tramo de train]   test [t2 .. t3)
    ...
    Fold K: train [0 .. tK)          val [ultimo tramo de train]   test [tK .. N)

La ventana de entrenamiento **se expande** en cada fold (nunca se descarta
historia pasada), el bloque de test de cada fold es estrictamente posterior
a todo lo usado para entrenar y validar ese fold, y ningun bloque de test se
reutiliza como entrenamiento de un fold anterior -- el modelo nunca ve datos
del futuro relativo a la decision que esta evaluando, en ningun fold.

Uso:
    .\\venv\\Scripts\\python.exe train_walkforward.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_conv_attention import CLASS_NAMES, N_CLASSES, WINDOW_LENGTH, Conv1DAttentionClassifier
from train_classifier import DATA_DIR, FEATURE_COLUMNS, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED, build_features_and_label, set_seeds
from train_classifier_real import load_or_fetch_real_data

DEADZONE_MULT = 0.3  # dead-zone del label = +-0.3 * vol_20 (ver README §3 para el analisis de balance de clases)

N_FOLDS = 5
MIN_TRAIN_FRACTION = 0.50  # fraccion inicial de la serie reservada solo para el primer bloque de entrenamiento
VAL_FRACTION_OF_TRAIN = 0.15  # cola del bloque de train de cada fold usada como validacion (early stopping)

BATCH_SIZE = 256
MAX_EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 1e-3

TRANSACTION_COST_BPS = 5.0  # comision taker tipica en Binance spot (~0.05%)
SLIPPAGE_BPS = 5.0  # slippage estimado adicional en ejecucion real
TOTAL_COST_BPS = TRANSACTION_COST_BPS + SLIPPAGE_BPS
HOURS_PER_YEAR = 24 * 365
POSITION_BY_CLASS = {0: -1.0, 1: 0.0, 2: 1.0}  # Bajista=short, Neutro=flat, Alcista=long


def build_multiclass_label(df: pl.DataFrame, deadzone_mult: float = DEADZONE_MULT) -> pl.DataFrame:
    """Extiende `build_features_and_label` (binario) con un label de 3
    clases sobre el retorno logaritmico hacia adelante (`fwd_log_ret`,
    misma vela objetivo t+1 que el label binario), usando un dead-zone
    proporcional a la volatilidad reciente (`vol_20`) en vez de un umbral
    fijo -- para que "Neutro" signifique lo mismo en un regimen de alta
    volatilidad que en uno tranquilo."""
    df = build_features_and_label(df)
    df = df.with_columns(fwd_log_ret=(pl.col("close").shift(-1).log() - pl.col("close").log()))
    threshold = deadzone_mult * pl.col("vol_20")
    df = df.with_columns(
        label_direction=pl.when(pl.col("fwd_log_ret") > threshold)
        .then(2)
        .when(pl.col("fwd_log_ret") < -threshold)
        .then(0)
        .otherwise(1)
    )
    return df.drop_nulls()


def build_sequences(df: pl.DataFrame, window_length: int = WINDOW_LENGTH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye ventanas deslizantes (n_windows, window_length, n_features).
    La ventana que termina en la fila `i` predice `label_direction[i]`, que
    ya es la direccion de la vela `i+1` -- no hay desplazamiento adicional
    que hacer aqui, solo empaquetar la historia que el modelo puede ver."""
    feats = df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
    labels = df["label_direction"].to_numpy().astype(np.int64)
    fwd_ret = df["fwd_log_ret"].to_numpy().astype(np.float32)

    n = len(df)
    n_windows = n - window_length + 1
    X = np.lib.stride_tricks.sliding_window_view(feats, window_length, axis=0)
    X = np.ascontiguousarray(X.transpose(0, 2, 1))  # -> (n_windows, window_length, n_features)
    y = labels[window_length - 1 :]
    ret = fwd_ret[window_length - 1 :]
    assert len(X) == n_windows == len(y) == len(ret)
    return X, y, ret


def walk_forward_splits(n_samples: int, n_folds: int = N_FOLDS, min_train_frac: float = MIN_TRAIN_FRACTION):
    """Genera `n_folds` particiones (train_slice, val_slice, test_slice) de
    origen rodante: el inicio del bloque de test avanza `test_block` en
    cada fold y el entrenamiento siempre arranca en 0 (ventana expansiva)."""
    test_region_start = int(n_samples * min_train_frac)
    test_block = (n_samples - test_region_start) // n_folds

    folds = []
    for k in range(n_folds):
        test_start = test_region_start + k * test_block
        test_end = n_samples if k == n_folds - 1 else test_start + test_block
        train_end = test_start
        val_start = int(train_end * (1 - VAL_FRACTION_OF_TRAIN))
        folds.append((slice(0, val_start), slice(val_start, train_end), slice(test_start, test_end)))
    return folds


@dataclass
class FoldHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = 0


def train_fold(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    n_features: int, device: torch.device, seed: int = SEED,
) -> tuple[Conv1DAttentionClassifier, FoldHistory]:
    torch.manual_seed(seed)
    model = Conv1DAttentionClassifier(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    history = FoldHistory()
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
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = float((val_pred == y_val).mean())

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def simulate_pnl(y_pred: np.ndarray, fwd_ret: np.ndarray, cost_bps: float = TOTAL_COST_BPS) -> dict:
    """Traduce las predicciones de clase en una estrategia trivial
    (Alcista=long, Bajista=short, Neutro=flat) y descuenta un costo de
    transaccion + slippage cada vez que la posicion cambia -- la unica
    forma honesta de saber si una senal de clasificacion sobrevive el
    costo real de operarla, en vez de reportar solo accuracy/F1."""
    position = np.array([POSITION_BY_CLASS[c] for c in y_pred], dtype=np.float64)
    gross_returns = position * fwd_ret.astype(np.float64)

    position_change = np.abs(np.diff(position, prepend=0.0))
    cost = position_change * (cost_bps / 10_000.0)
    net_returns = gross_returns - cost

    net_std = net_returns.std()
    sharpe_like = float(net_returns.mean() / net_std * np.sqrt(HOURS_PER_YEAR)) if net_std > 0 else 0.0

    return {
        "n_trades": int((position_change > 0).sum()),
        "gross_return_total": float(gross_returns.sum()),
        "cost_total": float(cost.sum()),
        "net_return_total": float(net_returns.sum()),
        "annualized_sharpe_like_net": sharpe_like,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "position": position,
    }


def plot_learning_curves(history: FoldHistory, fold_idx: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.train_loss, label="Train loss", color="#4C72B0")
    axes[0].plot(history.val_loss, label="Val loss", color="#DD8452")
    axes[0].axvline(history.best_epoch - 1, color="gray", linestyle="--", linewidth=1, label="Mejor epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Curva de perdida")
    axes[0].legend()

    axes[1].plot(history.val_accuracy, color="#55A868")
    axes[1].axvline(history.best_epoch - 1, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val accuracy")
    axes[1].set_title("Accuracy de validacion")

    fig.suptitle(f"Curvas de aprendizaje -- Fold {fold_idx} (Conv1D + Multi-Head Attention)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "walkforward_learning_curves.png", dpi=150)
    plt.close(fig)

    animate_learning_curve(history, fold_idx)


def animate_learning_curve(history: FoldHistory, fold_idx: int) -> None:
    """Racing line chart (GIF) de la curva de perdida train/val del fold --
    usa los mismos history.train_loss / history.val_loss ya calculados
    durante el entrenamiento real, sin datos sinteticos."""
    import matplotlib.animation as animation

    train_loss = history.train_loss
    val_loss = history.val_loss
    n_epochs = min(len(train_loss), len(val_loss))
    if n_epochs < 2:
        return

    n_frames = min(60, max(30, n_epochs))
    n_frames = min(n_frames, n_epochs)
    frame_epochs = sorted(set(np.linspace(1, n_epochs, n_frames, dtype=int)))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(1, n_epochs + 1)
        ax.set_xlim(1, n_epochs)
        y_all = train_loss[:n_epochs] + val_loss[:n_epochs]
        pad = (max(y_all) - min(y_all)) * 0.15 + 1e-6
        ax.set_ylim(min(y_all) - pad, max(y_all) + pad)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title(f"Curva de perdida -- Fold {fold_idx} (Conv1D + Attention, animado)")

        line_train, = ax.plot([], [], color="#4C72B0", linewidth=2, label="Train loss")
        line_val, = ax.plot([], [], color="#DD8452", linewidth=2, label="Val loss")
        ax.legend(loc="upper right")

        label_train = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#4C72B0", ec="white", alpha=0.85),
        )
        label_val = ax.annotate(
            "", xy=(0, 0), xytext=(10, -20), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#DD8452", ec="white", alpha=0.85),
        )

        def update(frame_idx):
            k = frame_epochs[frame_idx]
            xs = x[:k]
            line_train.set_data(xs, train_loss[:k])
            line_val.set_data(xs, val_loss[:k])
            label_train.xy = (xs[-1], train_loss[k - 1])
            label_train.set_text(f"Train: {train_loss[k - 1]:.4f}")
            label_val.xy = (xs[-1], val_loss[k - 1])
            label_val.set_text(f"Val: {val_loss[k - 1]:.4f}")
            return line_train, line_val, label_train, label_val

        ani = animation.FuncAnimation(fig, update, frames=len(frame_epochs), interval=120, blit=False)
        out_path = FIGURES_DIR / "walkforward_learning_curves_animated.gif"
        ani.save(out_path, writer="pillow")
        plt.close(fig)


def plot_confusion_matrix_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion agregada -- walk-forward\n(todos los folds de test, out-of-sample)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "walkforward_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_pnl_curve(gross_returns: np.ndarray, net_returns: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(np.cumsum(gross_returns), color="#4C72B0", linewidth=1.5, label="PnL bruto (sin costos)")
    ax.plot(np.cumsum(net_returns), color="#C44E52", linewidth=1.5, label=f"PnL neto (costo+slippage = {TOTAL_COST_BPS:.0f} bps/trade)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Vela de test (concatenado cronologicamente a traves de los folds)")
    ax.set_ylabel("Retorno logaritmico acumulado")
    ax.set_title("PnL simulado, out-of-sample walk-forward -- BTCUSDT real, Conv1D + Attention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "walkforward_pnl_curve.png", dpi=150)
    plt.close(fig)

    animate_pnl_curve(gross_returns, net_returns)


def animate_pnl_curve(gross_returns: np.ndarray, net_returns: np.ndarray) -> None:
    """Racing line chart (GIF) del PnL acumulado bruto vs neto, out-of-sample
    walk-forward -- usa los mismos gross_returns/net_returns reales ya
    calculados por simulate_pnl, subsampleados a lo sumo a 60 frames."""
    import matplotlib.animation as animation

    gross_cum = np.cumsum(gross_returns)
    net_cum = np.cumsum(net_returns)
    n_points = len(gross_cum)
    if n_points < 2:
        return

    n_frames = min(60, max(30, n_points))
    n_frames = min(n_frames, n_points)
    frame_idxs = sorted(set(np.linspace(1, n_points, n_frames, dtype=int)))

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(1, n_points + 1)
        ax.set_xlim(1, n_points)
        y_all = np.concatenate([gross_cum, net_cum])
        pad = (y_all.max() - y_all.min()) * 0.15 + 1e-6
        ax.set_ylim(y_all.min() - pad, y_all.max() + pad)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Vela de test (concatenado cronologicamente a traves de los folds)")
        ax.set_ylabel("Retorno logaritmico acumulado")
        ax.set_title("PnL simulado, out-of-sample walk-forward -- BTCUSDT real (animado)")

        line_gross, = ax.plot([], [], color="#4C72B0", linewidth=1.8, label="PnL bruto (sin costos)")
        line_net, = ax.plot([], [], color="#C44E52", linewidth=1.8, label="PnL neto (con costo+slippage)")
        ax.legend(loc="upper right")

        label_gross = ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#4C72B0", ec="white", alpha=0.85),
        )
        label_net = ax.annotate(
            "", xy=(0, 0), xytext=(10, -20), textcoords="offset points",
            color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#C44E52", ec="white", alpha=0.85),
        )

        def update(frame_idx):
            k = frame_idxs[frame_idx]
            xs = x[:k]
            line_gross.set_data(xs, gross_cum[:k])
            line_net.set_data(xs, net_cum[:k])
            label_gross.xy = (xs[-1], gross_cum[k - 1])
            label_gross.set_text(f"Bruto: {gross_cum[k - 1]:.3f}")
            label_net.xy = (xs[-1], net_cum[k - 1])
            label_net.set_text(f"Neto: {net_cum[k - 1]:.3f}")
            return line_gross, line_net, label_gross, label_net

        ani = animation.FuncAnimation(fig, update, frames=len(frame_idxs), interval=120, blit=False)
        out_path = FIGURES_DIR / "walkforward_pnl_curve_animated.gif"
        ani.save(out_path, writer="pillow")
        plt.close(fig)


def main() -> None:
    set_seeds(SEED)
    for d in (DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    raw = load_or_fetch_real_data()
    print(f"{len(raw):,} velas reales de BTCUSDT cargadas ({raw['timestamp'].min()} -> {raw['timestamp'].max()})")

    df = build_multiclass_label(raw)
    class_counts = df["label_direction"].value_counts().sort("label_direction")
    print(f"Balance de clases (dead-zone = {DEADZONE_MULT} * vol_20):")
    for row in class_counts.to_dicts():
        print(f"  {CLASS_NAMES[row['label_direction']]}: {row['count']:,}")

    X, y, fwd_ret = build_sequences(df)
    n_windows = len(X)
    print(f"\n{n_windows:,} ventanas de longitud {WINDOW_LENGTH} construidas")

    folds = walk_forward_splits(n_windows)
    print(f"\nWalk-forward: {N_FOLDS} folds, ventana de train expansiva desde {MIN_TRAIN_FRACTION:.0%} de los datos")

    fold_reports = []
    all_test_true, all_test_pred = [], []
    last_fold_history = None

    for fold_idx, (train_sl, val_sl, test_sl) in enumerate(folds, start=1):
        X_train, y_train = X[train_sl], y[train_sl]
        X_val, y_val = X[val_sl], y[val_sl]
        X_test, y_test = X[test_sl], y[test_sl]
        ret_test = fwd_ret[test_sl]

        n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
        print(f"\n=== Fold {fold_idx}/{N_FOLDS} === train={n_train:,}  val={n_val:,}  test={n_test:,}")

        n_features = X_train.shape[2]
        scaler = StandardScaler().fit(X_train.reshape(-1, n_features))
        X_train_s = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape).astype(np.float32)
        X_val_s = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape).astype(np.float32)
        X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape).astype(np.float32)

        model, history = train_fold(X_train_s, y_train, X_val_s, y_val, n_features, device)
        last_fold_history = history
        print(f"  mejor epoch={history.best_epoch}  val_loss={history.val_loss[history.best_epoch - 1]:.4f}  "
              f"val_acc={history.val_accuracy[history.best_epoch - 1]:.3f}")

        model.eval()
        with torch.no_grad():
            test_logits = model(torch.from_numpy(X_test_s).to(device))
            y_pred = test_logits.argmax(dim=1).cpu().numpy()

        fold_f1_macro = f1_score(y_test, y_pred, average="macro")
        pnl = simulate_pnl(y_pred, ret_test)

        fold_reports.append({
            "fold": fold_idx, "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "test_accuracy": float((y_pred == y_test).mean()),
            "test_f1_macro": float(fold_f1_macro),
            "best_epoch": history.best_epoch,
            "n_trades": pnl["n_trades"],
            "gross_return_total": pnl["gross_return_total"],
            "cost_total": pnl["cost_total"],
            "net_return_total": pnl["net_return_total"],
            "annualized_sharpe_like_net": pnl["annualized_sharpe_like_net"],
        })
        print(f"  test_acc={fold_reports[-1]['test_accuracy']:.3f}  test_f1_macro={fold_f1_macro:.3f}  "
              f"trades={pnl['n_trades']}  PnL_neto={pnl['net_return_total']:+.4f}  PnL_bruto={pnl['gross_return_total']:+.4f}")

        all_test_true.append(y_test)
        all_test_pred.append(y_pred)

        if fold_idx == N_FOLDS:
            torch.save(model.state_dict(), MODELS_DIR / "conv1d_attention_walkforward.pt")

    y_true_all = np.concatenate(all_test_true)
    y_pred_all = np.concatenate(all_test_pred)

    # PnL agregado: recalculado sobre la serie de test COMPLETA y concatenada
    # (no la suma de los PnL por fold, que resetean la posicion a 0 al
    # inicio de cada fold) -- los bloques de test son cronologicamente
    # contiguos entre folds, asi que tratar la evaluacion completa como una
    # sola serie continua es la convencion correcta para el costo de
    # transaccion en los bordes de fold, y es lo que se reporta como
    # numero principal y en la curva de PnL.
    overall_accuracy = float((y_pred_all == y_true_all).mean())
    overall_f1_macro = float(f1_score(y_true_all, y_pred_all, average="macro"))
    overall_pnl = simulate_pnl(y_pred_all, np.concatenate([fwd_ret[sl] for _, _, sl in folds]))

    print("\nGraficando resultados agregados...")
    plot_learning_curves(last_fold_history, N_FOLDS)
    plot_confusion_matrix_multiclass(y_true_all, y_pred_all)
    plot_pnl_curve(overall_pnl["gross_returns"], overall_pnl["net_returns"])

    pl.DataFrame({
        "epoch": list(range(1, len(last_fold_history.train_loss) + 1)),
        "train_loss": last_fold_history.train_loss,
        "val_loss": last_fold_history.val_loss,
        "val_accuracy": last_fold_history.val_accuracy,
    }).write_csv(REPORTS_DIR / "walkforward_last_fold_history.csv")

    summary = {
        "n_folds": N_FOLDS,
        "window_length": WINDOW_LENGTH,
        "deadzone_mult": DEADZONE_MULT,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "total_cost_bps": TOTAL_COST_BPS,
        "n_test_windows_total": int(len(y_true_all)),
        "overall_accuracy": overall_accuracy,
        "overall_f1_macro": overall_f1_macro,
        "overall_n_trades": overall_pnl["n_trades"],
        "overall_gross_return_total": overall_pnl["gross_return_total"],
        "overall_cost_total": overall_pnl["cost_total"],
        "overall_net_return_total": overall_pnl["net_return_total"],
        "overall_annualized_sharpe_like_net": overall_pnl["annualized_sharpe_like_net"],
        "random_baseline_accuracy": 1.0 / N_CLASSES,
        "folds": fold_reports,
    }
    with open(REPORTS_DIR / "walkforward_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    equity_df = pl.DataFrame({
        "y_true": y_true_all, "y_pred": y_pred_all,
        "gross_return": overall_pnl["gross_returns"], "net_return": overall_pnl["net_returns"],
        "gross_cum": np.cumsum(overall_pnl["gross_returns"]), "net_cum": np.cumsum(overall_pnl["net_returns"]),
    })
    equity_df.write_csv(REPORTS_DIR / "walkforward_equity_curve.csv")

    print(f"\n=== Resumen walk-forward (out-of-sample, {len(y_true_all):,} ventanas de test) ===")
    print(f"Accuracy: {overall_accuracy:.3f}  (baseline aleatorio 3 clases = {1/N_CLASSES:.3f})")
    print(f"F1 macro: {overall_f1_macro:.3f}")
    print(f"Trades: {overall_pnl['n_trades']:,}  Costo total: {overall_pnl['cost_total']:+.4f}")
    print(f"PnL bruto total: {overall_pnl['gross_return_total']:+.4f}  PnL neto total: {overall_pnl['net_return_total']:+.4f}")
    print(f"Sharpe anualizado (neto): {overall_pnl['annualized_sharpe_like_net']:.3f}")
    print(f"\nArtefactos guardados en {FIGURES_DIR}, {MODELS_DIR}, {REPORTS_DIR}")


if __name__ == "__main__":
    main()
