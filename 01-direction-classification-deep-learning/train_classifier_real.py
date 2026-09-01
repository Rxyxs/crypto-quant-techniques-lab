"""
Modo "datos reales": corre el mismo pipeline de train_classifier.py
(ingenieria de features, split cronologico, comparacion ReLU vs Tanh,
evaluacion) pero sobre velas OHLCV reales de BTCUSDT descargadas de Binance
en vez del dataset sintetico, para medir honestamente la diferencia entre
ambos.

Requiere haber corrido fetch_binance_data.py antes (o lo corre automatico
si data/ohlcv_real_binance.csv no existe).

Uso:
    .\\venv\\Scripts\\python.exe train_classifier_real.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from fetch_binance_data import OUTPUT_PATH as REAL_DATA_PATH
from fetch_binance_data import fetch_real_ohlcv
from train_classifier import (
    DATA_DIR,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    TRAIN_FRACTION,
    VAL_FRACTION,
    build_features_and_label,
    plot_confusion_matrix,
    plot_precision_recall_table,
    plot_training_comparison,
    set_seeds,
    train_variant,
)

CANDLESTICK_WINDOW = 200  # velas mas recientes mostradas en el grafico de velas real


def load_or_fetch_real_data() -> pl.DataFrame:
    if REAL_DATA_PATH.exists():
        print(f"Cargando datos reales ya descargados desde {REAL_DATA_PATH}")
        return pl.read_csv(REAL_DATA_PATH, try_parse_dates=True)
    print("No se encontraron datos reales locales, descargando desde Binance...")
    df = fetch_real_ohlcv()
    df.write_csv(REAL_DATA_PATH)
    return df


def plot_real_candlestick(raw: pl.DataFrame) -> None:
    """Grafico de velas real (no sintetico) de las ultimas CANDLESTICK_WINDOW
    velas de BTCUSDT, tal como se ven en Binance -- la pieza visual que
    complementa los resultados del modelo con la serie de precios real de la
    que provienen."""
    window = raw.tail(CANDLESTICK_WINDOW).to_pandas().set_index("timestamp")
    window = window.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    fig, axes = mpf.plot(
        window,
        type="candle",
        style="binance",
        volume=True,
        title=f"\nBTCUSDT 1h -- ultimas {CANDLESTICK_WINDOW} velas reales (Binance)",
        returnfig=True,
        figsize=(12, 6),
    )
    fig.savefig(FIGURES_DIR / "real_candlestick_btcusdt.png", dpi=150)
    plt.close(fig)


def main() -> None:
    set_seeds(SEED)
    for d in (DATA_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    raw = load_or_fetch_real_data()
    print(f"\n{len(raw):,} velas reales de BTCUSDT cargadas ({raw['timestamp'].min()} -> {raw['timestamp'].max()})")

    print("\nGraficando velas reales recientes (Binance)...")
    plot_real_candlestick(raw)

    df = build_features_and_label(raw)
    print(f"  {len(df):,} velas utilizables tras calcular features")
    bullish_rate = df["label_next_bullish"].mean()
    print(f"  balance de clases real: {bullish_rate:.1%} alcistas / {1 - bullish_rate:.1%} bajistas")

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

    print("\nEntrenando variante ReLU sobre datos reales...")
    model_relu, history_relu = train_variant("relu", X_train, y_train, X_val, y_val, device)
    print("\nEntrenando variante Tanh sobre datos reales...")
    model_tanh, history_tanh = train_variant("tanh", X_train, y_train, X_val, y_val, device)

    plot_training_comparison(history_relu, history_tanh, suffix="_real")

    if history_relu.best_val_auc >= history_tanh.best_val_auc:
        winner_name, winner_model, winner_history = "relu", model_relu, history_relu
    else:
        winner_name, winner_model, winner_history = "tanh", model_tanh, history_tanh

    print(
        f"\nActivacion ganadora (datos reales) por AUC de validacion: {winner_name.upper()} "
        f"(ReLU AUC={history_relu.best_val_auc:.3f} vs Tanh AUC={history_tanh.best_val_auc:.3f})"
    )

    torch.save(model_relu.state_dict(), MODELS_DIR / "classifier_relu_real.pt")
    torch.save(model_tanh.state_dict(), MODELS_DIR / "classifier_tanh_real.pt")

    winner_model.eval()
    with torch.no_grad():
        test_logits = winner_model(torch.from_numpy(X_test).to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    y_pred = (test_probs >= 0.5).astype(np.int8)
    y_true = y_test.astype(np.int8)

    test_auc = roc_auc_score(y_true, test_probs)
    print(f"\nEvaluacion en test real (modelo {winner_name.upper()}): AUC={test_auc:.3f}")

    plot_confusion_matrix(y_true, y_pred, winner_name, suffix="_real")
    metrics = plot_precision_recall_table(y_true, y_pred, winner_name, suffix="_real")
    metrics["test_auc"] = float(test_auc)
    metrics["winner_activation"] = winner_name
    metrics["relu_best_val_auc"] = history_relu.best_val_auc
    metrics["tanh_best_val_auc"] = history_tanh.best_val_auc
    metrics["n_train"] = len(train_df)
    metrics["n_val"] = len(val_df)
    metrics["n_test"] = len(test_df)
    metrics["data_range_start"] = str(raw["timestamp"].min())
    metrics["data_range_end"] = str(raw["timestamp"].max())

    with open(REPORTS_DIR / "metrics_real.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAccuracy test (real): {metrics['accuracy']:.3f}")

    synthetic_metrics_path = REPORTS_DIR / "metrics.json"
    if synthetic_metrics_path.exists():
        with open(synthetic_metrics_path) as f:
            synthetic_metrics = json.load(f)
        print("\n--- Sintetico vs. real ---")
        print(f"  Accuracy:  sintetico={synthetic_metrics['accuracy']:.3f}  real={metrics['accuracy']:.3f}")
        print(f"  ROC-AUC:   sintetico={synthetic_metrics['test_auc']:.3f}  real={metrics['test_auc']:.3f}")

    print(f"\nArtefactos guardados en {DATA_DIR}, {FIGURES_DIR}, {MODELS_DIR}, {REPORTS_DIR}")


if __name__ == "__main__":
    main()
