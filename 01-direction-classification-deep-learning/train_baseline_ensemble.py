"""
Modelos complementarios al clasificador denso en PyTorch (train_classifier.py):

1. Un baseline interpretable -- Regresion Logistica -- para tener un piso de
   comparacion simple y explicable (coeficientes con signo claro por feature).
2. Un ensamble de arboles -- LightGBM (Gradient Boosting) -- que suele ser
   competitivo o superior a una red densa pequena sobre features tabulares
   como las de este proyecto, y sirve de referencia "no-deep-learning" fuerte.

Reutiliza exactamente la misma simulacion de datos, ingenieria de features y
split cronologico de train_classifier.py para que la comparacion entre los
tres enfoques (Logistic Regression / LightGBM / Dense NN) sea justa (mismos
datos, mismo split, mismas features).

Persiste las metricas de los tres enfoques en outputs/reports/model_db.duckdb
(tabla `model_comparison`) para poder consultarlas con SQL despues.

Uso:
    .\\venv\\Scripts\\python.exe train_baseline_ensemble.py
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from train_classifier import (
    DATA_DIR,
    FEATURE_COLUMNS,
    N_CANDLES,
    SEED,
    TRAIN_FRACTION,
    VAL_FRACTION,
    build_features_and_label,
    set_seeds,
    simulate_ohlcv_data,
)

ROOT = Path(__file__).parent
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"
DB_PATH = REPORTS_DIR / "model_db.duckdb"


def load_dataset():
    """Carga el CSV simulado ya escrito por train_classifier.py si existe;
    si no, lo regenera con la misma semilla (determinista) para no depender
    del orden de ejecucion de los scripts."""
    csv_path = DATA_DIR / "ohlcv_simulated.csv"
    if csv_path.exists():
        import polars as pl

        raw = pl.read_csv(csv_path, try_parse_dates=True)
    else:
        raw = simulate_ohlcv_data(N_CANDLES, SEED)
    return build_features_and_label(raw)


def chronological_split(df):
    n = len(df)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)
    train_df = df[:n_train]
    val_df = df[n_train : n_train + n_val]
    test_df = df[n_train + n_val :]
    return train_df, val_df, test_df


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(np.int8)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    return {
        "model": name,
        "accuracy": float((y_true == y_pred).mean()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision_bajista": float(precision[0]),
        "recall_bajista": float(recall[0]),
        "f1_bajista": float(f1[0]),
        "precision_alcista": float(precision[1]),
        "recall_alcista": float(recall[1]),
        "f1_alcista": float(f1[1]),
        "n_test": int(len(y_true)),
    }


def plot_roc_comparison(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = {"logistic_regression": "#4C72B0", "lightgbm": "#55A868", "dense_nn_tanh": "#DD8452"}
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors.get(name, "gray"))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Azar (AUC=0.500)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC -- comparacion de los 3 enfoques (test)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_comparison_3models.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model: LGBMClassifier) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=importances[order],
        y=[FEATURE_COLUMNS[i] for i in order],
        color="#55A868",
        ax=ax,
    )
    ax.set_xlabel("Importancia (gain)")
    ax.set_title("LightGBM -- importancia de features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lightgbm_feature_importance.png", dpi=150)
    plt.close(fig)


def persist_to_duckdb(rows: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS model_comparison (
            run_ts TIMESTAMP DEFAULT current_timestamp,
            model VARCHAR,
            accuracy DOUBLE,
            roc_auc DOUBLE,
            precision_bajista DOUBLE,
            recall_bajista DOUBLE,
            f1_bajista DOUBLE,
            precision_alcista DOUBLE,
            recall_alcista DOUBLE,
            f1_alcista DOUBLE,
            n_test INTEGER
        )
        """
    )
    for row in rows:
        con.execute(
            """
            INSERT INTO model_comparison
            (model, accuracy, roc_auc, precision_bajista, recall_bajista, f1_bajista,
             precision_alcista, recall_alcista, f1_alcista, n_test)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["model"],
                row["accuracy"],
                row["roc_auc"],
                row["precision_bajista"],
                row["recall_bajista"],
                row["f1_bajista"],
                row["precision_alcista"],
                row["recall_alcista"],
                row["f1_alcista"],
                row["n_test"],
            ],
        )
    print("\nContenido de model_comparison en DuckDB:")
    print(con.execute("SELECT model, accuracy, roc_auc FROM model_comparison ORDER BY run_ts DESC LIMIT 10").fetchdf())
    con.close()


def main() -> None:
    set_seeds(SEED)
    for d in (FIGURES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset simulado (mismas features/split que train_classifier.py)...")
    df = load_dataset()
    train_df, val_df, test_df = chronological_split(df)
    print(f"  split cronologico -> train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    X_train = train_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float64)
    X_val = val_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float64)
    X_test = test_df.select(FEATURE_COLUMNS).to_numpy().astype(np.float64)
    y_train = train_df["label_next_bullish"].to_numpy().astype(np.int8)
    y_val = val_df["label_next_bullish"].to_numpy().astype(np.int8)
    y_test = test_df["label_next_bullish"].to_numpy().astype(np.int8)

    # Logistic regression usa train+val (no necesita early stopping por val loss).
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    scaler = StandardScaler().fit(X_trainval)
    X_trainval_scaled = scaler.transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)

    print("\n[1/2] Entrenando baseline: Regresion Logistica...")
    logreg = LogisticRegression(max_iter=1000, random_state=SEED)
    logreg.fit(X_trainval_scaled, y_trainval)
    logreg_prob = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_metrics = evaluate("logistic_regression", y_test, logreg_prob)
    print(f"  accuracy={logreg_metrics['accuracy']:.3f}  roc_auc={logreg_metrics['roc_auc']:.3f}")

    print("\n[2/2] Entrenando ensamble de arboles: LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbosity=-1,
    )
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[],
    )
    lgbm_prob = lgbm.predict_proba(X_test)[:, 1]
    lgbm_metrics = evaluate("lightgbm", y_test, lgbm_prob)
    print(f"  accuracy={lgbm_metrics['accuracy']:.3f}  roc_auc={lgbm_metrics['roc_auc']:.3f}")

    plot_feature_importance(lgbm)

    # Curva ROC comparativa: incluye tambien la NN densa (Tanh) del pipeline
    # principal, leyendo sus metricas ya persistidas si estan disponibles.
    curves = {
        "logistic_regression": (*roc_curve(y_test, logreg_prob)[:2], logreg_metrics["roc_auc"]),
        "lightgbm": (*roc_curve(y_test, lgbm_prob)[:2], lgbm_metrics["roc_auc"]),
    }
    dense_metrics_path = REPORTS_DIR / "metrics.json"
    dense_row = None
    if dense_metrics_path.exists():
        with open(dense_metrics_path) as f:
            dense = json.load(f)
        # No se guardaron las probabilidades de test de la NN densa, asi que
        # aproximamos su curva ROC con el punto (test_auc) usando una linea
        # recta placeholder solo si no hay datos crudos; preferimos omitirla
        # del grafico si no hay fpr/tpr reales para no inducir a error.
        dense_row = {
            "model": "dense_nn_" + dense.get("winner_activation", "nn"),
            "accuracy": dense.get("accuracy"),
            "roc_auc": dense.get("test_auc"),
            "precision_bajista": dense.get("precision_bajista"),
            "recall_bajista": dense.get("recall_bajista"),
            "f1_bajista": dense.get("f1_bajista"),
            "precision_alcista": dense.get("precision_alcista"),
            "recall_alcista": dense.get("recall_alcista"),
            "f1_alcista": dense.get("f1_alcista"),
            "n_test": dense.get("n_test"),
        }

    plot_roc_comparison(curves)

    rows = [logreg_metrics, lgbm_metrics]
    if dense_row is not None:
        rows.append(dense_row)

    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)

    persist_to_duckdb(rows)

    print("\nResumen comparativo:")
    for row in rows:
        print(f"  {row['model']:<22} accuracy={row['accuracy']:.3f}  roc_auc={row['roc_auc']:.3f}")

    print(f"\nArtefactos guardados en {FIGURES_DIR} y {REPORTS_DIR} (incl. {DB_PATH.name})")


if __name__ == "__main__":
    main()
