"""Modelos de clasificación para la señal direccional (up/down), compartidos
entre `backtest_engine.py` y `compare_strategies.py`: una regresión
logística (baseline lineal) y un clasificador LightGBM (Gradient Boosting).

Ambos se entrenan sobre exactamente el mismo split cronológico y el mismo
conjunto de features rezagadas (sin lookahead) -- la única variable que
cambia entre modelos es la arquitectura del clasificador, así que cualquier
diferencia de desempeño en `compare_strategies.py` es atribuible al modelo,
no a una ventaja de datos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42

LGBM_PARAMS = dict(
    n_estimators=200,
    num_leaves=15,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=SEED,
    verbosity=-1,
)


def train_logistic_regression(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (señal 0/1, probabilidad de subida) sobre `test_df`."""
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(train_df[feature_cols], train_df["label_up"])
    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    return (proba >= 0.5).astype(int), proba


def train_lightgbm(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (señal 0/1, probabilidad de subida) sobre `test_df`.

    `class_weight="balanced"` no es necesario aquí (la clase up/down en un
    activo cripto está cerca de 50/50, a diferencia de un problema de fraude
    o churn), así que se omite deliberadamente.
    """
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(train_df[feature_cols], train_df["label_up"])
    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    return (proba >= 0.5).astype(int), proba


def feature_importance(train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Reentrenar-y-extraer no hace falta: se entrena una vez más sobre todo
    train para reportar importancia de features del modelo LightGBM."""
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(train_df[feature_cols], train_df["label_up"])
    return pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
