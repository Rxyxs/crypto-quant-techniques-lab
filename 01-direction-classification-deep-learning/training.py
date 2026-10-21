"""Bucle de entrenamiento compartido por los clasificadores del proyecto
(denso en ``train_classifier.py`` y Conv1D+Attention en ``train_walkforward.py``).

- Al menos ``min_epochs`` epocas (o ``max_epochs`` si es menor); desde ahi,
  early stopping cuando la loss de validacion no mejora en ``patience``
  epocas, restaurando los pesos de la mejor epoca.
- La validacion la elige quien llama y siempre es un tramo *anterior* al test
  (orden temporal); el test nunca entra aqui.
- Checkpoint del estado completo cada epoca, con escritura atomica: si el
  proceso se corta, la siguiente corrida con los mismos datos y configuracion
  retoma desde la ultima epoca guardada.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MIN_EPOCHS = 100
MAX_EPOCHS = 300
PATIENCE = 15


def _fingerprint(*arrays: np.ndarray, **config) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def fit(
    model: nn.Module,
    criterion: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    device: torch.device,
    batch_size: int, lr: float, seed: int,
    val_metrics: Callable[[torch.Tensor], dict] | None = None,
    min_epochs: int = MIN_EPOCHS, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
    checkpoint_path: Path | None = None, log: Callable[[int, dict], None] | None = None,
) -> dict:
    """Entrena ``model`` y devuelve la historia por epoca:
    {"train_loss", "val_loss", <metricas de val_metrics>..., "best_epoch", "epochs_run", "at_best"}."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=batch_size, shuffle=True, generator=generator)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    fp = _fingerprint(X_train, y_train, X_val, y_val, model=repr(model), criterion=repr(criterion),
                      batch_size=batch_size, lr=lr, seed=seed, min_epochs=min_epochs,
                      max_epochs=max_epochs, patience=patience)
    state = {"epoch": 0, "history": {"train_loss": [], "val_loss": []}, "best_val": float("inf"),
             "best_epoch": 0, "best_state": None, "at_best": {}, "bad_epochs": 0, "done": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            state = ckpt["state"]
            print(f"  retomando desde el checkpoint en la epoca {state['epoch']}")

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state}, tmp)
        tmp.replace(path)

    while not state["done"] and state["epoch"] < max_epochs:
        generator.manual_seed(seed + state["epoch"])  # mismo orden de batches al retomar
        torch.manual_seed(seed + 10_000 + state["epoch"])  # mismas mascaras de dropout al retomar
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        state["epoch"] += 1

        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            row = {"train_loss": total / len(X_train), "val_loss": criterion(logits, y_val_t).item()}
            if val_metrics is not None:
                row.update(val_metrics(logits))
        for k, v in row.items():
            state["history"].setdefault(k, []).append(v)
        if log is not None:
            log(state["epoch"], row)

        if row["val_loss"] < state["best_val"] - 1e-6:
            state.update(best_val=row["val_loss"], best_epoch=state["epoch"], bad_epochs=0, at_best=dict(row),
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if state["epoch"] >= min(min_epochs, max_epochs) and state["bad_epochs"] >= patience:
            state["done"] = True
        save_checkpoint()

    if state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()
    return {**state["history"], "best_epoch": state["best_epoch"], "epochs_run": state["epoch"],
            "at_best": state["at_best"]}
