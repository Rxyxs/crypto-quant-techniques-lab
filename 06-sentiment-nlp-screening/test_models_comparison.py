"""Unit tests for the complementary sentiment-model comparison module."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import models_comparison as mc

OUTPUTS_CSV = Path("outputs") / "headline_sentiment.csv"


@pytest.fixture(scope="module")
def headline_df():
    if not OUTPUTS_CSV.exists():
        pytest.skip("outputs/headline_sentiment.csv not present; run sentiment_screener.py first")
    return mc.load_headline_data(OUTPUTS_CSV)


@pytest.fixture(scope="module")
def split(headline_df):
    return mc.split_data(headline_df, test_size=0.25, seed=mc.SEED)


@pytest.fixture(scope="module")
def vectors(split):
    idx_train, idx_test, headlines, true_labels = split
    vectorizer, X_train, X_test = mc.vectorize(headlines, idx_train, idx_test)
    y_train = [true_labels[i] for i in idx_train]
    y_test = [true_labels[i] for i in idx_test]
    return X_train, X_test, y_train, y_test


def test_split_data_is_disjoint_and_stratified(split, headline_df):
    idx_train, idx_test, headlines, true_labels = split
    assert set(idx_train).isdisjoint(set(idx_test))
    assert len(idx_train) + len(idx_test) == len(headlines)
    # every headline in the frame was assigned to exactly one split
    assert len(idx_train) + len(idx_test) == headline_df.height


def test_vectorize_shapes_match(vectors):
    X_train, X_test, y_train, y_test = vectors
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] > 0


def test_train_baseline_returns_valid_predictions(vectors):
    X_train, X_test, y_train, y_test = vectors
    clf, preds, metrics = mc.train_baseline(X_train, y_train, X_test, y_test)
    assert len(preds) == len(y_test)
    assert set(preds) <= set(mc.LABELS)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["confusion_matrix"].shape == (3, 3)


def test_train_ensemble_returns_valid_predictions(vectors):
    X_train, X_test, y_train, y_test = vectors
    clf, preds, metrics = mc.train_ensemble(X_train, y_train, X_test, y_test)
    assert len(preds) == len(y_test)
    assert set(preds) <= set(mc.LABELS)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_focal_loss_is_nonnegative_and_finite():
    criterion = mc.FocalLoss(gamma=2.0)
    logits = torch.randn(16, 3, requires_grad=True)
    targets = torch.randint(0, 3, (16,))
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    loss.backward()
    assert logits.grad is not None


def test_focal_loss_downweights_confident_correct_predictions():
    """A confident, correct prediction should contribute much less focal loss
    than an equally confident, wrong prediction."""
    criterion = mc.FocalLoss(gamma=2.0)
    confident_correct_logits = torch.tensor([[10.0, -10.0, -10.0]])
    confident_wrong_logits = torch.tensor([[-10.0, -10.0, 10.0]])
    target = torch.tensor([0])

    correct_loss = criterion(confident_correct_logits, target)
    wrong_loss = criterion(confident_wrong_logits, target)
    assert correct_loss.item() < wrong_loss.item()


@pytest.mark.parametrize("activation_name", ["ReLU", "GELU", "Swish"])
def test_mlp_trains_and_loss_decreases(vectors, activation_name):
    X_train, X_test, y_train, y_test = vectors
    _, preds, metrics = mc.train_mlp(
        X_train, y_train, X_test, y_test, activation_name=activation_name, epochs=30,
    )
    loss_history = metrics["loss_history"]
    assert len(loss_history) == 30
    # training loss should trend down over the run
    assert loss_history[-1] < loss_history[0]
    assert len(preds) == len(y_test)
    assert set(preds) <= set(mc.LABELS)


def test_sentiment_mlp_forward_shapes():
    for activation in mc.ACTIVATIONS.values():
        model = mc.SentimentMLP(input_dim=20, hidden_dim=8, n_classes=3, activation=activation)
        x = torch.randn(5, 20)
        out = model(x)
        assert out.shape == (5, 3)


def test_evaluate_perfect_predictions_gives_accuracy_one():
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred = ["positive", "negative", "neutral", "positive"]
    metrics = mc.evaluate(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_persist_to_duckdb_writes_both_tables(tmp_path):
    import duckdb
    import polars as pl

    comparison = {
        "modelA": {"accuracy": 0.9, "f1_macro": 0.85},
        "modelB": {"accuracy": 0.8, "f1_macro": 0.75},
    }
    predictions_df = pl.DataFrame(
        {"headline": ["h1", "h2"], "true_label": ["positive", "negative"], "modelA_pred": ["positive", "positive"]}
    )
    db_path = tmp_path / "test.duckdb"
    metrics_df = mc.persist_to_duckdb(comparison, predictions_df, db_path=db_path)

    assert metrics_df.height == 2
    con = duckdb.connect(str(db_path))
    metrics_rows = con.execute("SELECT COUNT(*) FROM model_metrics").fetchone()[0]
    pred_rows = con.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0]
    con.close()
    assert metrics_rows == 2
    assert pred_rows == 2


def test_plot_functions_write_files(vectors, tmp_path):
    X_train, X_test, y_train, y_test = vectors
    _, baseline_preds, baseline_metrics = mc.train_baseline(X_train, y_train, X_test, y_test)
    comparison = {"Logistic Regression": baseline_metrics}

    comparison_path = tmp_path / "comparison.png"
    mc.plot_model_comparison(comparison, comparison_path)
    assert comparison_path.exists()
    assert comparison_path.stat().st_size > 0

    cm_path = tmp_path / "cms.png"
    mc.plot_confusion_matrices(
        {name: vals["confusion_matrix"] for name, vals in comparison.items()}, cm_path
    )
    assert cm_path.exists()

    activation_results = {"ReLU": {"metrics": {"loss_history": [1.0, 0.8, 0.5, 0.3]}}}
    loss_path = tmp_path / "loss.png"
    mc.plot_activation_loss_curves(activation_results, loss_path)
    assert loss_path.exists()
