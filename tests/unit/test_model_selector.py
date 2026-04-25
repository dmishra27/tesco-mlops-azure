"""
Unit tests for ml.local.model_selection.ModelSelector.
Written BEFORE model_selection.py exists (TDD RED phase).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from ml.local.model_selection import ModelSelector, NoModelApprovedError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_model(name: str) -> MagicMock:
    m = MagicMock()
    m.__class__.__name__ = name
    return m


def _metrics(**kwargs) -> dict:
    """Build a metrics entry with sensible defaults that pass all gates."""
    defaults = {
        "test_auc":         0.85,
        "train_auc":        0.87,
        "cv_std":           0.02,
        "lift_at_decile1":  3.5,
    }
    defaults.update(kwargs)
    return defaults


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_baseline_gate_rejects_weak_model():
    """Complexity is only justified if it delivers >= 0.03 AUC gain over LR."""
    selector = ModelSelector()
    models = {
        "logistic_regression": _mock_model("LR"),
        "random_forest":       _mock_model("RF"),
    }
    metrics = {
        "logistic_regression": _metrics(test_auc=0.71, train_auc=0.715),
        "random_forest":       _metrics(test_auc=0.73, train_auc=0.735),  # gain=0.02 < 0.03
    }
    result = selector.select(models, metrics)
    assert result["selected_model_name"] != "random_forest"
    rf_rejection = next(r for r in result["rejected_models"] if r["model_name"] == "random_forest")
    assert "baseline gate"  in rf_rejection["reason"]
    assert "0.02"           in rf_rejection["reason"]
    assert "0.03"           in rf_rejection["reason"]


def test_overfit_gate_rejects_model():
    """Train-test gap >= 0.08 means overfitting — model is rejected."""
    selector = ModelSelector()
    models = {
        "logistic_regression": _mock_model("LR"),
        "random_forest":       _mock_model("RF"),
    }
    metrics = {
        "logistic_regression": _metrics(test_auc=0.71, train_auc=0.715),
        "random_forest":       _metrics(test_auc=0.79, train_auc=0.91),  # gap=0.12 > 0.08
    }
    result = selector.select(models, metrics)
    rf_rejection = next(r for r in result["rejected_models"] if r["model_name"] == "random_forest")
    assert "overfit gate" in rf_rejection["reason"]


def test_occams_razor_prefers_simpler():
    """Within 0.01 AUC difference, prefer the simpler model."""
    selector = ModelSelector()
    models = {
        "logistic_regression": _mock_model("LR"),
        "random_forest":       _mock_model("RF"),
        "lightgbm":            _mock_model("LGBM"),
    }
    metrics = {
        "logistic_regression": _metrics(test_auc=0.71,  train_auc=0.715),
        "random_forest":       _metrics(test_auc=0.841, train_auc=0.845),
        "lightgbm":            _metrics(test_auc=0.847, train_auc=0.850),  # gap=0.006
    }
    result = selector.select(models, metrics)
    assert result["selected_model_name"] == "random_forest"
    assert "Occam" in result["selection_reason"]
    assert "0.006" in result["selection_reason"]


def test_ensemble_justification_threshold():
    """Ensemble only wins if AUC gain > 0.015 over best single model."""
    selector = ModelSelector()
    models = {
        "logistic_regression": _mock_model("LR"),
        "lightgbm":            _mock_model("LGBM"),
        "stacking_ensemble":   _mock_model("Stack"),
    }
    metrics = {
        "logistic_regression": _metrics(test_auc=0.71,  train_auc=0.715),
        "lightgbm":            _metrics(test_auc=0.843, train_auc=0.848),
        "stacking_ensemble":   _metrics(test_auc=0.851, train_auc=0.855),  # gain=0.008 < 0.015
    }
    result = selector.select(models, metrics)
    assert result["selected_model_name"] == "lightgbm"
    ens_rejection = next(r for r in result["rejected_models"] if r["model_name"] == "stacking_ensemble")
    assert "ensemble justification" in ens_rejection["reason"]
    assert "0.008" in ens_rejection["reason"]
    assert "0.015" in ens_rejection["reason"]


def test_stability_gate_rejects_model():
    """cv_std >= 0.03 means performance is too variable across time periods."""
    selector = ModelSelector()
    models = {
        "logistic_regression": _mock_model("LR"),
        "lightgbm":            _mock_model("LGBM"),
    }
    metrics = {
        "logistic_regression": _metrics(test_auc=0.71,  train_auc=0.715),
        "lightgbm":            _metrics(test_auc=0.849, train_auc=0.852, cv_std=0.045),
    }
    result = selector.select(models, metrics)
    lgbm_rejection = next(r for r in result["rejected_models"] if r["model_name"] == "lightgbm")
    assert "stability gate"  in lgbm_rejection["reason"]
    assert "0.045"           in lgbm_rejection["reason"]
    assert "0.030"           in lgbm_rejection["reason"]


def test_no_model_passes_all_gates():
    """If no model passes all gates, NoModelApprovedError is raised."""
    selector = ModelSelector()
    models = {
        "logistic_regression": _mock_model("LR"),
        "random_forest":       _mock_model("RF"),
    }
    metrics = {
        "logistic_regression": _metrics(test_auc=0.71, train_auc=0.715, cv_std=0.04),  # LR fails G4
        "random_forest":       _metrics(test_auc=0.73, train_auc=0.735),               # RF fails G1
    }
    with pytest.raises(NoModelApprovedError) as exc_info:
        selector.select(models, metrics)
    msg = str(exc_info.value)
    assert "no model passed all quality gates" in msg
    assert "logistic_regression" in msg
    assert "random_forest"       in msg


def test_selection_returns_full_report():
    """Selection report must be fully loggable to MLflow without extra processing."""
    selector = ModelSelector()
    models = {
        "logistic_regression": _mock_model("LR"),
        "lightgbm":            _mock_model("LGBM"),
    }
    metrics = {
        "logistic_regression": _metrics(test_auc=0.71,  train_auc=0.715),
        "lightgbm":            _metrics(test_auc=0.852, train_auc=0.855),
    }
    result = selector.select(models, metrics)
    assert isinstance(result["selected_model_name"], str)
    assert result["selected_model_object"] is not None
    assert isinstance(result["selection_reason"],  str)
    assert isinstance(result["all_gate_results"],  dict)
    assert isinstance(result["rejected_models"],   list)
    assert isinstance(result["metrics_table"],     list)
