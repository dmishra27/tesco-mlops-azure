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


def test_no_model_approved_error_raised():
    """
    NoModelApprovedError must be raised when every model fails at least one gate.
    This test exercises the G3 lift-gate failure path (lines 140, 180) which is
    not covered by the existing test_no_model_passes_all_gates (which fails on G4).
    - LR fails G3: lift_at_decile1 1.9 < threshold 2.5
    - RF fails G1: AUC gain over LR is only 0.02 < required 0.03
    """
    selector = ModelSelector()
    models = {
        "logistic_regression": _mock_model("LR"),
        "random_forest":       _mock_model("RF"),
    }
    metrics = {
        "logistic_regression": _metrics(
            test_auc=0.75, train_auc=0.755,
            lift_at_decile1=1.9,   # G3 FAIL: 1.9 < 2.5
        ),
        "random_forest": _metrics(
            test_auc=0.77, train_auc=0.775,   # G1 FAIL: gain 0.02 < 0.03
        ),
    }
    with pytest.raises(NoModelApprovedError) as exc:
        selector.select(models, metrics)
    msg = str(exc.value)
    assert "logistic_regression" in msg
    assert "random_forest"       in msg
    assert "lift gate"           in msg   # confirms the G3 branch was reached


def test_tiebreaker_with_three_equal_models():
    """
    When three models all pass the quality gates and two of them (RF and LightGBM)
    have AUC scores within the 0.01 tiebreaker delta of the best, Occam's razor
    must select the simpler one (RF over LightGBM).
    LR is approved but sits outside the tiebreaker window (0.043 below best).
    """
    selector = ModelSelector()
    models = {
        "logistic_regression": _mock_model("LR"),
        "random_forest":       _mock_model("RF"),
        "lightgbm":            _mock_model("LGBM"),
    }
    metrics = {
        # LR auto-passes G1; all other gates pass with these values
        "logistic_regression": _metrics(test_auc=0.800, train_auc=0.804),
        # RF beats LR by 0.036 (> 0.03 threshold) — passes G1
        "random_forest":       _metrics(test_auc=0.836, train_auc=0.840),
        # LightGBM beats LR by 0.043 (> 0.03 threshold) — passes G1
        # Gap between RF and LightGBM is 0.007 <= tiebreaker delta 0.01
        "lightgbm":            _metrics(test_auc=0.843, train_auc=0.847),
    }
    result = selector.select(models, metrics)

    # Occam's razor: RF and LightGBM are in the tie window; RF is simpler
    assert result["selected_model_name"] == "random_forest"
    assert "Occam" in result["selection_reason"]
    # All three pass their gates — no rejections
    assert len(result["rejected_models"]) == 0
    # Full report must include every model
    reported_names = {row["model"] for row in result["metrics_table"]}
    assert {"logistic_regression", "random_forest", "lightgbm"} == reported_names


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
