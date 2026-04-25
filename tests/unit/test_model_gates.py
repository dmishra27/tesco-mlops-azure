"""
Unit tests for ml.local.model_gates.
Written BEFORE model_gates.py exists (TDD RED phase).
"""

from __future__ import annotations

import pytest
from datetime import datetime

from ml.local.model_gates import GateFailure, run_segmentation_gates, run_propensity_gates


# ── Segmentation gate tests ───────────────────────────────────────────────────

def test_segmentation_low_silhouette_fails():
    """Silhouette < 0.25 means segments overlap — targeting no better than random."""
    with pytest.raises(GateFailure) as exc_info:
        run_segmentation_gates(silhouette_score=0.18, segment_sizes=[0.45, 0.35, 0.20])
    exc = exc_info.value
    assert "silhouette_score" in str(exc)
    assert "0.18" in str(exc)
    assert "0.25" in str(exc)


def test_segmentation_dominant_cluster_fails():
    """One segment with 95% of customers means differentiated campaigns are impossible."""
    with pytest.raises(GateFailure) as exc_info:
        run_segmentation_gates(silhouette_score=0.45, segment_sizes=[0.95, 0.03, 0.02])
    exc = exc_info.value
    assert "dominant_cluster" in str(exc)
    assert "95" in str(exc)
    assert "60" in str(exc)


def test_segmentation_tiny_cluster_fails():
    """A segment < 1% of customers is noise, not a real pattern."""
    with pytest.raises(GateFailure) as exc_info:
        run_segmentation_gates(silhouette_score=0.45, segment_sizes=[0.60, 0.39, 0.01])
    exc = exc_info.value
    assert "tiny_cluster" in str(exc)
    assert "1" in str(exc)


def test_propensity_low_auc_fails():
    """AUC < 0.65 barely outperforms random — campaign ROI would not justify personalisation."""
    with pytest.raises(GateFailure) as exc_info:
        run_propensity_gates(test_auc=0.58, train_auc=0.60,
                             previous_production_auc=0.70, lift_at_decile1=3.0)
    exc = exc_info.value
    assert "test_auc" in str(exc)
    assert "0.58" in str(exc)
    assert "0.65" in str(exc)


def test_propensity_auc_regression_fails():
    """New model must not be significantly worse than the model currently in production."""
    with pytest.raises(GateFailure) as exc_info:
        run_propensity_gates(test_auc=0.71, train_auc=0.75,
                             previous_production_auc=0.80, lift_at_decile1=3.0)
    exc = exc_info.value
    assert "auc_regression" in str(exc)
    assert "0.09" in str(exc)


def test_propensity_overfit_gate_fails():
    """Large train/test gap means model memorised history rather than learning patterns."""
    with pytest.raises(GateFailure) as exc_info:
        run_propensity_gates(test_auc=0.71, train_auc=0.95,
                             previous_production_auc=0.68, lift_at_decile1=3.0)
    exc = exc_info.value
    assert "overfitting" in str(exc)
    assert "0.24" in str(exc)


def test_propensity_low_lift_fails():
    """Top decile lift < 2.5 means targeted promotion economics break down."""
    with pytest.raises(GateFailure) as exc_info:
        run_propensity_gates(test_auc=0.82, train_auc=0.84,
                             previous_production_auc=0.78, lift_at_decile1=1.8)
    exc = exc_info.value
    assert "lift_at_decile1" in str(exc)
    assert "1.8" in str(exc)
    assert "2.5" in str(exc)


def test_all_gates_pass_returns_report():
    """When all gates pass, a structured report must be returned for MLflow logging."""
    report = run_segmentation_gates(
        silhouette_score=0.42,
        segment_sizes=[0.45, 0.35, 0.20],
    )
    assert report["passed"] is True
    assert report["gates_checked"] > 0
    assert report["gates_passed"] == report["gates_checked"]
    assert isinstance(report["metrics"], dict)
    assert isinstance(report["timestamp"], datetime)

    report2 = run_propensity_gates(
        test_auc=0.847, train_auc=0.853,
        previous_production_auc=0.831, lift_at_decile1=4.1,
    )
    assert report2["passed"] is True
    assert report2["gates_checked"] > 0


def test_gate_failure_contains_structured_data():
    """GateFailure must be machine-readable so CI/CD can log it without string parsing."""
    with pytest.raises(GateFailure) as exc_info:
        run_propensity_gates(test_auc=0.58, train_auc=0.60,
                             previous_production_auc=0.65, lift_at_decile1=3.0)
    exc = exc_info.value
    assert hasattr(exc, "gate_name")
    assert hasattr(exc, "actual_value")
    assert hasattr(exc, "threshold")
    assert hasattr(exc, "business_impact")
    assert isinstance(exc.business_impact, str)
    assert len(exc.business_impact) > 0
