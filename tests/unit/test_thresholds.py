"""
Unit tests for ml.config.thresholds — single source of truth for all gate thresholds.
"""

from __future__ import annotations

from ml.config.thresholds import SELECTION_THRESHOLDS

REQUIRED_KEYS = {
    "propensity_auc_min",
    "overfit_gap_max",
    "lift_decile1_min",
    "cv_std_max",
    "silhouette_min",
    "dominant_cluster_max",
    "tiny_cluster_min",
    "tiebreaker_delta",
    "ensemble_justification_delta",
    "baseline_gate_min_gain",
}


def test_all_required_keys_exist():
    """SELECTION_THRESHOLDS must contain all 10 canonical threshold keys."""
    missing = REQUIRED_KEYS - SELECTION_THRESHOLDS.keys()
    assert not missing, f"Missing keys: {missing}"


def test_no_threshold_is_none_or_negative():
    """Every threshold must be non-None and strictly positive."""
    for key, value in SELECTION_THRESHOLDS.items():
        assert value is not None, f"{key} is None"
        assert value > 0, f"{key} = {value} is not positive"


def test_model_gate_uses_correct_auc_threshold():
    """propensity_auc_min must be 0.70 — regression guard against silent drift."""
    assert SELECTION_THRESHOLDS["propensity_auc_min"] == 0.70


def test_tiebreaker_delta_less_than_baseline_gain():
    """tiebreaker_delta (0.01) must be < baseline_gate_min_gain (0.03).
    If this inverts, Occam's razor would prefer simpler models that cannot
    beat the baseline, breaking the selection logic."""
    assert (
        SELECTION_THRESHOLDS["tiebreaker_delta"]
        < SELECTION_THRESHOLDS["baseline_gate_min_gain"]
    )
