"""
Unit tests for ml.local.generate_drift_data.
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.local.generate_drift_data import (
    FEATURE_COLS,
    compute_psi,
    generate_drifted_features,
    generate_stable_features,
)

NUMERICAL_COLS = [c for c in FEATURE_COLS if c != "has_promoted_category"]


def test_stable_features_psi_below_threshold():
    """
    Two independent samples from the same stable distribution should have
    PSI < 0.05 on all numerical features.
    """
    stable1 = generate_stable_features(n_customers=2000, seed=42)
    stable2 = generate_stable_features(n_customers=2000, seed=123)

    for feat in NUMERICAL_COLS:
        psi = compute_psi(stable1[feat].values, stable2[feat].values)
        assert psi < 0.05, (
            f"PSI for '{feat}' = {psi:.4f} between two stable samples (expected < 0.05)"
        )


def test_drifted_features_psi_above_threshold():
    """
    Drifted features must have PSI > 0.20; non-drifted features must stay < 0.10.
    """
    drift_feats = ["recency_days", "online_ratio"]
    stable  = generate_stable_features(n_customers=2000, seed=42)
    drifted = generate_drifted_features(
        n_customers=2000,
        drift_features=drift_feats,
        drift_magnitude=0.25,
        seed=42,
    )

    for feat in drift_feats:
        psi = compute_psi(stable[feat].values, drifted[feat].values)
        assert psi > 0.20, (
            f"PSI for drifted '{feat}' = {psi:.4f} (expected > 0.20)"
        )

    non_drift = [c for c in NUMERICAL_COLS if c not in drift_feats]
    for feat in non_drift:
        psi = compute_psi(stable[feat].values, drifted[feat].values)
        assert psi < 0.10, (
            f"PSI for non-drifted '{feat}' = {psi:.4f} (expected < 0.10)"
        )
