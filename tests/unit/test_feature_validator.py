"""
Unit tests for ml.local.feature_validator.validate_features.
Written BEFORE feature_validator.py exists (TDD RED phase).

Business context: catching data quality issues at the feature table boundary
prevents corrupted predictions from reaching production CRM systems.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from ml.local.feature_validator import validate_features  # FAILS until file exists

REQUIRED_COLUMNS = [
    "customer_id", "recency_days", "frequency", "monetary",
    "avg_basket_size", "basket_std", "online_ratio", "active_days",
]


def _valid_df(n: int = 20) -> pd.DataFrame:
    """Returns a fully valid feature DataFrame."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        rows.append({
            "customer_id":     f"CUST-{i:05d}",
            "recency_days":    float(rng.integers(1, 90)),
            "frequency":       float(rng.integers(1, 30)),
            "monetary":        round(float(rng.uniform(10.0, 800.0)), 2),
            "avg_basket_size": round(float(rng.uniform(5.0, 80.0)), 2),
            "basket_std":      round(float(rng.uniform(0.0, 20.0)), 2),
            "online_ratio":    round(float(rng.uniform(0.0, 1.0)), 4),
            "active_days":     float(rng.integers(1, 30)),
        })
    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_negative_recency_raises():
    """recency_days represents days since last purchase; negative is impossible."""
    df = _valid_df()
    df.loc[0, "recency_days"] = -1.0
    with pytest.raises(ValueError) as exc_info:
        validate_features(df)
    msg = str(exc_info.value)
    assert "recency_days must be >= 0" in msg
    assert "-1" in msg


def test_zero_frequency_raises():
    """A customer in the feature table must have at least one transaction."""
    df = _valid_df()
    df.loc[0, "frequency"] = 0.0
    with pytest.raises(ValueError, match="frequency must be >= 1"):
        validate_features(df)


def test_online_ratio_above_one_raises():
    """online_ratio is a proportion; >1 indicates a division error upstream."""
    df = _valid_df()
    df.loc[0, "online_ratio"] = 1.5
    with pytest.raises(ValueError, match="online_ratio must be between 0 and 1"):
        validate_features(df)


def test_online_ratio_negative_raises():
    """online_ratio cannot be negative."""
    df = _valid_df()
    df.loc[0, "online_ratio"] = -0.1
    with pytest.raises(ValueError, match="online_ratio must be between 0 and 1"):
        validate_features(df)


def test_duplicate_customer_id_raises():
    """Each customer must appear exactly once; duplicates inflate their influence."""
    df = _valid_df(n=10)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate row 0
    with pytest.raises(ValueError) as exc_info:
        validate_features(df)
    msg = str(exc_info.value)
    assert "duplicate customer_id" in msg
    assert "1" in msg  # count of duplicates


def test_missing_required_column_raises():
    """All trained-on features must be present at inference time."""
    df = _valid_df().drop(columns=["monetary"])
    with pytest.raises(ValueError) as exc_info:
        validate_features(df)
    assert "missing required column" in str(exc_info.value)
    assert "monetary" in str(exc_info.value)


def test_two_missing_columns_both_named():
    """When multiple columns are missing, all names appear in the error."""
    df = _valid_df().drop(columns=["monetary", "basket_std"])
    with pytest.raises(ValueError) as exc_info:
        validate_features(df)
    msg = str(exc_info.value)
    assert "monetary" in msg
    assert "basket_std" in msg


def test_null_rate_above_threshold_raises():
    """High null rates indicate upstream pipeline problems."""
    df = _valid_df(n=100)
    null_idx = df.sample(frac=0.10, random_state=0).index
    df.loc[null_idx, "recency_days"] = np.nan  # 10% nulls, threshold=5%
    with pytest.raises(ValueError) as exc_info:
        validate_features(df)
    msg = str(exc_info.value)
    assert "null rate exceeds threshold" in msg
    assert "recency_days" in msg


def test_valid_dataframe_passes_silently():
    """Valid data must pass without transformation or exception."""
    df = _valid_df()
    original_shape = df.shape
    original_values = df.copy()
    result = validate_features(df)
    assert result["validation_passed"] is True
    assert df.shape == original_shape
    pd.testing.assert_frame_equal(df, original_values)


def test_empty_dataframe_raises():
    """An empty feature table means no customers to score — a pipeline failure."""
    df = _valid_df().iloc[0:0]
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_features(df)


def test_monetary_zero_raises():
    """Zero spend is impossible for a real customer and indicates data corruption."""
    df = _valid_df()
    df.loc[0, "monetary"] = 0.0
    with pytest.raises(ValueError, match="monetary must be > 0"):
        validate_features(df)


def test_monetary_negative_raises():
    """Negative spend indicates data corruption."""
    df = _valid_df()
    df.loc[0, "monetary"] = -50.0
    with pytest.raises(ValueError, match="monetary must be > 0"):
        validate_features(df)


def test_validation_report_returned():
    """Validator must return a structured report for MLflow logging."""
    df = _valid_df(n=50)
    report = validate_features(df)
    assert isinstance(report["rows_validated"], int)
    assert report["rows_validated"] == 50
    assert isinstance(report["columns_checked"], list)
    assert len(report["columns_checked"]) > 0
    assert isinstance(report["null_rates"], dict)
    assert report["validation_passed"] is True
    assert isinstance(report["timestamp"], datetime)
