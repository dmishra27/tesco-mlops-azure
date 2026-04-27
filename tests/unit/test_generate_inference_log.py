"""
Unit tests for ml.local.generate_inference_log.
"""
from __future__ import annotations

import datetime

import pytest

from ml.local.generate_inference_log import generate_inference_log


@pytest.fixture(scope="module")
def log_df():
    return generate_inference_log(n_customers=500, n_weeks=8, seed=42)


def test_output_has_required_columns(log_df):
    required = {"customer_id", "propensity_score", "segment_id", "scored_at", "model_version"}
    assert required.issubset(set(log_df.columns)), (
        f"Missing columns: {required - set(log_df.columns)}"
    )


def test_scored_at_in_last_8_weeks(log_df):
    now = datetime.datetime.now(datetime.UTC)
    cutoff = now - datetime.timedelta(weeks=8)
    earliest = log_df["scored_at"].min()
    latest   = log_df["scored_at"].max()
    assert earliest >= cutoff, f"scored_at too old: {earliest} < {cutoff}"
    assert latest   <= now,    f"scored_at in the future: {latest} > {now}"


def test_propensity_score_between_0_and_1(log_df):
    assert (log_df["propensity_score"] >= 0.0).all(), "Negative propensity scores found"
    assert (log_df["propensity_score"] <= 1.0).all(), "Propensity scores > 1.0 found"


def test_no_null_values(log_df):
    nulls = log_df.isnull().sum()
    assert nulls.sum() == 0, f"Null values found:\n{nulls[nulls > 0]}"
