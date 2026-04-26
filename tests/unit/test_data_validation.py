"""
Unit tests for ml.local.data_validation.
Each test exercises one expectation type from ge_suite/tesco_transactions.json.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ml.local.data_validation import load_suite, validate


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _row(**overrides) -> dict:
    """Return a dict representing a fully valid transaction row."""
    base = {
        "transaction_id": "TXN-001",
        "total_amount":   9.99,
        "customer_id":    "CUST-12345",
        "timestamp":      datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        "channel":        "online",
        "quantity":       2,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def expectations():
    return load_suite()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_valid_transactions_pass(expectations):
    """A DataFrame matching all schema rules must achieve score 1.0."""
    df     = pd.DataFrame([_row(), _row(transaction_id="TXN-002", customer_id="CUST-99")])
    result = validate(df, expectations)

    assert result["score"] == 1.0
    assert result["passed"] is True
    assert all(r["passed"] for r in result["results"])


def test_null_transaction_id_fails(expectations):
    """A row with transaction_id = None must fail the not-null expectation."""
    df     = pd.DataFrame([_row(transaction_id=None)])
    result = validate(df, expectations)

    failing = [
        r for r in result["results"]
        if r["expectation"] == "expect_column_values_to_not_be_null"
        and r["column"] == "transaction_id"
    ]
    assert failing, "Expected a not-null result entry for transaction_id"
    assert not failing[0]["passed"]
    assert failing[0]["failed_rows"] == 1


def test_negative_total_amount_fails(expectations):
    """A row with total_amount = -5.00 must fail the between expectation."""
    df     = pd.DataFrame([_row(total_amount=-5.00)])
    result = validate(df, expectations)

    failing = [
        r for r in result["results"]
        if r["expectation"] == "expect_column_values_to_be_between"
        and r["column"] == "total_amount"
    ]
    assert failing, "Expected a between result entry for total_amount"
    assert not failing[0]["passed"]
    assert failing[0]["failed_rows"] == 1


def test_future_timestamp_fails(expectations):
    """A row with a year-2099 timestamp must fail the not-in-future expectation."""
    future_ts = datetime(2099, 1, 1, tzinfo=timezone.utc)
    df        = pd.DataFrame([_row(timestamp=future_ts)])
    result    = validate(df, expectations)

    failing = [
        r for r in result["results"]
        if r["expectation"] == "expect_column_values_to_not_be_in_future"
    ]
    assert failing, "Expected a not-in-future result entry for timestamp"
    assert not failing[0]["passed"]
    assert failing[0]["failed_rows"] == 1
