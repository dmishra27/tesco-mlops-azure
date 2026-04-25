"""
Unit tests for ml.local.splits.TemporalSplitter.
All tests are written BEFORE the implementation exists (TDD RED phase).

Business context: temporal splitting prevents data leakage in time-series
retail data. Using future data in training gives the model knowledge it
would not have in production and produces inflated metrics.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ml.local.splits import TemporalSplitter  # FAILS until splits.py exists


# ── Shared test data factory ──────────────────────────────────────────────────

BASE_DATE = datetime(2024, 1, 1)


def _make_customers(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Returns customer-level DataFrame with last_transaction_date spread over 180 days."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        day_offset = int(rng.integers(0, 180))
        rows.append({
            "customer_id":            f"CUST-{i:05d}",
            "last_transaction_date":  BASE_DATE + timedelta(days=day_offset),
            "recency_days":           179 - day_offset,
            "frequency":              int(rng.integers(1, 30)),
            "monetary":               round(float(rng.uniform(10.0, 800.0)), 2),
            "label":                  int(rng.random() < 0.25),
        })
    return pd.DataFrame(rows)


def _make_splitter() -> TemporalSplitter:
    return TemporalSplitter(
        train_end_day=119,
        val_end_day=149,
        snapshot_day=179,
        start_date=BASE_DATE,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_future_leakage():
    """
    Business requirement: training features must never contain information
    from after the training window end date. Violating this gives the model
    knowledge of the future and inflates all metrics artificially.
    """
    customers = _make_customers(n=1000)
    splitter  = _make_splitter()
    train_df, val_df, test_df = splitter.split(customers)

    train_end_date = BASE_DATE + timedelta(days=119)
    val_end_date   = BASE_DATE + timedelta(days=149)

    # Every train customer's last_transaction_date is <= train window end
    assert (pd.to_datetime(train_df["last_transaction_date"]) <= train_end_date).all(), \
        "Train split contains customers with future transaction dates — data leakage"

    # Every test customer's last_transaction_date is strictly after val window end
    assert (pd.to_datetime(test_df["last_transaction_date"]) > val_end_date).all(), \
        "Test split contains customers from within the validation window — leakage"


def test_all_splits_non_empty():
    """
    Business requirement: each split must contain enough customers for
    meaningful model training. An empty split silently produces no model.
    """
    customers = _make_customers(n=1000)
    splitter  = _make_splitter()
    train_df, val_df, test_df = splitter.split(customers)

    assert len(train_df) >= 100, f"Train too small: {len(train_df)}"
    assert len(val_df)   >= 100, f"Val too small: {len(val_df)}"
    assert len(test_df)  >= 100, f"Test too small: {len(test_df)}"


def test_split_sizes_sum_to_total():
    """
    Business requirement: no customer is lost or duplicated during splitting.
    Lost customers = smaller training set. Duplicates = data leakage.
    """
    customers = _make_customers(n=1000)
    splitter  = _make_splitter()
    train_df, val_df, test_df = splitter.split(customers)

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(customers), \
        f"Split sizes sum to {total}, expected {len(customers)}"

    all_ids = pd.concat([
        train_df["customer_id"],
        val_df["customer_id"],
        test_df["customer_id"],
    ])
    assert all_ids.nunique() == len(customers), \
        "customer_id appears in more than one split — data leakage"


def test_temporal_ordering_respected():
    """
    Business requirement: the test set must contain only the most recent data,
    reflecting production conditions where the model scores customers whose
    future behaviour is unknown.
    """
    customers = _make_customers(n=1000)
    splitter  = _make_splitter()
    train_df, val_df, test_df = splitter.split(customers)

    train_max = pd.to_datetime(train_df["last_transaction_date"]).max()
    val_min   = pd.to_datetime(val_df["last_transaction_date"]).min()
    val_max   = pd.to_datetime(val_df["last_transaction_date"]).max()
    test_min  = pd.to_datetime(test_df["last_transaction_date"]).min()

    assert train_max < val_min, \
        f"Temporal ordering violated: train_max={train_max}, val_min={val_min}"
    assert val_max < test_min, \
        f"Temporal ordering violated: val_max={val_max}, test_min={test_min}"


def test_class_balance_logged():
    """
    Business requirement: class imbalance must be visible before training.
    Severe imbalance (< 2% positive) means propensity signal is too weak.
    """
    customers = _make_customers(n=1000)
    splitter  = _make_splitter()
    train_df, val_df, test_df = splitter.split(customers)

    balance = splitter.class_balance(train_df, val_df, test_df, label_col="label")

    assert "train" in balance
    assert "val"   in balance
    assert "test"  in balance

    for key, rate in balance.items():
        assert rate is not None,              f"{key} balance is None"
        assert 0.0 <= rate <= 1.0,            f"{key} balance {rate} outside [0, 1]"


def test_single_customer_raises():
    """
    Edge case: a single customer cannot be split into three non-empty sets.
    """
    customers = _make_customers(n=1)
    splitter  = _make_splitter()

    with pytest.raises(ValueError, match="insufficient customers to split"):
        splitter.split(customers)


def test_window_overlap_raises():
    """
    Edge case: if val_end_day <= train_end_day the windows overlap and leak data.
    """
    with pytest.raises(ValueError, match="window overlap"):
        TemporalSplitter(
            train_end_day=149,   # val_start would be 150
            val_end_day=129,     # val_end < train_end  -> overlap
            snapshot_day=179,
            start_date=BASE_DATE,
        )
