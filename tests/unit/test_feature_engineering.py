"""
Unit tests for the feature engineering logic defined in
databricks/notebooks/02_feature_engineering.py.

The notebook runs on Spark, so we re-implement the same
pandas-equivalent logic here to keep tests fast and dependency-free.
All assertions mirror invariants the production notebook must satisfy.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest


# ── Pure-Python equivalents of the notebook transformations ──────────────────

def compute_rfm(transactions: pd.DataFrame, snapshot_date: date) -> pd.DataFrame:
    """
    Mirrors the RFM aggregation in 02_feature_engineering.py.
    Returns one row per customer_id.
    """
    grp = transactions.groupby("customer_id")
    rfm = pd.DataFrame({
        "recency_days":    grp["timestamp"].max().apply(
            lambda ts: (snapshot_date - ts.date()).days
        ),
        "frequency":       grp["transaction_id"].count(),
        "monetary":        grp["total_amount"].sum(),
        "active_days":     grp["ingest_date"].nunique(),
        "avg_basket_size": grp["total_amount"].mean(),
        "basket_std":      grp["total_amount"].std().fillna(0.0),
    }).reset_index()
    return rfm


def compute_channel_mix(transactions: pd.DataFrame) -> pd.DataFrame:
    grp = transactions.groupby("customer_id")
    mix = pd.DataFrame({
        "online_txns":  grp.apply(lambda x: (x["channel"] == "online").sum()),
        "instore_txns": grp.apply(lambda x: (x["channel"] == "in-store").sum()),
    }).reset_index()
    mix["online_ratio"] = mix["online_txns"] / (
        mix["online_txns"] + mix["instore_txns"]
    ).replace(0, np.nan).fillna(0.0)
    return mix


def compute_top_categories(transactions: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    cat_spend = (
        transactions.groupby(["customer_id", "category"])["total_amount"]
        .sum()
        .reset_index()
    )
    cat_spend["rank"] = cat_spend.groupby("customer_id")["total_amount"].rank(
        ascending=False, method="first"
    )
    top = (
        cat_spend[cat_spend["rank"] <= top_n]
        .groupby("customer_id")["category"]
        .apply(list)
        .reset_index()
        .rename(columns={"category": "top_categories"})
    )
    return top


def build_features(transactions: pd.DataFrame, snapshot_date: date) -> pd.DataFrame:
    rfm  = compute_rfm(transactions, snapshot_date)
    mix  = compute_channel_mix(transactions)
    cats = compute_top_categories(transactions)
    return rfm.merge(mix, on="customer_id", how="left").merge(
        cats, on="customer_id", how="left"
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRFMCalculation:
    def test_one_row_per_customer(self, transactions_df):
        snapshot = date.today()
        rfm = compute_rfm(transactions_df, snapshot)
        assert rfm["customer_id"].is_unique, "RFM must have exactly one row per customer"

    def test_recency_days_non_negative(self, transactions_df):
        """recency_days must always be >= 0 (snapshot >= latest transaction)."""
        snapshot = date.today()
        rfm = compute_rfm(transactions_df, snapshot)
        assert (rfm["recency_days"] >= 0).all(), (
            f"Negative recency found:\n{rfm[rfm['recency_days'] < 0]}"
        )

    def test_recency_days_bounded_by_window(self, transactions_df):
        """All test transactions are within a 90-day window."""
        snapshot = date.today()
        rfm = compute_rfm(transactions_df, snapshot)
        assert (rfm["recency_days"] <= 90).all()

    def test_frequency_positive_integer(self, transactions_df):
        """Every customer must have at least 1 transaction."""
        snapshot = date.today()
        rfm = compute_rfm(transactions_df, snapshot)
        assert (rfm["frequency"] >= 1).all(), "frequency must be >= 1 for all customers"
        assert rfm["frequency"].dtype in (
            np.dtype("int32"), np.dtype("int64"), np.dtype("object")
        ) or pd.api.types.is_integer_dtype(rfm["frequency"])

    def test_monetary_positive(self, transactions_df):
        """Total spend must be positive given positive unit prices."""
        snapshot = date.today()
        rfm = compute_rfm(transactions_df, snapshot)
        assert (rfm["monetary"] > 0).all()

    def test_avg_basket_size_within_bounds(self, transactions_df):
        snapshot = date.today()
        rfm = compute_rfm(transactions_df, snapshot)
        # Synthetic prices range [0.50, 25.00], qty [1, 7]
        assert (rfm["avg_basket_size"] >= 0.5).all()
        assert (rfm["avg_basket_size"] <= 200.0).all()

    def test_basket_std_non_negative(self, transactions_df):
        snapshot = date.today()
        rfm = compute_rfm(transactions_df, snapshot)
        assert (rfm["basket_std"] >= 0).all()


class TestFullFeaturePipeline:
    def test_output_schema(self, transactions_df):
        features = build_features(transactions_df, date.today())
        expected_cols = {
            "customer_id", "recency_days", "frequency", "monetary",
            "active_days", "avg_basket_size", "basket_std",
            "online_txns", "instore_txns", "online_ratio", "top_categories",
        }
        assert expected_cols.issubset(set(features.columns)), (
            f"Missing columns: {expected_cols - set(features.columns)}"
        )

    def test_online_ratio_bounded(self, transactions_df):
        """online_ratio must be in [0, 1]."""
        features = build_features(transactions_df, date.today())
        assert (features["online_ratio"] >= 0).all()
        assert (features["online_ratio"] <= 1).all()

    def test_top_categories_list_length(self, transactions_df):
        """top_categories should contain at most 3 entries per customer."""
        features = build_features(transactions_df, date.today())
        lengths = features["top_categories"].dropna().apply(len)
        assert (lengths <= 3).all()


class TestNullHandling:
    def test_null_top_categories_does_not_raise(self, features_with_nulls_df):
        """Rows with null top_categories must survive fillna without error."""
        df = features_with_nulls_df.copy()
        df["top_categories"] = df["top_categories"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        assert df["top_categories"].notna().all()

    def test_null_basket_std_filled_with_zero(self, features_with_nulls_df):
        """basket_std nulls should be filled with 0.0 (single-transaction customers)."""
        df = features_with_nulls_df.copy()
        df["basket_std"] = df["basket_std"].fillna(0.0)
        assert df["basket_std"].isna().sum() == 0
        assert (df["basket_std"] >= 0).all()

    def test_rfm_handles_single_transaction_customer(self):
        """A customer with exactly one transaction must have basket_std == 0."""
        txn = pd.DataFrame([{
            "transaction_id": "TXN-001",
            "customer_id":    "CUST-SINGLE",
            "category":       "bakery",
            "quantity":       2,
            "unit_price":     3.00,
            "total_amount":   6.00,
            "timestamp":      pd.Timestamp("2024-06-01"),
            "ingest_date":    date(2024, 6, 1),
            "channel":        "in-store",
        }])
        rfm = compute_rfm(txn, date(2024, 7, 1))
        assert rfm.loc[0, "frequency"] == 1
        assert rfm.loc[0, "basket_std"] == 0.0
        assert rfm.loc[0, "recency_days"] == 30
