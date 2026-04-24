"""
Shared pytest fixtures for the Tesco MLOps test suite.
Provides synthetic customer transaction and feature DataFrames
that mirror the schema produced by the real Databricks notebooks.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Synthetic data factories ──────────────────────────────────────────────────

def _make_transactions(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    today = date.today()
    categories = ["ready_meals", "bakery", "produce", "dairy", "beverages"]
    channels   = ["in-store", "online"]

    rows = []
    for i in range(n):
        qty   = int(rng.integers(1, 8))
        price = round(float(rng.uniform(0.5, 25.0)), 2)
        rows.append({
            "transaction_id": f"TXN-{i:06d}",
            "customer_id":    f"CUST-{rng.integers(1, 20):04d}",
            "store_id":       f"STORE-{rng.integers(1, 10):04d}",
            "product_id":     f"PROD-{rng.integers(1, 50):05d}",
            "category":       categories[i % len(categories)],
            "quantity":       qty,
            "unit_price":     price,
            "total_amount":   round(qty * price, 2),
            "timestamp":      pd.Timestamp(today - timedelta(days=int(rng.integers(0, 90)))),
            "channel":        channels[i % len(channels)],
            "ingest_date":    today,
        })
    return pd.DataFrame(rows)


def _make_customer_features(n_customers: int = 20, seed: int = 42) -> pd.DataFrame:
    """Returns a DataFrame with the schema produced by 02_feature_engineering."""
    rng = np.random.default_rng(seed)
    today = date.today()
    categories = ["ready_meals", "bakery", "produce", "dairy", "beverages"]

    rows = []
    for i in range(n_customers):
        online  = int(rng.integers(0, 30))
        instore = int(rng.integers(0, 30))
        rows.append({
            "customer_id":     f"CUST-{i + 1:04d}",
            "recency_days":    int(rng.integers(0, 90)),
            "frequency":       int(rng.integers(1, 50)),
            "monetary":        round(float(rng.uniform(10.0, 2000.0)), 2),
            "active_days":     int(rng.integers(1, 30)),
            "avg_basket_size": round(float(rng.uniform(5.0, 80.0)), 2),
            "basket_std":      round(float(rng.uniform(0.0, 20.0)), 2),
            "online_txns":     online,
            "instore_txns":    instore,
            "online_ratio":    round(online / max(online + instore, 1), 4),
            "top_categories":  list(rng.choice(categories, size=3, replace=False)),
            "snapshot_date":   today,
        })
    return pd.DataFrame(rows)


def _make_features_with_nulls(base_df: pd.DataFrame) -> pd.DataFrame:
    """Injects nulls into category and basket_std columns."""
    df = base_df.copy()
    null_idx = df.sample(frac=0.2, random_state=0).index
    df.loc[null_idx, "top_categories"] = None
    df.loc[null_idx, "basket_std"]     = None
    return df


def _make_minimal_pipeline() -> Pipeline:
    """A tiny fitted KMeans pipeline for use as a mock model."""
    rng = np.random.default_rng(0)
    X = rng.random((50, 9))
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=3, random_state=0, n_init=5)),
    ])
    pipeline.fit(X)
    return pipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def transactions_df() -> pd.DataFrame:
    return _make_transactions()


@pytest.fixture(scope="session")
def customer_features_df() -> pd.DataFrame:
    return _make_customer_features()


@pytest.fixture(scope="session")
def features_with_nulls_df(customer_features_df) -> pd.DataFrame:
    return _make_features_with_nulls(customer_features_df)


@pytest.fixture(scope="session")
def fitted_pipeline() -> Pipeline:
    return _make_minimal_pipeline()


@pytest.fixture
def mock_mlflow_model(fitted_pipeline):
    """Patches mlflow.sklearn.load_model to return the fitted pipeline."""
    with patch("ml.score.mlflow.sklearn.load_model", return_value=fitted_pipeline):
        yield fitted_pipeline
