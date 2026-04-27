"""
Direct unit tests for ml.local.feature_engineering.build_features and
assign_persona_labels.  These tests import from the module itself; they are
distinct from test_feature_engineering.py which reimplements its own
Databricks-notebook-compatible version of the same logic.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

from ml.local.feature_engineering import assign_persona_labels, build_features, main

# Snapshot date: all test transactions fall before this date → recency >= 0.
SNAPSHOT = date(2024, 1, 31)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_txns() -> pd.DataFrame:
    """
    Six transactions across three customers:
      C1 – 3 txns, 2 online,   1 in-store, touches a promoted category (bakery)
      C2 – 2 txns, 0 online,   2 in-store, no promoted category
      C3 – 1 txn,  1 online,   0 in-store, touches a promoted category (ready_meals)
    """
    rows = [
        dict(customer_id="C1", date="2024-01-05", category="bakery",
             channel="online",   basket_value=10.0),
        dict(customer_id="C1", date="2024-01-10", category="dairy",
             channel="online",   basket_value=15.0),
        dict(customer_id="C1", date="2024-01-20", category="produce",
             channel="in-store", basket_value=20.0),
        dict(customer_id="C2", date="2024-01-03", category="snacks",
             channel="in-store", basket_value=8.0),
        dict(customer_id="C2", date="2024-01-15", category="frozen",
             channel="in-store", basket_value=12.0),
        dict(customer_id="C3", date="2024-01-25", category="ready_meals",
             channel="online",   basket_value=30.0),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def features(sample_txns) -> pd.DataFrame:
    return build_features(sample_txns, SNAPSHOT)


# ── build_features: five required tests ──────────────────────────────────────

def test_recency_days_non_negative(features):
    """recency_days must always be >= 0 (snapshot date is after all transactions)."""
    assert (features["recency_days"] >= 0).all(), (
        f"Negative recency found:\n{features[features['recency_days'] < 0]}"
    )


def test_frequency_equals_transaction_count(sample_txns, features):
    """frequency must equal the actual number of transaction rows per customer."""
    expected = sample_txns.groupby("customer_id").size()
    merged = features.set_index("customer_id")["frequency"]
    for cid, exp_freq in expected.items():
        assert merged[cid] == exp_freq, (
            f"frequency mismatch for {cid}: got {merged[cid]}, expected {exp_freq}"
        )


def test_monetary_equals_basket_sum(sample_txns, features):
    """monetary must equal the sum of basket_value for each customer."""
    expected = sample_txns.groupby("customer_id")["basket_value"].sum()
    merged = features.set_index("customer_id")["monetary"]
    for cid, exp_total in expected.items():
        assert abs(merged[cid] - exp_total) < 1e-9, (
            f"monetary mismatch for {cid}: got {merged[cid]:.4f}, expected {exp_total:.4f}"
        )


def test_online_ratio_between_0_and_1(features):
    """online_ratio must be in [0, 1] for every customer."""
    assert (features["online_ratio"] >= 0.0).all()
    assert (features["online_ratio"] <= 1.0).all()


def test_no_duplicate_customer_ids(features):
    """Output must have exactly one row per customer."""
    assert features["customer_id"].is_unique, (
        f"Duplicate customer_ids: {features[features['customer_id'].duplicated()]['customer_id'].tolist()}"
    )


# ── build_features: promoted category flag ───────────────────────────────────

def test_has_promoted_category_flag(features):
    """
    C1 and C3 have at least one transaction in a promoted category;
    C2 does not.  The flag must reflect this exactly.
    """
    row = features.set_index("customer_id")["has_promoted_category"]
    assert row["C1"] == 1, "C1 bought bakery (promoted) — flag should be 1"
    assert row["C2"] == 0, "C2 only bought snacks+frozen (not promoted) — flag should be 0"
    assert row["C3"] == 1, "C3 bought ready_meals (promoted) — flag should be 1"


# ── build_features: online_ratio values are correct ──────────────────────────

def test_online_ratio_values(sample_txns, features):
    """Verify the calculated online_ratio against manual computation."""
    row = features.set_index("customer_id")["online_ratio"]
    # C1: 2 online, 1 in-store → 2/3
    assert abs(row["C1"] - 2 / 3) < 1e-9
    # C2: 0 online, 2 in-store → 0.0
    assert row["C2"] == 0.0
    # C3: 1 online, 0 in-store → 1.0
    assert row["C3"] == 1.0


# ── assign_persona_labels ─────────────────────────────────────────────────────

def test_assign_persona_labels_a_rate():
    """
    Persona A should produce labels with approximately 70% positive rate.
    We use 2000 samples and a generous tolerance (±15 pp) to avoid flakiness.
    """
    n = 2000
    ids = pd.Series([f"CUST-{i:04d}" for i in range(n)], name="customer_id")
    persona_map = pd.DataFrame({"customer_id": ids, "persona": ["A"] * n})
    labels = assign_persona_labels(ids, persona_map, pd.DataFrame(), seed=0)
    rate = labels.mean()
    assert 0.55 <= rate <= 0.85, (
        f"Persona A label rate {rate:.2%} outside expected [55%, 85%]"
    )


def test_assign_persona_labels_c_rate():
    """
    Persona C should produce labels with approximately 8% positive rate.
    """
    n = 2000
    ids = pd.Series([f"CUST-{i:04d}" for i in range(n)], name="customer_id")
    persona_map = pd.DataFrame({"customer_id": ids, "persona": ["C"] * n})
    labels = assign_persona_labels(ids, persona_map, pd.DataFrame(), seed=0)
    rate = labels.mean()
    assert 0.03 <= rate <= 0.15, (
        f"Persona C label rate {rate:.2%} outside expected [3%, 15%]"
    )


def test_assign_persona_labels_unknown_persona_defaults_to_c_rate():
    """
    An unknown persona should default to the Persona C rate (8%).
    """
    n = 2000
    ids = pd.Series([f"CUST-{i:04d}" for i in range(n)], name="customer_id")
    persona_map = pd.DataFrame({"customer_id": ids, "persona": ["Z"] * n})
    labels = assign_persona_labels(ids, persona_map, pd.DataFrame(), seed=0)
    rate = labels.mean()
    assert rate < 0.20, (
        f"Unknown persona produced rate {rate:.2%}; expected near 8%"
    )


# ── CLI refactor tests ────────────────────────────────────────────────────────

def test_main_accepts_cli_arguments(tmp_path):
    """
    main() must accept --input-path and --output-path so it can be called
    programmatically without touching sys.argv.
    """
    from ml.local.generate import generate

    # Generate synthetic data into a temp input directory
    input_dir = str(tmp_path / "synthetic")
    generate(out_dir=input_dir)

    output_dir = str(tmp_path / "output")
    main([
        "--input-path", input_dir,
        "--output-path", output_dir,
    ])

    assert os.path.exists(os.path.join(output_dir, "features", "customer_features.csv")), \
        "customer_features.csv not produced"
    assert os.path.exists(os.path.join(output_dir, "splits", "train.csv")), \
        "train.csv not produced"
    assert os.path.exists(os.path.join(output_dir, "splits", "val.csv")), \
        "val.csv not produced"
    assert os.path.exists(os.path.join(output_dir, "splits", "test.csv")), \
        "test.csv not produced"


def test_main_fails_gracefully_on_missing_path(tmp_path):
    """
    main() must raise FileNotFoundError (or OSError) when the input path
    does not exist — not crash with an unhandled AttributeError or KeyError.
    """
    with pytest.raises((FileNotFoundError, OSError)):
        main([
            "--input-path", str(tmp_path / "nonexistent_synthetic"),
            "--output-path", str(tmp_path / "output"),
        ])
