"""
Direct unit tests for ml.local.generate.

generate() is called once per module via a session-scoped fixture so the
5000-customer data set is built only once and shared across all tests.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ml.local.generate import (
    END_DATE,
    N_CUSTOMERS,
    PERSONAS,
    START_DATE,
    _exponential_recency,
    _power_law_basket,
    generate,
)


# ── Module-scoped fixture: generate data once ─────────────────────────────────

@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """
    Runs generate() once for the whole module into a temp directory.
    Returns (customers_df, txns_df) for all tests to share.
    """
    out = tmp_path_factory.mktemp("generate_data")
    generate(out_dir=str(out))
    customers = pd.read_csv(out / "customers.csv")
    txns      = pd.read_csv(out / "transactions.csv", parse_dates=["date"])
    return customers, txns


# ── Five required tests ───────────────────────────────────────────────────────

def test_customer_count(generated):
    """Output must have exactly N_CUSTOMERS rows — one row per synthetic customer."""
    customers, _ = generated
    assert len(customers) == N_CUSTOMERS, (
        f"Expected {N_CUSTOMERS} customers, got {len(customers)}"
    )


def test_persona_a_frequency_always_gt_persona_c(generated):
    """
    Persona A target_freq range (15-30) is entirely above Persona C (1-5),
    so the minimum of all A values must strictly exceed the maximum of all C values.
    """
    customers, _ = generated
    a_min = customers[customers["persona"] == "A"]["target_freq"].min()
    c_max = customers[customers["persona"] == "C"]["target_freq"].max()
    assert a_min > c_max, (
        f"Persona A min target_freq ({a_min}) must be > Persona C max ({c_max})"
    )


def test_persona_a_monetary_always_gt_persona_c(generated):
    """
    Persona A spend range (300-800) is entirely above Persona C (20-100),
    so the minimum of all A values must strictly exceed the maximum of all C values.
    """
    customers, _ = generated
    a_min = customers[customers["persona"] == "A"]["target_spend"].min()
    c_max = customers[customers["persona"] == "C"]["target_spend"].max()
    assert a_min > c_max, (
        f"Persona A min target_spend ({a_min:.2f}) must be > Persona C max ({c_max:.2f})"
    )


def test_no_null_values_in_output(generated):
    """Neither customers.csv nor transactions.csv may contain any null values."""
    customers, txns = generated
    cust_nulls = customers.isnull().sum().sum()
    txn_nulls  = txns.isnull().sum().sum()
    assert cust_nulls == 0, f"customers.csv contains {cust_nulls} null values"
    assert txn_nulls  == 0, f"transactions.csv contains {txn_nulls} null values"


def test_transaction_dates_within_specified_range(generated):
    """All transaction dates must fall within [START_DATE, END_DATE]."""
    _, txns = generated
    actual_min = txns["date"].dt.date.min()
    actual_max = txns["date"].dt.date.max()
    assert actual_min >= START_DATE, (
        f"Earliest transaction {actual_min} is before START_DATE {START_DATE}"
    )
    assert actual_max <= END_DATE, (
        f"Latest transaction {actual_max} is after END_DATE {END_DATE}"
    )


# ── Additional tests for helper functions and persona structure ───────────────

def test_persona_counts_match_config(generated):
    """Each persona must have exactly the number of customers declared in PERSONAS."""
    customers, _ = generated
    for persona, cfg in PERSONAS.items():
        actual = int((customers["persona"] == persona).sum())
        assert actual == cfg["n"], (
            f"Persona {persona}: expected {cfg['n']} customers, got {actual}"
        )


def test_all_personas_present(generated):
    """All three personas A, B, and C must appear in the output."""
    customers, _ = generated
    assert set(customers["persona"].unique()) == {"A", "B", "C"}


def test_power_law_basket_values_within_bounds():
    """_power_law_basket must always return values in [low, high]."""
    rng = np.random.default_rng(42)
    low, high = 5.0, 50.0
    result = _power_law_basket(rng, 2000, low, high)
    assert (result >= low).all(),  f"Some baskets below low bound {low}"
    assert (result <= high).all(), f"Some baskets above high bound {high}"


def test_exponential_recency_clipped_to_window():
    """_exponential_recency must return integer days-ago clipped to [0, 179]."""
    rng = np.random.default_rng(42)
    result = _exponential_recency(rng, 5000)
    assert result.dtype == int, "days_ago must be integer"
    assert int(result.min()) >= 0,   "days_ago must be >= 0"
    assert int(result.max()) <= 179, "days_ago must be <= 179 (180-day window)"
