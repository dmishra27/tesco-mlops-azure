"""
Feature table validator for the Tesco propensity pipeline.

Every rule below corresponds to a specific production failure mode:
  - Negative recency     : impossible value indicates upstream join bug
  - Zero frequency       : phantom customer from a bad join
  - online_ratio bounds  : division error in feature engineering
  - Duplicate customer   : inflates customer influence during training
  - Missing column       : silent NaN injection at inference time
  - High null rate       : upstream data quality problem
  - Zero/negative spend  : data corruption — real customers always spend > 0
  - Empty DataFrame      : complete upstream pipeline failure
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

# ── Business-configurable constants ──────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = [
    "customer_id",
    "recency_days",
    "frequency",
    "monetary",
    "avg_basket_size",
    "basket_std",
    "online_ratio",
    "active_days",
]

NULL_THRESHOLD: float = 0.05   # 5 % null rate triggers validation failure


# ── Private validation helpers ────────────────────────────────────────────────

def _check_empty(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise ValueError("DataFrame is empty — no customers to validate or score")


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        cols = ", ".join(missing)
        raise ValueError(f"missing required column: {cols}")


def _check_duplicates(df: pd.DataFrame) -> None:
    dup_count = df["customer_id"].duplicated().sum()
    if dup_count > 0:
        raise ValueError(
            f"duplicate customer_id found: {dup_count} duplicate(s) — "
            "each customer must appear exactly once"
        )


def _check_null_rates(
    df: pd.DataFrame, null_threshold: float
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        rate = float(df[col].isna().mean())
        rates[col] = rate
        if rate > null_threshold:
            raise ValueError(
                f"null rate exceeds threshold for '{col}': "
                f"{rate:.1%} (threshold={null_threshold:.1%})"
            )
    return rates


def _check_value_ranges(df: pd.DataFrame) -> None:
    # recency_days >= 0
    bad = df[df["recency_days"] < 0]
    if len(bad) > 0:
        actual = bad["recency_days"].iloc[0]
        raise ValueError(
            f"recency_days must be >= 0 — found value: {actual}"
        )

    # frequency >= 1
    if (df["frequency"] < 1).any():
        raise ValueError("frequency must be >= 1 — zero frequency indicates a phantom customer")

    # monetary > 0
    if (df["monetary"] <= 0).any():
        raise ValueError(
            "monetary must be > 0 — zero or negative spend indicates data corruption"
        )

    # online_ratio in [0, 1]
    if (df["online_ratio"] < 0).any() or (df["online_ratio"] > 1).any():
        raise ValueError(
            "online_ratio must be between 0 and 1 — "
            "values outside this range indicate a division error in feature engineering"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def validate_features(
    df: pd.DataFrame,
    null_threshold: float = NULL_THRESHOLD,
) -> dict[str, Any]:
    """
    Validate a customer feature DataFrame against all business rules.

    Parameters
    ----------
    df : pd.DataFrame
        Customer-level features, one row per customer.
    null_threshold : float
        Maximum allowed null rate per column (default 5%).

    Returns
    -------
    dict
        Validation report containing rows_validated, columns_checked,
        null_rates, validation_passed, and timestamp.

    Raises
    ------
    ValueError
        On the first rule violation found, with a message describing
        the rule, the actual value, and the threshold.
    """
    _check_empty(df)
    _check_required_columns(df)
    _check_duplicates(df)
    null_rates = _check_null_rates(df, null_threshold)
    _check_value_ranges(df)

    return {
        "rows_validated":  len(df),
        "columns_checked": REQUIRED_COLUMNS,
        "null_rates":      null_rates,
        "validation_passed": True,
        "timestamp":       datetime.now(),
    }
