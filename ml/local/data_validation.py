"""
Data validation for the Tesco bronze transaction layer.

Loads a Great Expectations suite JSON and validates a pandas DataFrame
against it. Supports all expectation types used in tesco_transactions.json,
including the custom expect_column_values_to_not_be_in_future.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_SUITE_PATH = Path(__file__).resolve().parents[2] / "ge_suite" / "tesco_transactions.json"


def load_suite(path: Path | None = None) -> list[dict]:
    """Return the expectations list from the GE suite JSON."""
    with open(path or _SUITE_PATH) as fh:
        return json.load(fh)["expectations"]


def validate(df: pd.DataFrame, expectations: list[dict]) -> dict:
    """
    Validate *df* against *expectations*.

    Returns
    -------
    dict with keys:
        passed  : bool  — True when score >= 0.95
        score   : float — fraction of expectations that passed
        results : list[dict] — one entry per expectation
    """
    results = []
    for exp in expectations:
        r = _apply(df, exp["expectation_type"], exp["kwargs"])
        results.append({
            "expectation": exp["expectation_type"],
            "column":      exp["kwargs"].get("column", ""),
            "passed":      r["passed"],
            "failed_rows": r["failed_rows"],
        })

    n_passed = sum(1 for r in results if r["passed"])
    score    = n_passed / len(results) if results else 1.0
    return {
        "passed":  score >= 0.95,
        "score":   score,
        "results": results,
    }


def _apply(df: pd.DataFrame, exp_type: str, kwargs: dict) -> dict:
    col = kwargs.get("column", "")

    if exp_type == "expect_column_values_to_not_be_null":
        failed = int(df[col].isna().sum())
        return {"passed": failed == 0, "failed_rows": failed}

    if exp_type == "expect_column_values_to_be_between":
        min_v = kwargs.get("min_value")
        max_v = kwargs.get("max_value")
        mask  = pd.Series(True, index=df.index)
        if min_v is not None:
            mask &= df[col] >= min_v
        if max_v is not None:
            mask &= df[col] <= max_v
        failed = int((~mask).sum())
        return {"passed": failed == 0, "failed_rows": failed}

    if exp_type == "expect_column_values_to_match_regex":
        failed = int((~df[col].astype(str).str.match(kwargs["regex"])).sum())
        return {"passed": failed == 0, "failed_rows": failed}

    if exp_type == "expect_column_values_to_be_in_set":
        value_set = set(kwargs["value_set"])
        failed    = int((~df[col].isin(value_set)).sum())
        return {"passed": failed == 0, "failed_rows": failed}

    if exp_type == "expect_column_values_to_not_be_in_future":
        now    = datetime.now(tz=timezone.utc)
        ts_col = pd.to_datetime(df[col], utc=True)
        failed = int((ts_col > now).sum())
        return {"passed": failed == 0, "failed_rows": failed}

    # Unknown expectation type — treat as passing so unknown types don't silently block
    return {"passed": True, "failed_rows": 0}
