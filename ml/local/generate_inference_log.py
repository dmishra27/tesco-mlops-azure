"""
Synthetic inference log generator.
Produces a scored_customers DataFrame matching the schema written by
05_outcome_tracking.py — used to test outcome tracking without real inference data.
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd


def generate_inference_log(
    n_customers: int = 5000,
    n_weeks: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic inference log covering the last n_weeks.

    Returns
    -------
    DataFrame with columns:
        customer_id      – CUST-XXXXX
        propensity_score – float [0, 1], higher for Persona A customers
        segment_id       – int in {0, 1, 2}
        scored_at        – datetime spread across last n_weeks
        model_version    – "1.0.0"
    """
    rng = np.random.default_rng(seed)
    now = datetime.datetime.utcnow()
    window_start = now - datetime.timedelta(weeks=n_weeks)

    # Persona split mirrors generate.py: A=10%, B=20%, C=70%
    n_a = max(1, n_customers // 10)
    n_b = max(1, n_customers * 2 // 10)
    n_c = n_customers - n_a - n_b

    # Propensity scores: Beta distributions tuned per persona
    scores_a = rng.beta(8, 2, n_a)          # mean ~0.80
    scores_b = rng.beta(4, 4, n_b)          # mean ~0.50
    scores_c = rng.beta(1.5, 6, n_c)        # mean ~0.20

    scores = np.concatenate([scores_a, scores_b, scores_c])
    scores = np.clip(scores, 0.0, 1.0)

    # Segment assignments (A→0 loyalists, B→1 digital-first, C→2 at-risk)
    segments = np.concatenate([
        np.zeros(n_a, dtype=int),
        np.ones(n_b, dtype=int),
        np.full(n_c, 2, dtype=int),
    ])

    # Shuffle so customer IDs aren't sorted by persona
    order = rng.permutation(n_customers)
    scores   = scores[order]
    segments = segments[order]

    # scored_at: random datetimes in the last n_weeks window
    seconds_window = int((now - window_start).total_seconds())
    offsets = rng.integers(0, seconds_window, n_customers)
    scored_at = [window_start + datetime.timedelta(seconds=int(s)) for s in offsets]

    return pd.DataFrame({
        "customer_id":      [f"CUST-{i:05d}" for i in range(1, n_customers + 1)],
        "propensity_score": np.round(scores, 4),
        "segment_id":       segments,
        "scored_at":        scored_at,
        "model_version":    "1.0.0",
    })


if __name__ == "__main__":
    df = generate_inference_log()
    print(f"Generated {len(df):,} inference records")
    print(df.head())
    print(f"\nPropensity score distribution:")
    print(df.groupby("segment_id")["propensity_score"].describe().round(3))
