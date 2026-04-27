"""
Synthetic drift data generators.

Produces DataFrames matching the silver layer schema (FEATURE_COLS) for
testing drift detection and PSI computation without real customer data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "recency_days", "frequency", "monetary", "avg_basket_size",
    "basket_std", "online_ratio", "active_days", "has_promoted_category",
]


def generate_stable_features(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a stable baseline feature set matching the silver layer schema.
    Distributions are fixed so two calls with different seeds produce
    statistically similar (low-PSI) DataFrames.
    """
    rng = np.random.default_rng(seed)
    n = n_customers
    return pd.DataFrame({
        "customer_id":          [f"CUST-{i:05d}" for i in range(1, n + 1)],
        "recency_days":         rng.integers(0, 180, n).astype(float),
        "frequency":            np.clip(rng.integers(1, 51, n), 1, None).astype(float),
        "monetary":             np.round(rng.lognormal(4.5, 1.0, n), 2),
        "avg_basket_size":      np.round(np.abs(rng.normal(25.0, 10.0, n)), 2),
        "basket_std":           np.round(np.abs(rng.normal(8.0, 4.0, n)), 2),
        "online_ratio":         np.clip(rng.beta(2, 3, n), 0.0, 1.0),
        "active_days":          np.clip(rng.integers(1, 61, n), 1, None).astype(float),
        "has_promoted_category": rng.integers(0, 2, n).astype(int),
    })


def generate_drifted_features(
    n_customers: int = 5000,
    drift_features: list[str] | None = None,
    drift_magnitude: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a drifted feature set derived from the stable baseline.

    The specified drift_features are shifted by 12 × std × drift_magnitude,
    which reliably produces PSI > 0.20 for magnitude ≥ 0.25.
    Non-drifted features retain their stable distributions (PSI < 0.10).
    """
    if drift_features is None:
        drift_features = ["recency_days", "online_ratio"]

    stable = generate_stable_features(n_customers=n_customers, seed=seed)
    drifted = stable.copy()
    rng = np.random.default_rng(seed + 999)

    for feat in drift_features:
        if feat not in drifted.columns:
            continue
        col = drifted[feat].values.astype(float)
        std = float(col.std()) if col.std() > 0 else 1.0

        # Shift by 3 × std for magnitude=0.25 → reliably exceeds PSI 0.20 threshold
        shift = std * 12.0 * drift_magnitude
        noise = rng.normal(0, std * drift_magnitude, n_customers)
        col = col + shift + noise

        # Clip to valid ranges
        if feat == "online_ratio":
            col = np.clip(col, 0.0, 1.0)
        elif feat == "has_promoted_category":
            col = np.clip(np.round(col).astype(int), 0, 1)
        else:
            col = np.maximum(col, 0.0)

        drifted[feat] = col

    return drifted


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between two 1-D arrays.

    PSI < 0.10 : stable
    PSI 0.10–0.20 : moderate shift
    PSI > 0.20 : significant drift
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)

    lo = min(float(expected.min()), float(actual.min()))
    hi = max(float(expected.max()), float(actual.max()))
    if hi == lo:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual,   bins=bins)

    eps = 1e-6
    exp_pct = np.maximum(exp_counts / max(len(expected), 1), eps)
    act_pct = np.maximum(act_counts / max(len(actual),   1), eps)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


if __name__ == "__main__":
    stable  = generate_stable_features(1000)
    drifted = generate_drifted_features(1000)
    print("PSI per feature:")
    for feat in FEATURE_COLS:
        if feat in stable.columns:
            psi = compute_psi(stable[feat].values, drifted[feat].values)
            flag = "DRIFT" if psi > 0.20 else ("WARN" if psi > 0.10 else "OK")
            print(f"  {feat:<30} {psi:.4f}  {flag}")
