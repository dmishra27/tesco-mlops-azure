# Databricks notebook: drift_detector
# Compares current gold-layer feature distributions against the reference
# distributions saved as MLflow artifacts during model training.
#
# Population Stability Index (PSI) thresholds:
#   PSI < 0.1  → stable       (no action)
#   PSI < 0.2  → yellow alert (WARNING logged, PSI metrics emitted)
#   PSI >= 0.2 → red alert    (DRIFT_DETECTED exit — triggers retraining)
#
# PSI formula per bucket i:
#   PSI = Σ (actual_i% − expected_i%) × ln(actual_i% / expected_i%)

import json
import tempfile
from datetime import datetime, timedelta, timezone

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# ── Config ────────────────────────────────────────────────────────────────────
STORAGE_ACCOUNT   = dbutils.secrets.get(scope="adls-scope", key="STORAGE_ACCOUNT")
SILVER_PATH       = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net/customer_features"
MODEL_NAME        = "tesco-customer-segmentation"
MODEL_STAGE       = "Production"
EXPERIMENT_NAME   = "/Shared/tesco-mlops/drift-monitoring"
LOOKBACK_DAYS     = 7
N_BUCKETS         = 10          # number of equal-width bins for PSI
PSI_WARN          = 0.1         # yellow threshold
PSI_DRIFT         = 0.2         # red threshold — triggers retraining

MONITORED_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_basket_size",
    "online_ratio",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _safe_pct(arr: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Normalise to percentages and clip to avoid log(0)."""
    pct = arr / arr.sum()
    return np.clip(pct, eps, None)


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_buckets: int = N_BUCKETS) -> float:
    """
    Compute PSI between a reference (expected) and current (actual) sample.
    Bins are derived from the expected distribution so they are stable across runs.
    """
    breakpoints = np.linspace(expected.min(), expected.max(), n_buckets + 1)
    breakpoints[0]  -= 1e-6   # include the minimum value
    breakpoints[-1] += 1e-6   # include the maximum value

    expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0].astype(float)

    expected_pct = _safe_pct(expected_counts)
    actual_pct   = _safe_pct(actual_counts)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def load_reference_distributions(client: mlflow.tracking.MlflowClient) -> dict[str, np.ndarray]:
    """
    Fetches reference feature distributions saved as 'reference_distributions.json'
    during model training (03_train_segmentation.py logs this artifact).
    Falls back to loading from the latest Production run if no artifact is found.
    """
    mv = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not mv:
        raise RuntimeError(f"No {MODEL_STAGE} version found for model '{MODEL_NAME}'")

    run_id = mv[0].run_id

    with tempfile.TemporaryDirectory() as tmp:
        try:
            local_path = client.download_artifacts(
                run_id, "reference_distributions.json", tmp
            )
            with open(local_path) as fh:
                raw = json.load(fh)
            return {feat: np.array(vals) for feat, vals in raw.items()}
        except Exception:
            # Artifact not present — fall back: reconstruct from logged params/metrics
            # In production, training notebooks should always save this artifact.
            raise RuntimeError(
                f"'reference_distributions.json' not found for run_id={run_id}. "
                "Ensure 03_train_segmentation.py logs reference distributions as an artifact."
            )


def load_current_distributions(lookback_days: int) -> dict[str, np.ndarray]:
    """Reads the last `lookback_days` of silver features and returns raw arrays."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()

    sdf = (
        spark.read.format("delta").load(SILVER_PATH)
        .filter(F.col("snapshot_date") >= F.lit(str(cutoff)))
        .select(*MONITORED_FEATURES)
        .dropna()
    )

    if sdf.rdd.isEmpty():
        raise RuntimeError(
            f"No silver features found in the last {lookback_days} days "
            f"(cutoff={cutoff}). Check the feature engineering pipeline."
        )

    pdf = sdf.toPandas()
    return {feat: pdf[feat].values for feat in MONITORED_FEATURES}


def classify_psi(psi: float) -> str:
    if psi >= PSI_DRIFT:
        return "RED"
    if psi >= PSI_WARN:
        return "YELLOW"
    return "GREEN"


# ── Main ──────────────────────────────────────────────────────────────────────

mlflow.set_experiment(EXPERIMENT_NAME)
client = mlflow.tracking.MlflowClient()

print(f"Loading reference distributions for model '{MODEL_NAME}' ({MODEL_STAGE}) ...")
reference = load_reference_distributions(client)

print(f"Loading current feature distributions (last {LOOKBACK_DAYS} days) ...")
current = load_current_distributions(LOOKBACK_DAYS)

# ── Compute PSI for each monitored feature ────────────────────────────────────
psi_results: dict[str, float] = {}
statuses:    dict[str, str]   = {}

for feature in MONITORED_FEATURES:
    if feature not in reference:
        print(f"  [SKIP] '{feature}' not in reference distributions — skipping.")
        continue

    psi_val   = compute_psi(reference[feature], current[feature])
    status    = classify_psi(psi_val)
    psi_results[feature] = psi_val
    statuses[feature]    = status

    symbol = {"GREEN": "✓", "YELLOW": "⚠", "RED": "✗"}[status]
    print(f"  {symbol} PSI({feature:>20s}) = {psi_val:.4f}  [{status}]")

# ── Log PSI scores to MLflow ──────────────────────────────────────────────────
mv      = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
run_id  = mv.run_id

with mlflow.start_run(
    run_id=run_id,
    experiment_id=mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id,
    nested=True,
    run_name=f"drift-check-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
) as drift_run:
    for feature, psi_val in psi_results.items():
        mlflow.log_metric(f"psi_{feature}", psi_val)
    mlflow.log_metric("psi_max", max(psi_results.values()))
    mlflow.log_dict(
        {f: {"psi": v, "status": statuses[f]} for f, v in psi_results.items()},
        "drift_report.json",
    )
    drift_run_id = drift_run.info.run_id

print(f"\nDrift metrics logged → MLflow run_id: {drift_run_id}")

# ── Threshold alerting ────────────────────────────────────────────────────────
yellow_features = [f for f, s in statuses.items() if s == "YELLOW"]
red_features    = [f for f, s in statuses.items() if s == "RED"]

if yellow_features:
    print(
        f"\n⚠  WARNING: PSI > {PSI_WARN} (yellow) detected for: "
        + ", ".join(f"{f}={psi_results[f]:.4f}" for f in yellow_features)
    )

if red_features:
    summary = ", ".join(f"{f}={psi_results[f]:.4f}" for f in red_features)
    print(f"\n✗  DRIFT_DETECTED: PSI > {PSI_DRIFT} (red) for: {summary}")
    print("   Exiting with DRIFT_DETECTED to trigger retraining pipeline.")
    dbutils.notebook.exit("DRIFT_DETECTED")

if not yellow_features and not red_features:
    print(f"\n✓  All features stable (max PSI = {max(psi_results.values()):.4f})")

dbutils.notebook.exit("OK")
