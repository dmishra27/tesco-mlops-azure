# Databricks notebook source
"""
05_outcome_tracking.py — Realised lift measurement and retraining trigger.

Joins the gold/inference_log against actual bronze transactions to compute
decile-level realised lift, logs results to MLflow on the Production model
run, and triggers retraining when lift@D1 < 1.5 or the model is older than
30 days.

Notebook exit values:
    OK              — lift healthy, no retraining required
    RETRAIN_REQUIRED — lift degraded or time-based trigger fired
"""

from datetime import date, timedelta

import mlflow
from mlflow import MlflowClient
from pyspark.sql import Window, functions as F

# ── Config ────────────────────────────────────────────────────────────────────

STORAGE_ACCOUNT    = dbutils.secrets.get(scope="adls-scope", key="storage-account-name")
GOLD_INFER_PATH    = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/inference_log"
BRONZE_TXN_PATH    = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net/transactions"
EXPERIMENT_PATH    = "/Shared/tesco-mlops/outcome_tracking"
PROPENSITY_MODEL   = "tesco-propensity"

LIFT_THRESHOLD     = 1.5   # retrain if realised lift@D1 falls below this
RETRAIN_DAYS_MAX   = 30    # retrain if Production model is older than this
DECILE_STATUS_GOOD = 2.0   # lift >= this → GOOD
DECILE_STATUS_POOR = 1.0   # lift <  this → POOR

today        = date.today()
window_end   = today - timedelta(days=7)    # predictions made at least 7 days ago
window_start = today - timedelta(days=14)   # ...but no more than 14 days ago

# ── Step 1 — Load inference log ───────────────────────────────────────────────

inference_log = (
    spark.read.format("delta").load(GOLD_INFER_PATH)
    .select("customer_id", "propensity_score", "segment_id", "scored_at", "model_version")
    .where(
        (F.col("scored_at").cast("date") >= F.lit(str(window_start)))
        & (F.col("scored_at").cast("date") <= F.lit(str(window_end)))
    )
)

scored_count = inference_log.count()
print(f"Step 1 — Loaded {scored_count:,} scored customers "
      f"(scored {window_start} → {window_end})")

# ── Step 2 — Load actual purchases ────────────────────────────────────────────

# Outcome window: 7 days after each prediction.
# Since scored_at is 7–14 days ago, purchases span today-14 → today-0.
# We flag purchased=1 for any customer with a transaction in that window.
purchases = (
    spark.read.format("delta").load(BRONZE_TXN_PATH)
    .where(
        F.col("transaction_date").cast("date") >= F.lit(str(window_start))
    )
    .groupBy("customer_id")
    .agg(F.lit(1).alias("purchased"))
)

purchase_count = purchases.count()
print(f"Step 2 — {purchase_count:,} customers made at least one purchase "
      f"in the outcome window ({window_start} → {today})")

# ── Step 3 — Join and compute lift ────────────────────────────────────────────

scored_with_outcome = (
    inference_log
    .join(purchases, "customer_id", "left")
    .withColumn("purchased", F.coalesce(F.col("purchased"), F.lit(0)))
)

baseline_rate = scored_with_outcome.agg(
    F.mean("purchased").alias("baseline_rate")
).collect()[0]["baseline_rate"] or 0.0

# Assign deciles — ntile(10) on propensity_score descending → decile 1 = highest scores
decile_window = Window.orderBy(F.col("propensity_score").desc())

lift_table = (
    scored_with_outcome
    .withColumn("decile", F.ntile(10).over(decile_window))
    .groupBy("decile")
    .agg(
        F.count("*").alias("n_customers"),
        F.round(F.mean("propensity_score"), 4).alias("predicted_positive_rate"),
        F.round(F.mean("purchased"),        4).alias("actual_positive_rate"),
    )
    .withColumn("baseline_rate", F.lit(round(baseline_rate, 4)))
    .withColumn(
        "realised_lift",
        F.round(F.col("actual_positive_rate") / F.lit(baseline_rate), 4)
        if baseline_rate > 0 else F.lit(0.0),
    )
    .orderBy("decile")
)

lift_rows = lift_table.collect()

# ── Step 4 — Print lift table ─────────────────────────────────────────────────

header = (
    f"{'Decile':>6}  {'N':>7}  {'Pred_rate':>9}  "
    f"{'Actual_rate':>11}  {'Lift':>6}  Status"
)
separator = "-" * len(header)
print(f"\n{header}\n{separator}")

for row in lift_rows:
    lift = row["realised_lift"] or 0.0
    if lift >= DECILE_STATUS_GOOD:
        status = "GOOD"
    elif lift >= DECILE_STATUS_POOR:
        status = "WARN"
    else:
        status = "POOR"
    print(
        f"{row['decile']:>6}  {row['n_customers']:>7,}  "
        f"{row['predicted_positive_rate']:>9.4f}  "
        f"{row['actual_positive_rate']:>11.4f}  "
        f"{lift:>6.4f}  {status}"
    )
print(f"\nBaseline conversion rate: {baseline_rate:.4f}")

# ── Step 5 — Log to MLflow ────────────────────────────────────────────────────

# Compute days since the Production model was registered
client = MlflowClient()
prod_versions = client.get_latest_versions(PROPENSITY_MODEL, stages=["Production"])
if prod_versions:
    mv             = prod_versions[0]
    created_ms     = mv.creation_timestamp          # epoch milliseconds
    created_date   = date.fromtimestamp(created_ms / 1000)
    days_since_retrain = (today - created_date).days
    model_version_tag  = mv.version
else:
    days_since_retrain = 0
    model_version_tag  = "unknown"

print(f"\nProduction model v{model_version_tag} — "
      f"registered {days_since_retrain} days ago")

# Primary metric: lift at decile 1
d1_row = next((r for r in lift_rows if r["decile"] == 1), None)
realised_lift_d1 = float(d1_row["realised_lift"]) if d1_row else 0.0

mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="outcome_tracking"):
    mlflow.log_param("days_since_retrain", days_since_retrain)
    mlflow.log_param("model_version",      model_version_tag)
    mlflow.log_param("outcome_window",     f"{window_start} → {today}")

    mlflow.log_metric("realised_lift_decile1", realised_lift_d1)
    mlflow.log_metric("baseline_rate",         round(baseline_rate, 4))
    mlflow.log_metric("scored_customers",      scored_count)

    for row in lift_rows:
        d = row["decile"]
        mlflow.log_metric(f"lift_decile{d}",          row["realised_lift"] or 0.0)
        mlflow.log_metric(f"actual_rate_decile{d}",   row["actual_positive_rate"])
        mlflow.log_metric(f"n_customers_decile{d}",   row["n_customers"])

    mlflow.set_tag("scored_window",   f"{window_start} → {window_end}")
    mlflow.set_tag("purchase_window", f"{window_start} → {today}")

# ── Step 6 — Retraining trigger ───────────────────────────────────────────────

if realised_lift_d1 < LIFT_THRESHOLD:
    print(
        f"\nTRIGGER: lift@D1 {realised_lift_d1:.4f} < {LIFT_THRESHOLD} — "
        "lift degraded below acceptable threshold."
    )
    dbutils.notebook.exit("RETRAIN_REQUIRED")

elif days_since_retrain > RETRAIN_DAYS_MAX:
    print(
        f"\nTRIGGER: {days_since_retrain} days since last retrain "
        f"(> {RETRAIN_DAYS_MAX}-day maximum)."
    )
    dbutils.notebook.exit("RETRAIN_REQUIRED")

else:
    print(
        f"\nNo retraining required — lift@D1 {realised_lift_d1:.4f} >= {LIFT_THRESHOLD} "
        f"and model age {days_since_retrain} days <= {RETRAIN_DAYS_MAX}."
    )
    dbutils.notebook.exit("OK")
