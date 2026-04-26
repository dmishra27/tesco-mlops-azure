# Databricks notebook source
"""
00_data_validation.py — Bronze layer data quality gate.

Loads the tesco_transactions GE suite, validates the latest bronze
Delta partition, saves results as an MLflow artifact, and exits with a
non-zero code (blocking the downstream DAG) when validation score < 0.95.

Notebook exit values:
    VALIDATION_PASSED — score >= 0.95, pipeline continues
    VALIDATION_FAILED — score <  0.95, Airflow task fails, DAG halts
"""

import json
import sys
from datetime import date
from pathlib import Path

import mlflow
from pyspark.sql import functions as F

# ── Config ────────────────────────────────────────────────────────────────────

BRONZE_CONTAINER = "bronze"
SUITE_DBFS_PATH  = "/dbfs/mnt/gold/ge_suite/tesco_transactions.json"
SCORE_THRESHOLD  = 0.95
EXPERIMENT_PATH  = "/Shared/tesco-mlops/data_validation"

storage_account = dbutils.secrets.get(scope="adls-scope", key="storage-account-name")
bronze_path     = (
    f"abfss://{BRONZE_CONTAINER}@{storage_account}"
    ".dfs.core.windows.net/transactions"
)

# ── Load latest bronze partition ──────────────────────────────────────────────

today_str = date.today().isoformat()
bronze_df = (
    spark.read.format("delta")
         .load(bronze_path)
         .where(F.col("ingestion_date") == today_str)
)
row_count = bronze_df.count()
print(f"Loaded {row_count:,} rows from bronze partition {today_str}")

# ── Validate ──────────────────────────────────────────────────────────────────

# Repo is mounted at /Workspace/Repos so the ml package is importable
sys.path.insert(0, "/Workspace/Repos/tesco-mlops-azure")
from ml.local.data_validation import load_suite, validate  # noqa: E402

sample_pd    = bronze_df.toPandas()
expectations = load_suite(Path(SUITE_DBFS_PATH))
val_results  = validate(sample_pd, expectations)
score        = val_results["score"]

# ── Print summary table ───────────────────────────────────────────────────────

header = f"{'Expectation':<52}  {'Status':<8}  {'Failed rows':>11}"
separator = "-" * len(header)
print(f"\n{header}\n{separator}")
for r in val_results["results"]:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"{r['expectation']:<52}  {status:<8}  {r['failed_rows']:>11,}")

overall = "PASS" if val_results["passed"] else "FAIL"
print(f"\nOverall score: {score:.4f}  ({overall})")

# ── Log to MLflow ─────────────────────────────────────────────────────────────

mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="data_validation"):
    mlflow.log_metric("validation_score",      score)
    mlflow.log_metric("bronze_row_count",      row_count)
    mlflow.log_metric("expectations_total",    len(val_results["results"]))
    mlflow.log_metric("expectations_passed",
                      sum(1 for r in val_results["results"] if r["passed"]))
    mlflow.set_tag("partition_date",           today_str)
    mlflow.set_tag("validation_status",        overall)

    results_path = "/tmp/validation_results.json"
    with open(results_path, "w") as fh:
        json.dump(val_results, fh, indent=2, default=str)
    mlflow.log_artifact(results_path, artifact_path="data_validation")

# ── Gate: halt DAG on failure ─────────────────────────────────────────────────

if not val_results["passed"]:
    failed_expectations = [r for r in val_results["results"] if not r["passed"]]
    for f in failed_expectations:
        print(
            f"  FAILED: {f['expectation']} on column '{f['column']}' "
            f"— {f['failed_rows']:,} failing rows"
        )
    dbutils.notebook.exit("VALIDATION_FAILED")
    raise SystemExit(
        f"Data validation failed: score {score:.4f} < {SCORE_THRESHOLD}. "
        f"{len(failed_expectations)} expectation(s) failed. DAG halted."
    )

print(f"\nValidation PASSED — score {score:.4f} >= {SCORE_THRESHOLD}. Pipeline continues.")
dbutils.notebook.exit("VALIDATION_PASSED")
