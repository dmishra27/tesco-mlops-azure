"""
DAG: tesco_ml_daily_pipeline
Orchestrates the full Tesco customer segmentation and propensity
training pipeline via Databricks notebooks, running daily at 02:00 UTC.

Task graph:
    ingest_streaming
         │
    feature_engineering
        ┌┴──────────────┐
train_segmentation  train_propensity   ← parallel
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator

DATABRICKS_CONN_ID = "databricks_default"
JOB_NAME = "tesco-mlops-training-pipeline"

# Job task keys must match those defined in databricks/jobs/run_job.json
TASK_INGEST       = "ingest"
TASK_FEATURES     = "feature_engineering"
TASK_SEGMENTATION = "train_segmentation"
TASK_PROPENSITY   = "train_propensity"

default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email": ["ml-platform-alerts@tesco.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}

with DAG(
    dag_id="tesco_ml_daily_pipeline",
    description="Daily Tesco customer segmentation and propensity model training",
    default_args=default_args,
    schedule_interval="0 2 * * *",   # 02:00 UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["tesco", "mlops", "databricks", "training"],
) as dag:

    # ── Task 1: ingest transactions from Event Hub into bronze ────────────────
    ingest_streaming = DatabricksRunNowOperator(
        task_id="ingest_streaming",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_name=JOB_NAME,
        notebook_params={},
        tasks_to_run=[TASK_INGEST],
    )

    # ── Task 2: build RFM + behavioural features into silver ──────────────────
    feature_engineering = DatabricksRunNowOperator(
        task_id="feature_engineering",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_name=JOB_NAME,
        notebook_params={},
        tasks_to_run=[TASK_FEATURES],
    )

    # ── Task 3: train KMeans segmentation (parallel with propensity) ──────────
    train_segmentation = DatabricksRunNowOperator(
        task_id="train_segmentation",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_name=JOB_NAME,
        notebook_params={"n_clusters": "5"},
        tasks_to_run=[TASK_SEGMENTATION],
    )

    # ── Task 4: train LightGBM propensity (parallel with segmentation) ────────
    train_propensity = DatabricksRunNowOperator(
        task_id="train_propensity",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_name=JOB_NAME,
        notebook_params={"target_category": "ready_meals"},
        tasks_to_run=[TASK_PROPENSITY],
    )

    # ── Dependency chain ──────────────────────────────────────────────────────
    # ingest → features → [segmentation ∥ propensity]
    ingest_streaming >> feature_engineering >> [train_segmentation, train_propensity]
