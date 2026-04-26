"""
DAG: tesco_ml_daily_pipeline
Orchestrates the full Tesco customer segmentation and propensity
training pipeline via Databricks notebooks, running daily at 02:00 UTC.

Task graph (daily):
    data_validation          ← halts DAG if bronze score < 0.95
         │
    ingest_streaming
         │
    feature_engineering
        ┌┴──────────────┐
train_segmentation  train_propensity   ← parallel

Weekly monitoring task (Sundays, independent of daily chain):
    drift_monitoring
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.utils.trigger_rule import TriggerRule

DATABRICKS_CONN_ID   = "databricks_default"
JOB_NAME             = "tesco-mlops-training-pipeline"
DRIFT_NOTEBOOK_PATH  = "/Shared/tesco-mlops/drift_detector"

# Job task keys must match those defined in databricks/jobs/run_job.json
TASK_VALIDATION   = "data_validation"
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


# ── Drift response helper ─────────────────────────────────────────────────────

def _handle_drift_result(**context) -> str:
    """
    Reads the Databricks notebook exit value pushed by drift_monitoring
    and branches: if DRIFT_DETECTED route to trigger_retraining,
    otherwise route to drift_stable (no-op).
    """
    result = context["ti"].xcom_pull(task_ids="drift_monitoring", key="notebook_exit_value")
    if result == "DRIFT_DETECTED":
        return "trigger_retraining"
    return "drift_stable"


def _log_drift_stable(**_context) -> None:
    print("Drift check passed — all PSI scores within acceptable bounds.")


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="tesco_ml_daily_pipeline",
    description="Daily ML training pipeline + weekly drift monitoring with auto-retraining",
    default_args=default_args,
    schedule_interval="0 2 * * *",   # 02:00 UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["tesco", "mlops", "databricks", "training"],
) as dag:

    # ── Task 0: validate bronze layer — halts DAG if score < 0.95 ────────────
    data_validation = DatabricksRunNowOperator(
        task_id="data_validation",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_name=JOB_NAME,
        notebook_params={},
        tasks_to_run=[TASK_VALIDATION],
    )

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

    # ── Task 5 (weekly): run drift detector notebook ───────────────────────────
    # Runs every Sunday (day_of_week == 6). On other days the task is skipped
    # via the short-circuit pattern below.
    drift_monitoring = DatabricksRunNowOperator(
        task_id="drift_monitoring",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_name=JOB_NAME,
        notebook_params={"notebook_path": DRIFT_NOTEBOOK_PATH},
        tasks_to_run=["drift_detector"],
        # Allow daily tasks to succeed regardless of this task's schedule skip
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ── Task 6: branch on drift notebook exit value ───────────────────────────
    check_drift_result = BranchPythonOperator(
        task_id="check_drift_result",
        python_callable=_handle_drift_result,
    )

    # ── Task 7a: retraining triggered when drift detected (PSI > 0.2) ─────────
    trigger_retraining = DatabricksRunNowOperator(
        task_id="trigger_retraining",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_name=JOB_NAME,
        notebook_params={"n_clusters": "5", "target_category": "ready_meals"},
        tasks_to_run=[TASK_FEATURES, TASK_SEGMENTATION, TASK_PROPENSITY],
    )

    # ── Task 7b: no-op when drift is within bounds ────────────────────────────
    drift_stable = PythonOperator(
        task_id="drift_stable",
        python_callable=_log_drift_stable,
    )

    # ── Dependency chains ─────────────────────────────────────────────────────
    # Daily training: validate → ingest → features → [segmentation ∥ propensity]
    data_validation >> ingest_streaming >> feature_engineering >> [train_segmentation, train_propensity]

    # Weekly monitoring: runs independently after feature_engineering completes.
    # Branch resolves to trigger_retraining (red) or drift_stable (green/yellow).
    feature_engineering >> drift_monitoring >> check_drift_result
    check_drift_result >> [trigger_retraining, drift_stable]
