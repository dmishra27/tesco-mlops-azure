"""
DAG: tesco_weekly_scoring
Loads the Production segmentation model from MLflow Registry,
scores all customers in the gold layer, and writes propensity
scores back to gold/propensity_scores (Delta, partitioned by
run_date).

Runs weekly on Sunday at 03:00 UTC.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import mlflow
import mlflow.sklearn
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
STORAGE_ACCOUNT     = os.environ["STORAGE_ACCOUNT"]
GOLD_SEG_PATH       = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/customer_segments"
GOLD_PROP_PATH      = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/propensity_scores"
SILVER_PATH         = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net/customer_features"

SEGMENTATION_MODEL  = "tesco-customer-segmentation"
PROPENSITY_MODEL    = "tesco-propensity-ready_meals"
MODEL_STAGE         = "Production"

FEATURE_COLS = [
    "recency_days", "frequency", "monetary",
    "avg_basket_size", "basket_std", "online_ratio",
    "online_txns", "instore_txns", "active_days",
]

default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email": ["ml-platform-alerts@tesco.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ── Callable task functions ───────────────────────────────────────────────────

def load_gold_customers(**context) -> None:
    """
    Reads customer features from silver and pushes the count to XCom
    so downstream tasks can log it.  The actual DataFrame is written
    to a temp parquet so it doesn't have to re-read on every task.
    """
    from pyspark.sql import SparkSession  # available when run on Databricks / MWAA with Spark

    spark = SparkSession.builder.getOrCreate()
    run_date = context["ds"]  # YYYY-MM-DD from Airflow logical date

    features = (
        spark.read.format("delta").load(SILVER_PATH)
        .select("customer_id", *FEATURE_COLS)
        .dropna()
    )
    segments = spark.read.format("delta").load(GOLD_SEG_PATH)

    pdf = (
        features.join(segments, "customer_id", "left")
        .toPandas()
    )

    tmp_path = f"/tmp/tesco_scoring_{run_date}.parquet"
    pdf.to_parquet(tmp_path, index=False)
    context["ti"].xcom_push(key="tmp_path", value=tmp_path)
    context["ti"].xcom_push(key="row_count", value=len(pdf))
    print(f"[load_gold_customers] Loaded {len(pdf):,} customers for {run_date}")


def score_customers(**context) -> None:
    """
    Loads Production models from MLflow Registry, scores every
    customer, and writes results to gold/propensity_scores (Delta).
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    run_date = context["ds"]
    tmp_path = context["ti"].xcom_pull(key="tmp_path")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load segmentation model
    seg_model = mlflow.sklearn.load_model(
        f"models:/{SEGMENTATION_MODEL}/{MODEL_STAGE}"
    )
    # Load propensity model
    prop_model = mlflow.lightgbm.load_model(
        f"models:/{PROPENSITY_MODEL}/{MODEL_STAGE}"
    )

    pdf = pd.read_parquet(tmp_path)
    X = pdf[FEATURE_COLS].fillna(0).values

    segment_ids      = seg_model.predict(X).tolist()
    propensity_scores = prop_model.predict_proba(X)[:, 1].tolist()

    results = pd.DataFrame({
        "customer_id":      pdf["customer_id"],
        "segment_id":       segment_ids,
        "propensity_score": propensity_scores,
        "target_category":  PROPENSITY_MODEL.replace("tesco-propensity-", ""),
        "run_date":         run_date,
        "model_version":    MODEL_STAGE,
    })

    sdf = spark.createDataFrame(results)
    (
        sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("replaceWhere", f"run_date = '{run_date}'")
        .partitionBy("run_date")
        .save(GOLD_PROP_PATH)
    )

    row_count = context["ti"].xcom_pull(key="row_count")
    print(
        f"[score_customers] Scored {row_count:,} customers → "
        f"{GOLD_PROP_PATH} (run_date={run_date})"
    )


def validate_scores(**context) -> None:
    """
    Sanity-checks the written partition: confirms row count matches
    source and that propensity scores are in [0, 1].
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.getOrCreate()
    run_date = context["ds"]
    expected = context["ti"].xcom_pull(key="row_count")

    written = (
        spark.read.format("delta").load(GOLD_PROP_PATH)
        .filter(F.col("run_date") == run_date)
    )

    actual = written.count()
    out_of_range = written.filter(
        (F.col("propensity_score") < 0) | (F.col("propensity_score") > 1)
    ).count()

    if actual != expected:
        raise ValueError(
            f"Row count mismatch: expected {expected:,}, got {actual:,}"
        )
    if out_of_range > 0:
        raise ValueError(
            f"{out_of_range:,} rows have propensity scores outside [0, 1]"
        )
    print(f"[validate_scores] Validation passed: {actual:,} rows, all scores in [0,1]")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="tesco_weekly_scoring",
    description="Weekly batch scoring: load Production MLflow models, score gold customers, write propensity scores",
    default_args=default_args,
    schedule_interval="0 3 * * 0",   # 03:00 UTC every Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["tesco", "mlops", "scoring", "batch"],
) as dag:

    load_customers = PythonOperator(
        task_id="load_gold_customers",
        python_callable=load_gold_customers,
    )

    score = PythonOperator(
        task_id="score_customers",
        python_callable=score_customers,
    )

    validate = PythonOperator(
        task_id="validate_scores",
        python_callable=validate_scores,
    )

    # load → score → validate
    load_customers >> score >> validate
