# Databricks notebook: 03_train_segmentation
# Trains a KMeans customer segmentation model on RFM features.
# Logs artefacts and metrics to MLflow (Databricks-managed tracking).

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

# Bug fix #6: replaced hardcoded <STORAGE_ACCOUNT> with dynamic secret lookup
STORAGE_ACCOUNT = dbutils.secrets.get(scope="adls-scope", key="STORAGE_ACCOUNT")

SILVER_PATH = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net/customer_features"
GOLD_PATH   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/customer_segments"

N_CLUSTERS = int(dbutils.widgets.get("n_clusters")) if dbutils.widgets.get("n_clusters") else 5
EXPERIMENT_NAME = "/Shared/tesco-mlops/customer-segmentation"

mlflow.set_experiment(EXPERIMENT_NAME)

# ── Load silver features ──────────────────────────────────────────────────────
feature_cols = ["recency_days", "frequency", "monetary", "avg_basket_size", "online_ratio"]
pdf = (
    spark.read.format("delta").load(SILVER_PATH)
    .select("customer_id", *feature_cols)
    .dropna()
    .toPandas()
)

X = pdf[feature_cols].values
customer_ids = pdf["customer_id"].values

# ── Train ─────────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="kmeans-segmentation") as run:
    mlflow.log_params({"n_clusters": N_CLUSTERS, "feature_cols": feature_cols})

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)),
    ])
    pipeline.fit(X)

    labels = pipeline.predict(X)
    sil_score = silhouette_score(X, labels, sample_size=min(10_000, len(X)), random_state=42)
    inertia   = pipeline.named_steps["kmeans"].inertia_

    mlflow.log_metrics({"silhouette_score": sil_score, "inertia": inertia})
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="tesco-customer-segmentation",
    )

    run_id = run.info.run_id
    print(f"MLflow run_id: {run_id} | silhouette: {sil_score:.4f} | inertia: {inertia:.2f}")

# ── Write segment assignments to gold ────────────────────────────────────────
segment_df = pd.DataFrame({"customer_id": customer_ids, "segment": labels})
sdf = spark.createDataFrame(segment_df).withColumnRenamed("segment", "segment_id")

(
    sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(GOLD_PATH)
)

dbutils.notebook.exit(run_id)
