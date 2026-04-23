# Databricks notebook: 04_propensity_model
# Trains a LightGBM propensity-to-buy model per product category.
# Input: silver customer features + gold segment labels.
# Output: per-customer propensity scores written to gold layer.

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Bug fix #6: replaced hardcoded <STORAGE_ACCOUNT> with dynamic secret lookup
STORAGE_ACCOUNT = dbutils.secrets.get(scope="adls-scope", key="STORAGE_ACCOUNT")

SILVER_PATH   = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net/customer_features"
GOLD_SEG_PATH = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/customer_segments"
GOLD_PROP_PATH = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net/propensity_scores"

TARGET_CATEGORY  = dbutils.widgets.get("target_category") or "ready_meals"
EXPERIMENT_NAME  = "/Shared/tesco-mlops/propensity-modelling"

mlflow.set_experiment(EXPERIMENT_NAME)

# ── Load data ─────────────────────────────────────────────────────────────────
features_sdf = spark.read.format("delta").load(SILVER_PATH)
segments_sdf = spark.read.format("delta").load(GOLD_SEG_PATH)

pdf = (
    features_sdf
    .join(segments_sdf, "customer_id", "left")
    .toPandas()
)

feature_cols = [
    "recency_days", "frequency", "monetary",
    "avg_basket_size", "basket_std", "online_ratio",
    "online_txns", "instore_txns", "active_days", "segment_id",
]

# Synthetic target: customers who spent in target_category (replace with real label join)
pdf["label"] = (
    pdf["top_categories"]
    .apply(lambda cats: int(TARGET_CATEGORY in (cats or [])))
)

X = pdf[feature_cols].fillna(0).values
y = pdf["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train ─────────────────────────────────────────────────────────────────────
params = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
}

with mlflow.start_run(run_name=f"lgbm-propensity-{TARGET_CATEGORY}") as run:
    mlflow.log_params({**params, "target_category": TARGET_CATEGORY})

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc   = roc_auc_score(y_val, preds)
    ap    = average_precision_score(y_val, preds)

    mlflow.log_metrics({"roc_auc": auc, "avg_precision": ap})
    mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        registered_model_name=f"tesco-propensity-{TARGET_CATEGORY}",
    )
    run_id = run.info.run_id
    print(f"run_id: {run_id} | AUC: {auc:.4f} | AP: {ap:.4f}")

# ── Score all customers and write to gold ────────────────────────────────────
all_scores = model.predict_proba(pdf[feature_cols].fillna(0).values)[:, 1]
result = pd.DataFrame({
    "customer_id": pdf["customer_id"],
    "target_category": TARGET_CATEGORY,
    "propensity_score": all_scores,
    "segment_id": pdf["segment_id"],
})

sdf = spark.createDataFrame(result)
(
    sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("target_category")
    .save(GOLD_PROP_PATH)
)

dbutils.notebook.exit(run_id)
