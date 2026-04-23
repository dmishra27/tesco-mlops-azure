"""
Entry-point for MLflow Projects runs and local experimentation.
Reads features from a parquet file (or ADLS path when run on Databricks),
trains a segmentation + propensity pipeline, and registers models.
"""

import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, roc_auc_score, average_precision_score


FEATURE_COLS = [
    "recency_days", "frequency", "monetary",
    "avg_basket_size", "basket_std", "online_ratio",
    "online_txns", "instore_txns", "active_days",
]


def train_segmentation(df: pd.DataFrame, n_clusters: int, experiment_name: str) -> str:
    mlflow.set_experiment(experiment_name)
    X = df[FEATURE_COLS].fillna(0).values

    with mlflow.start_run(run_name="kmeans-segmentation") as run:
        mlflow.log_params({"n_clusters": n_clusters, "feature_cols": FEATURE_COLS})

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ])
        pipeline.fit(X)

        labels = pipeline.predict(X)
        sil = silhouette_score(X, labels, sample_size=min(10_000, len(X)), random_state=42)
        mlflow.log_metrics({"silhouette_score": sil, "inertia": pipeline.named_steps["kmeans"].inertia_})
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="tesco-customer-segmentation",
        )
        print(f"[segmentation] silhouette={sil:.4f}  run_id={run.info.run_id}")
        return run.info.run_id


def train_propensity(
    df: pd.DataFrame,
    target_category: str,
    experiment_name: str,
) -> str:
    mlflow.set_experiment(experiment_name)

    feature_cols = FEATURE_COLS + ["segment_id"]
    df["label"] = df["top_categories"].apply(
        lambda cats: int(target_category in (cats or []))
    )

    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "objective": "binary",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "n_estimators": 500,
        "verbose": -1,
    }

    with mlflow.start_run(run_name=f"lgbm-propensity-{target_category}") as run:
        mlflow.log_params({**params, "target_category": target_category})

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        ap  = average_precision_score(y_val, preds)
        mlflow.log_metrics({"roc_auc": auc, "avg_precision": ap})
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"tesco-propensity-{target_category}",
        )
        print(f"[propensity/{target_category}] auc={auc:.4f} ap={ap:.4f}  run_id={run.info.run_id}")
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Tesco MLOps training entry-point")
    parser.add_argument("--features-path", required=True, help="Path to customer features parquet")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--target-category", default="ready_meals")
    parser.add_argument("--experiment-name", default="/Shared/tesco-mlops/training")
    args = parser.parse_args()

    df = pd.read_parquet(args.features_path)
    train_segmentation(df, args.n_clusters, args.experiment_name)
    train_propensity(df, args.target_category, args.experiment_name)


if __name__ == "__main__":
    main()
