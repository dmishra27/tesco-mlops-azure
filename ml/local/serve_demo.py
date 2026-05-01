"""
FastAPI inference demo — scores customers via /predict and /explain.

Runs against the ASGI test client (no live server required) using the same
pattern as test_score_api_tdd.py: sets score_module._model directly so the
MLflow lifespan is never invoked.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks://mock-workspace")
os.environ.setdefault("MODEL_NAME",  "tesco-customer-segmentation")
os.environ.setdefault("MODEL_STAGE", "Production")

from httpx import ASGITransport, AsyncClient
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ml.score as score_module

SCORED_CSV  = Path("data/results/scored_customers.csv")
OUT_DIR     = Path("models/inference_demo")
REPORT_PATH = Path("docs/inference_demo_report.md")

FAST_CONFIG: dict = {
    "n_customers":    1500,
    "n_transactions": 15000,
    "n_optuna_trials": 20,
    "seed":           42,
    "nonlinear":      True,
    "out_dir":        "data",
}


def _make_fitted_pipeline() -> Pipeline:
    """Minimal KMeans pipeline — same construction as conftest.py fitted_pipeline."""
    rng = np.random.default_rng(0)
    X = rng.random((50, len(score_module.FEATURE_COLS)))
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=3, random_state=0, n_init=5)),
    ])
    pipeline.fit(X)
    return pipeline


def _generate_scored_customers() -> None:
    from ml.local.run_pipeline import run_pipeline
    run_pipeline(FAST_CONFIG)


def _load_test_data(n: int = 200) -> pd.DataFrame:
    if not SCORED_CSV.exists():
        _generate_scored_customers()
    df = pd.read_csv(SCORED_CSV).head(n).copy()
    for col in score_module.FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df["customer_id"] = df["customer_id"].astype(str)
    df["monetary"]    = df["monetary"].clip(lower=0.01)
    return df


def _df_to_customers(df: pd.DataFrame) -> list[dict]:
    feature_cols = score_module.FEATURE_COLS
    rows = []
    for _, row in df.iterrows():
        entry: dict = {"customer_id": str(row["customer_id"])}
        for col in feature_cols:
            entry[col] = float(row.get(col, 0) or 0)
        rows.append(entry)
    return rows


async def _run_async(n_customers: int) -> dict:
    df        = _load_test_data(n_customers)
    customers = _df_to_customers(df)

    pipeline = _make_fitted_pipeline()
    score_module._model = pipeline
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        transport = ASGITransport(app=score_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:

            # ── Step 3: POST /predict (batch) ─────────────────────────────────
            r = await client.post("/predict", json={"customers": customers})
            r.raise_for_status()
            predict_body = r.json()
            predictions  = predict_body["predictions"]
            scores       = np.array([p["propensity_score"] for p in predictions])

            dist = {
                "mean": round(float(scores.mean()), 6),
                "std":  round(float(scores.std()),  6),
                "min":  round(float(scores.min()),  6),
                "p10":  round(float(np.percentile(scores, 10)), 6),
                "p25":  round(float(np.percentile(scores, 25)), 6),
                "p50":  round(float(np.percentile(scores, 50)), 6),
                "p75":  round(float(np.percentile(scores, 75)), 6),
                "p90":  round(float(np.percentile(scores, 90)), 6),
                "p95":  round(float(np.percentile(scores, 95)), 6),
                "p99":  round(float(np.percentile(scores, 99)), 6),
                "max":  round(float(scores.max()),  6),
            }
            seg_dist = {
                str(k): v
                for k, v in sorted(
                    Counter(p["segment_id"] for p in predictions).items()
                )
            }
            sorted_desc = sorted(predictions, key=lambda x: -x["propensity_score"])
            top_10      = sorted_desc[:10]
            bottom_10   = sorted(predictions, key=lambda x: x["propensity_score"])[:10]

            results_predict = {
                "scored_at":            predict_body["scored_at"],
                "n_customers":          len(predictions),
                "model_name":           predict_body["model_name"],
                "model_stage":          predict_body["model_stage"],
                "predictions":          predictions,
                "score_distribution":   dist,
                "segment_distribution": seg_dist,
                "top_10_customers":     top_10,
                "bottom_10_customers":  bottom_10,
            }
            (OUT_DIR / "results_predict.json").write_text(
                json.dumps(results_predict, indent=2), encoding="utf-8"
            )
            (OUT_DIR / "test_input_sample.json").write_text(
                json.dumps(customers[:20], indent=2), encoding="utf-8"
            )

            # ── Step 4: POST /explain (top 5 individually) ───────────────────
            top5_ids   = {p["customer_id"] for p in sorted_desc[:5]}
            top5_custs = [c for c in customers if c["customer_id"] in top5_ids][:5]
            explain_list: list[dict] = []
            for cust in top5_custs:
                r2 = await client.post("/explain", json=cust)
                r2.raise_for_status()
                explain_list.append(r2.json())

            (OUT_DIR / "results_explain.json").write_text(
                json.dumps(explain_list, indent=2), encoding="utf-8"
            )

    finally:
        score_module._model = None

    _write_report(results_predict, explain_list, n_customers)

    return {
        "n_customers":          len(predictions),
        "n_explain":            len(explain_list),
        "score_distribution":   dist,
        "segment_distribution": seg_dist,
    }


def _write_report(predict_data: dict, explain_data: list, n_customers: int) -> None:
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    dist  = predict_data["score_distribution"]
    seg   = predict_data["segment_distribution"]
    top10 = predict_data["top_10_customers"]
    total = predict_data["n_customers"]

    lines = [
        "# FastAPI Inference Demo Report",
        f"Generated: {now}",
        "Model: tesco-customer-segmentation Production",
        f"Test customers: {n_customers}",
        "",
        "## Score Distribution",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for metric in ("mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"):
        lines.append(f"| {metric.upper():<6} | {dist[metric]:.3f} |")

    lines += [
        "",
        "## Segment Distribution",
        "",
        "| Segment | Count | % of customers |",
        "|---------|-------|----------------|",
    ]
    for sid, cnt in seg.items():
        pct = cnt / total * 100
        lines.append(f"| {sid:<7} | {cnt:<5} | {pct:.1f}%           |")

    lines += [
        "",
        "## Top 10 Customers by Propensity Score",
        "",
        "| customer_id | propensity_score | segment_id |",
        "|-------------|-----------------|------------|",
    ]
    for p in top10:
        lines.append(
            f"| {p['customer_id']:<11} | {p['propensity_score']:.6f}       | {p['segment_id']}          |"
        )

    lines += ["", "## Sample Explanations (Top 5 Customers)", ""]

    for exp in explain_data:
        lines += [
            f"### Customer ID: {exp['customer_id']}",
            f"**Propensity Score:** {exp['propensity_score']:.3f}",
            f"**Explanation:** {exp['explanation']}",
            "",
            "**Top Features:**",
        ]
        for i, feat in enumerate(exp.get("top_features", [])[:5], 1):
            lines.append(
                f"  {i}. {feat['feature']}: impact {feat['impact']:.4f} ({feat['direction']})"
            )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(n_customers: int = 200) -> dict:
    summary = asyncio.run(_run_async(n_customers))
    print("\n" + "=" * 60)
    print("FastAPI Inference Demo — COMPLETE")
    print("=" * 60)
    print(f"  Customers scored    : {summary['n_customers']}")
    print(f"  Customers explained : {summary['n_explain']}")
    print(f"  Score mean          : {summary['score_distribution']['mean']:.4f}")
    print(f"  Score p50           : {summary['score_distribution']['p50']:.4f}")
    print(f"  Score p90           : {summary['score_distribution']['p90']:.4f}")
    print(f"  Segments            : {summary['segment_distribution']}")
    print()
    print("  Output files:")
    print("    models/inference_demo/results_predict.json")
    print("    models/inference_demo/results_explain.json")
    print("    models/inference_demo/test_input_sample.json")
    print("    docs/inference_demo_report.md")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    main()
