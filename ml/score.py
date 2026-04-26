"""
FastAPI scoring service for Tesco customer segmentation and propensity.

Design decisions:
  - /health: liveness probe — always 200, checks process not model
  - /ready:  readiness probe — 503 until model is loaded
  - /predict: returns segment_id + propensity_score + full audit metadata
  - /explain: single-customer SHAP-inspired perturbation explanation + GDPR audit log
  - Batch limit of 10,000 customers to prevent OOMKilled pod eviction
  - propensity_score derived from KMeans cluster-distance confidence
"""

from __future__ import annotations

import csv
import os
import logging
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI  = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME           = os.getenv("MODEL_NAME",    "tesco-customer-segmentation")
MODEL_STAGE          = os.getenv("MODEL_STAGE",   "Production")
MODEL_VERSION        = os.getenv("MODEL_VERSION", "1.0.0")
MAX_BATCH_SIZE       = 10_000
EXPLANATION_LOG_PATH = os.getenv(
    "EXPLANATION_LOG_PATH",
    os.path.join(tempfile.gettempdir(), "tesco_explanation_log.csv"),
)

FEATURE_COLS = [
    "recency_days", "frequency", "monetary",
    "avg_basket_size", "basket_std", "online_ratio",
    "online_txns", "instore_txns", "active_days",
]

_model: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from %s", model_uri)
    _model = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully")
    yield
    _model = None


app = FastAPI(
    title="Tesco MLOps Scoring API",
    version=MODEL_VERSION,
    description="Customer segmentation and propensity scoring service",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    customer_id:     str
    recency_days:    float = Field(...,    ge=0,  description="Days since last purchase")
    frequency:       float = Field(...,    ge=0,  description="Total transaction count")
    monetary:        float = Field(...,    gt=0,  description="Total spend in GBP")
    avg_basket_size: float = Field(default=0.0, ge=0)
    basket_std:      float = Field(default=0.0, ge=0)
    online_ratio:    float = Field(default=0.0, ge=0, le=1)
    online_txns:     float = Field(default=0.0, ge=0)
    instore_txns:    float = Field(default=0.0, ge=0)
    active_days:     float = Field(default=0.0, ge=0)


class PredictRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    customer_id:      str
    segment_id:       int
    propensity_score: float = Field(description="Cluster-confidence score in [0, 1]")


class PredictResponse(BaseModel):
    predictions:   list[PredictionResult]
    model_name:    str
    model_stage:   str
    model_version: str
    scored_at:     str   # ISO 8601


class TopFeature(BaseModel):
    feature:   str
    impact:    float
    direction: str   # "positive" | "negative"


class ExplainResponse(BaseModel):
    customer_id:      str
    propensity_score: float
    segment_id:       int
    explanation:      str
    top_features:     list[TopFeature]
    model_name:       str
    model_stage:      str
    scored_at:        str   # ISO 8601


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_propensity(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Derive a [0, 1] confidence score from the model.

    For sklearn Pipelines with a KMeans final step: use the distance to the
    nearest cluster centre normalised to (0, 1].
    Falls back to 0.5 for models without a transform method.
    """
    try:
        if hasattr(model, "named_steps"):
            # Pipeline — scale then get KMeans distances
            scaler = model.named_steps.get("scaler")
            kmeans = model.named_steps.get("kmeans")
            if scaler is not None and kmeans is not None:
                X_scaled   = scaler.transform(X)
                distances  = kmeans.transform(X_scaled)          # (n, k)
                min_dist   = distances.min(axis=1)               # (n,)
                score      = 1.0 / (1.0 + min_dist)             # (0, 1]
                return np.clip(score, 0.0, 1.0)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
    except Exception:
        pass
    return np.full(len(X), 0.5)


def _shap_approx(
    model: Any, X_raw: np.ndarray, feature_cols: list[str]
) -> list[dict]:
    """
    Perturbation-based feature importance approximating SHAP values.

    For each feature: replace its scaled value with the population mean
    (zero in StandardScaler space) and measure the resulting propensity
    score change.  Works with any sklearn Pipeline that has a 'scaler'
    and 'kmeans' step; falls back to uniform importance otherwise.
    """
    try:
        scaler = getattr(model, "named_steps", {}).get("scaler")
        kmeans = getattr(model, "named_steps", {}).get("kmeans")

        if scaler is None or kmeans is None:
            uniform = round(1.0 / max(len(feature_cols), 1), 4)
            return [
                {"feature": f, "impact": uniform, "direction": "positive"}
                for f in feature_cols
            ]

        X_scaled  = scaler.transform(X_raw)
        base_dist = float(kmeans.transform(X_scaled).min(axis=1)[0])
        base_score = 1.0 / (1.0 + base_dist)

        impacts = []
        for i, feat in enumerate(feature_cols):
            X_pert    = X_scaled.copy()
            X_pert[0, i] = 0.0                              # replace with population mean
            p_dist    = float(kmeans.transform(X_pert).min(axis=1)[0])
            p_score   = 1.0 / (1.0 + p_dist)
            impact    = abs(base_score - p_score)
            direction = "positive" if X_raw[0, i] > scaler.mean_[i] else "negative"
            impacts.append({"feature": feat, "impact": round(impact, 4), "direction": direction})

        impacts.sort(key=lambda x: -x["impact"])
        return impacts

    except Exception:
        logger.exception("SHAP approximation failed — returning uniform importance")
        return [{"feature": f, "impact": 0.0, "direction": "positive"} for f in feature_cols]


def _feature_description(feature: str, value: float, direction: str) -> str:
    above = direction == "positive"
    rel   = "above" if above else "below"
    if feature == "frequency":
        return f"their purchase frequency ({int(value)} visits) is {rel} average"
    if feature == "recency_days":
        return (
            f"they shopped recently ({int(value)} days ago)"
            if not above
            else f"their last purchase was {int(value)} days ago, above the average"
        )
    if feature == "monetary":
        return f"their total spend (£{value:.2f}) is {rel} average"
    if feature == "avg_basket_size":
        return f"their average basket size (£{value:.2f}) is {rel} average"
    if feature == "basket_std":
        return f"their basket variability (£{value:.2f} std) is {rel} average"
    if feature == "online_ratio":
        return f"their online shopping ratio ({value:.0%}) is {rel} average"
    if feature == "online_txns":
        return f"their online transaction count ({int(value)}) is {rel} average"
    if feature == "instore_txns":
        return f"their in-store transaction count ({int(value)}) is {rel} average"
    if feature == "active_days":
        return f"their activity level ({int(value)} active days) is {rel} average"
    return f"their {feature.replace('_', ' ')} ({value:.2f}) is {rel} average"


def _generate_explanation(
    propensity_score: float,
    top_features: list[dict],
    customer: CustomerFeatures,
) -> str:
    score_str = f"{propensity_score:.2f}"
    top2 = top_features[:2]

    if not top2:
        return f"This customer received a propensity score of {score_str}."

    parts = [
        _feature_description(f["feature"], getattr(customer, f["feature"]), f["direction"])
        for f in top2
    ]

    if len(parts) == 2:
        return (
            f"This customer received a propensity score of {score_str} primarily because "
            f"{parts[0]} and {parts[1]}."
        )
    return (
        f"This customer received a propensity score of {score_str} primarily because "
        f"{parts[0]}."
    )


def _log_explanation(
    customer_id: str,
    scored_at: str,
    model_version: str,
    propensity_score: float,
    top_feature: str,
    top_feature_impact: float,
) -> None:
    """Append one row to the GDPR explanation audit log — fire and forget."""
    try:
        write_header = not os.path.exists(EXPLANATION_LOG_PATH)
        with open(EXPLANATION_LOG_PATH, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "customer_id", "scored_at", "model_version",
                "propensity_score", "top_feature", "top_feature_impact",
            ])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "customer_id":        customer_id,
                "scored_at":          scored_at,
                "model_version":      model_version,
                "propensity_score":   propensity_score,
                "top_feature":        top_feature,
                "top_feature_impact": top_feature_impact,
            })
    except Exception:
        logger.exception("Explanation audit log write failed")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    """Liveness probe — returns 200 if the process is alive, regardless of model state."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
async def ready():
    """Readiness probe — returns 200 only when the model is loaded."""
    if _model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Model not yet loaded"},
        )
    return {
        "status":        "ready",
        "model_name":    MODEL_NAME,
        "model_stage":   MODEL_STAGE,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: Request, body: PredictRequest):
    if _model is None:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})

    if len(body.customers) > MAX_BATCH_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "detail": (
                    f"Batch size {len(body.customers)} exceeds limit of {MAX_BATCH_SIZE}. "
                    "Split into smaller batches."
                )
            },
        )

    try:
        df    = pd.DataFrame([c.model_dump() for c in body.customers])
        X     = df[FEATURE_COLS].fillna(0).values
        labels     = _model.predict(X).tolist()
        propensity = _compute_propensity(_model, X).tolist()
    except Exception as exc:
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"detail": f"Prediction error: {exc}"})

    return PredictResponse(
        predictions=[
            PredictionResult(
                customer_id      = c.customer_id,
                segment_id       = int(label),
                propensity_score = float(round(prop, 6)),
            )
            for c, label, prop in zip(body.customers, labels, propensity)
        ],
        model_name    = MODEL_NAME,
        model_stage   = MODEL_STAGE,
        model_version = MODEL_VERSION,
        scored_at     = datetime.now(timezone.utc).isoformat(),
    )


@app.post("/explain", response_model=ExplainResponse, tags=["inference"])
async def explain(customer: CustomerFeatures):
    """
    Return a propensity score plus a SHAP-inspired per-feature explanation
    for a single customer.  Every call is appended to the GDPR audit log.
    """
    if _model is None:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})

    try:
        df    = pd.DataFrame([customer.model_dump()])
        X     = df[FEATURE_COLS].fillna(0).values

        segment_id       = int(_model.predict(X)[0])
        propensity_score = float(round(_compute_propensity(_model, X)[0], 6))
        top_features     = _shap_approx(_model, X, FEATURE_COLS)
        explanation      = _generate_explanation(propensity_score, top_features, customer)
        scored_at        = datetime.now(timezone.utc).isoformat()

        _log_explanation(
            customer_id        = customer.customer_id,
            scored_at          = scored_at,
            model_version      = MODEL_VERSION,
            propensity_score   = propensity_score,
            top_feature        = top_features[0]["feature"] if top_features else "",
            top_feature_impact = top_features[0]["impact"]  if top_features else 0.0,
        )
    except Exception as exc:
        logger.exception("Explanation failed")
        return JSONResponse(status_code=500, content={"detail": f"Explanation error: {exc}"})

    return ExplainResponse(
        customer_id      = customer.customer_id,
        propensity_score = propensity_score,
        segment_id       = segment_id,
        explanation      = explanation,
        top_features     = [TopFeature(**f) for f in top_features],
        model_name       = MODEL_NAME,
        model_stage      = MODEL_STAGE,
        scored_at        = scored_at,
    )


if __name__ == "__main__":
    uvicorn.run("score:app", host="0.0.0.0", port=8080, workers=1)
