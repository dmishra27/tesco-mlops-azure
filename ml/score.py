"""
FastAPI scoring service for Tesco customer segmentation and propensity.

Design decisions:
  - /health: liveness probe — always 200, checks process not model
  - /ready:  readiness probe — 503 until model is loaded
  - /predict: returns segment_id + propensity_score + full audit metadata
  - Batch limit of 10,000 customers to prevent OOMKilled pod eviction
  - propensity_score derived from KMeans cluster-distance confidence
"""

from __future__ import annotations

import os
import logging
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

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME          = os.getenv("MODEL_NAME",    "tesco-customer-segmentation")
MODEL_STAGE         = os.getenv("MODEL_STAGE",   "Production")
MODEL_VERSION       = os.getenv("MODEL_VERSION", "1.0.0")
MAX_BATCH_SIZE      = 10_000

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


if __name__ == "__main__":
    uvicorn.run("score:app", host="0.0.0.0", port=8080, workers=1)
