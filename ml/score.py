"""
Bug fix #7: Replaced Flask with FastAPI + uvicorn for production.
- Added /health liveness endpoint
- Added /ready readiness endpoint
- Proper structured error responses with HTTP status codes
- Async-compatible with Kubernetes probes
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME          = os.getenv("MODEL_NAME", "tesco-customer-segmentation")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Production")

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
    version="1.0.0",
    description="Customer segmentation and propensity scoring service",
    lifespan=lifespan,
)


# ── Request / response schemas ────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    customer_id: str
    recency_days: float = Field(..., ge=0)
    frequency: float    = Field(..., ge=0)
    monetary: float     = Field(..., ge=0)
    avg_basket_size: float = Field(default=0.0, ge=0)
    basket_std: float      = Field(default=0.0, ge=0)
    online_ratio: float    = Field(default=0.0, ge=0, le=1)
    online_txns: float     = Field(default=0.0, ge=0)
    instore_txns: float    = Field(default=0.0, ge=0)
    active_days: float     = Field(default=0.0, ge=0)


class PredictRequest(BaseModel):
    customers: list[CustomerFeatures]


class PredictionResult(BaseModel):
    customer_id: str
    segment_id: int


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    model_name: str
    model_stage: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
async def ready():
    """Readiness probe — returns 200 only when the model is loaded."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return {"status": "ready", "model": MODEL_NAME, "stage": MODEL_STAGE}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.customers:
        raise HTTPException(status_code=422, detail="customers list must not be empty")

    try:
        df = pd.DataFrame([c.model_dump() for c in request.customers])
        X  = df[FEATURE_COLS].fillna(0).values
        labels = _model.predict(X).tolist()
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    return PredictResponse(
        predictions=[
            PredictionResult(customer_id=c.customer_id, segment_id=int(label))
            for c, label in zip(request.customers, labels)
        ],
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
    )


if __name__ == "__main__":
    uvicorn.run("score:app", host="0.0.0.0", port=8080, workers=1)
