"""
Unit tests for the FastAPI scoring service (ml/score.py).

Uses httpx.AsyncClient with ASGITransport so tests run without
a real server or a real MLflow connection. The fitted_pipeline
fixture from conftest.py is injected via mock_mlflow_model.
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Bootstrap: set required env vars before importing score ───────────────────
os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks://mock-workspace")
os.environ.setdefault("MODEL_NAME",  "tesco-customer-segmentation")
os.environ.setdefault("MODEL_STAGE", "Production")

# Patch mlflow.sklearn.load_model at module level so the lifespan
# startup doesn't attempt a real registry call during import.
_mock_load = MagicMock()

with patch("mlflow.sklearn.load_model", _mock_load):
    # Ensure a clean import when running the full suite
    if "ml.score" in sys.modules:
        del sys.modules["ml.score"]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import importlib
    import ml.score as score_module


VALID_CUSTOMER = {
    "customer_id":     "CUST-0001",
    "recency_days":    10.0,
    "frequency":       25.0,
    "monetary":        350.50,
    "avg_basket_size": 14.02,
    "basket_std":      3.50,
    "online_ratio":    0.6,
    "online_txns":     15.0,
    "instore_txns":    10.0,
    "active_days":     20.0,
}


@pytest.fixture
def app_with_mock_model(fitted_pipeline):
    """
    Returns the FastAPI app with _model pre-loaded (bypasses lifespan).
    """
    score_module._model = fitted_pipeline
    yield score_module.app
    score_module._model = None


@pytest_asyncio.fixture
async def async_client(app_with_mock_model):
    transport = ASGITransport(app=app_with_mock_model)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ── /health ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200(async_client):
    response = await async_client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_body(async_client):
    response = await async_client.get("/health")
    assert response.json() == {"status": "ok"}


# ── /ready ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ready_returns_200_when_model_loaded(async_client):
    response = await async_client.get("/ready")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_ready_body_contains_model_info(async_client):
    response = await async_client.get("/ready")
    body = response.json()
    assert body["status"] == "ready"
    assert "model" in body
    assert "stage" in body


@pytest.mark.asyncio
async def test_ready_returns_503_when_model_not_loaded(app_with_mock_model):
    score_module._model = None
    transport = ASGITransport(app=app_with_mock_model)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")
    assert response.status_code == 503
    score_module._model = app_with_mock_model  # restore (fixture teardown handles None)


# ── /predict ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_valid_single_customer(async_client):
    payload = {"customers": [VALID_CUSTOMER]}
    response = await async_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1
    pred = body["predictions"][0]
    assert pred["customer_id"] == "CUST-0001"
    assert isinstance(pred["segment_id"], int)


@pytest.mark.asyncio
async def test_predict_valid_batch(async_client):
    customers = [
        {**VALID_CUSTOMER, "customer_id": f"CUST-{i:04d}"}
        for i in range(5)
    ]
    response = await async_client.post("/predict", json={"customers": customers})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 5


@pytest.mark.asyncio
async def test_predict_response_includes_model_metadata(async_client):
    payload = {"customers": [VALID_CUSTOMER]}
    response = await async_client.post("/predict", json=payload)
    body = response.json()
    assert "model_name" in body
    assert "model_stage" in body


@pytest.mark.asyncio
async def test_predict_segment_ids_are_non_negative_integers(async_client):
    customers = [
        {**VALID_CUSTOMER, "customer_id": f"CUST-{i:04d}"}
        for i in range(10)
    ]
    response = await async_client.post("/predict", json={"customers": customers})
    assert response.status_code == 200
    for pred in response.json()["predictions"]:
        assert isinstance(pred["segment_id"], int)
        assert pred["segment_id"] >= 0


# ── /predict — malformed payloads (expect 422) ────────────────────────────────

@pytest.mark.asyncio
async def test_predict_missing_customers_key_returns_422(async_client):
    response = await async_client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_empty_customers_list_returns_422(async_client):
    """Empty list is explicitly rejected by the endpoint."""
    response = await async_client.post("/predict", json={"customers": []})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_missing_required_field_returns_422(async_client):
    """recency_days is required (ge=0 constraint); omitting it must 422."""
    bad_customer = {k: v for k, v in VALID_CUSTOMER.items() if k != "recency_days"}
    response = await async_client.post("/predict", json={"customers": [bad_customer]})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_negative_recency_returns_422(async_client):
    """recency_days has ge=0 — a negative value must be rejected."""
    bad = {**VALID_CUSTOMER, "recency_days": -1.0}
    response = await async_client.post("/predict", json={"customers": [bad]})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_online_ratio_above_1_returns_422(async_client):
    """online_ratio has le=1 constraint."""
    bad = {**VALID_CUSTOMER, "online_ratio": 1.5}
    response = await async_client.post("/predict", json={"customers": [bad]})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_non_json_body_returns_422(async_client):
    response = await async_client.post(
        "/predict",
        content=b"not json at all",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
