"""
Contract tests for ml.score FastAPI service.
Written using TDD — tests define the contract; score.py must satisfy them.

Contract context: failures here correspond to production incidents:
  - 503 before model loads → AKS traffic bounced correctly
  - 413 on oversized batch → no OOMKilled pod eviction
  - audit fields → GDPR compliance and campaign post-mortems
"""

from __future__ import annotations

import sys
import os
import time
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks://mock-workspace")
os.environ.setdefault("MODEL_NAME",  "tesco-customer-segmentation")
os.environ.setdefault("MODEL_STAGE", "Production")

_mock_load = MagicMock()
with patch("mlflow.sklearn.load_model", _mock_load):
    if "ml.score" in sys.modules:
        del sys.modules["ml.score"]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
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
def app_with_model(fitted_pipeline):
    score_module._model = fitted_pipeline
    yield score_module.app
    score_module._model = None


@pytest.fixture
def app_no_model():
    score_module._model = None
    yield score_module.app


@pytest_asyncio.fixture
async def client(app_with_model):
    transport = ASGITransport(app=app_with_model)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def client_no_model(app_no_model):
    transport = ASGITransport(app=app_no_model)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200_with_correct_body(client):
    """Liveness probe must return exactly 200 + {"status": "ok"}."""
    t0 = time.monotonic()
    r = await client.get("/health")
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert elapsed_ms < 500  # generous local allowance


@pytest.mark.asyncio
async def test_health_always_returns_200_even_without_model(client_no_model):
    """Health checks container liveness, NOT model readiness — must return 200 always."""
    r = await client_no_model.get("/health")
    assert r.status_code == 200


# ── /ready ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ready_returns_503_before_model_loaded(client_no_model):
    """Readiness probe must return 503 before model loads so AKS withholds traffic."""
    r = await client_no_model.get("/ready")
    assert r.status_code == 503
    assert r.json().get("status") == "not_ready"


@pytest.mark.asyncio
async def test_ready_returns_200_when_model_loaded(client):
    """After model load, readiness probe returns 200 with model metadata."""
    r = await client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert "model_name"    in body
    assert "model_stage"   in body
    assert "model_version" in body


# ── /predict — validation ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_rejects_missing_feature(client):
    """Missing field must return 422, not 500."""
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "frequency"}
    r = await client.post("/predict", json={"customers": [bad]})
    assert r.status_code == 422
    body_str = r.text
    assert "frequency" in body_str
    assert "traceback" not in body_str.lower()
    assert "site-packages" not in body_str.lower()


@pytest.mark.asyncio
async def test_predict_rejects_negative_recency(client):
    """Domain validation must happen at API boundary, not inside the model."""
    bad = {**VALID_CUSTOMER, "recency_days": -5.0}
    r = await client.post("/predict", json={"customers": [bad]})
    assert r.status_code == 422
    assert "recency_days" in r.text


@pytest.mark.asyncio
async def test_predict_rejects_negative_monetary(client):
    """Negative monetary value must be rejected at the API boundary."""
    bad = {**VALID_CUSTOMER, "monetary": -100.0}
    r = await client.post("/predict", json={"customers": [bad]})
    assert r.status_code == 422
    assert "monetary" in r.text


# ── /predict — response contract ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_returns_valid_segment_id(client):
    """segment_id must be a non-negative integer — out-of-range crashes CRM lookup."""
    r = await client.post("/predict", json={"customers": [VALID_CUSTOMER]})
    assert r.status_code == 200
    pred = r.json()["predictions"][0]
    assert "segment_id" in pred
    assert isinstance(pred["segment_id"], int)
    assert pred["segment_id"] >= 0


@pytest.mark.asyncio
async def test_predict_returns_propensity_between_0_and_1(client):
    """propensity_score is used as a probability — values outside [0,1] break budgets."""
    r = await client.post("/predict", json={"customers": [VALID_CUSTOMER]})
    assert r.status_code == 200
    pred = r.json()["predictions"][0]
    assert "propensity_score" in pred
    assert 0.0 <= pred["propensity_score"] <= 1.0


@pytest.mark.asyncio
async def test_predict_enforces_batch_limit(client):
    """10001 customers in one request must return 413 — prevent OOMKilled eviction."""
    customers = [{**VALID_CUSTOMER, "customer_id": f"CUST-{i:05d}"} for i in range(10_001)]
    r = await client.post("/predict", json={"customers": customers})
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_predict_response_includes_audit_fields(client):
    """Every batch must include audit metadata for GDPR compliance and post-mortems."""
    r = await client.post("/predict", json={"customers": [VALID_CUSTOMER]})
    assert r.status_code == 200
    body = r.json()
    assert "model_name"    in body
    assert "model_stage"   in body
    assert "scored_at"     in body
    assert "model_version" in body


@pytest.mark.asyncio
async def test_predict_handles_single_customer(client):
    """Batch of 1 must work — real-time personalisation scores one at a time."""
    r = await client.post("/predict", json={"customers": [VALID_CUSTOMER]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 1


# ── /explain ──────────────────────────────────────────────────────────────────

VALID_EXPLAIN = {
    "customer_id":     "CUST-0042",
    "recency_days":    14.0,
    "frequency":       28.0,
    "monetary":        412.50,
    "avg_basket_size": 14.73,
    "basket_std":      4.20,
    "online_ratio":    0.65,
    "online_txns":     18.0,
    "instore_txns":    10.0,
    "active_days":     22.0,
}


@pytest.mark.asyncio
async def test_explain_returns_200(client):
    """POST /explain with a valid single-customer payload must return 200."""
    r = await client.post("/explain", json=VALID_EXPLAIN)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_explain_response_has_explanation_field(client):
    """Response must contain a non-empty explanation sentence (> 20 characters)."""
    r = await client.post("/explain", json=VALID_EXPLAIN)
    assert r.status_code == 200
    body = r.json()
    assert "explanation" in body
    assert isinstance(body["explanation"], str)
    assert len(body["explanation"]) > 20


@pytest.mark.asyncio
async def test_explain_top_features_match_model_features(client):
    """Every feature name returned in top_features must be a valid model feature column."""
    r = await client.post("/explain", json=VALID_EXPLAIN)
    assert r.status_code == 200
    top_features = r.json()["top_features"]
    assert len(top_features) > 0
    for f in top_features:
        assert f["feature"] in score_module.FEATURE_COLS


@pytest.mark.asyncio
async def test_explain_includes_audit_fields(client):
    """Response must carry model_name, model_stage, and scored_at for GDPR audit trail."""
    r = await client.post("/explain", json=VALID_EXPLAIN)
    assert r.status_code == 200
    body = r.json()
    assert "model_name"  in body
    assert "model_stage" in body
    assert "scored_at"   in body
