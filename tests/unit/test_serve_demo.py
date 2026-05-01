"""
Tests for ml.local.serve_demo — FastAPI inference demo.

All three tests share a single module-scoped fixture that calls main() once
with n_customers=50 (fast) and checks the produced files on disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks://mock-workspace")
os.environ.setdefault("MODEL_NAME",  "tesco-customer-segmentation")
os.environ.setdefault("MODEL_STAGE", "Production")

from ml.local.serve_demo import main


@pytest.fixture(scope="module")
def demo_output():
    """Run the demo once for all tests in this module (n=50 for speed)."""
    main(n_customers=50)


def test_serve_demo_produces_predict_json(demo_output):
    path = Path("models/inference_demo/results_predict.json")
    assert path.exists(), "results_predict.json was not created"
    data = json.loads(path.read_text(encoding="utf-8"))
    required_keys = (
        "scored_at", "n_customers", "predictions",
        "score_distribution", "segment_distribution", "top_10_customers",
    )
    for key in required_keys:
        assert key in data, f"results_predict.json missing key: {key}"
    assert isinstance(data["predictions"], list)
    assert data["n_customers"] > 0


def test_serve_demo_produces_explain_json(demo_output):
    path = Path("models/inference_demo/results_explain.json")
    assert path.exists(), "results_explain.json was not created"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "results_explain.json should be a JSON array"
    assert len(data) == 5, f"Expected 5 explain responses, got {len(data)}"
    for item in data:
        for key in ("customer_id", "explanation", "top_features", "propensity_score"):
            assert key in item, f"Explain response missing key: {key}"


def test_serve_demo_produces_markdown_report(demo_output):
    path = Path("docs/inference_demo_report.md")
    assert path.exists(), "inference_demo_report.md was not created"
    content = path.read_text(encoding="utf-8")
    assert "Score Distribution"   in content
    assert "Segment Distribution" in content
    assert "Top 10 Customers"     in content
