"""
Tests for ml.local.drift_simulation — 4-phase drift scenario.

All tests share a single module-scoped fixture that runs the simulation once
with n_customers=300 (fast) then checks the results file on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.local.drift_simulation import run_simulation
from ml.local.visualise import plot_drift_simulation

_SIM_JSON = Path("models/drift_sim/simulation_results.json")


@pytest.fixture(scope="module")
def sim_results() -> dict:
    """Run simulation once for all tests (small n for speed)."""
    return run_simulation(n_customers=300, drift_magnitude=0.30)


def test_simulation_produces_json(sim_results):
    assert _SIM_JSON.exists(), "simulation_results.json was not created"
    data = json.loads(_SIM_JSON.read_text(encoding="utf-8"))
    required_phases = (
        "phase1_baseline",
        "phase2_stale_model_on_drifted_data",
        "phase3_retrained_model",
        "phase4_recovery_metrics",
    )
    for key in required_phases:
        assert key in data, f"simulation_results.json missing key: {key}"
    assert data["phase2_stale_model_on_drifted_data"]["drift_detected"] is True, (
        "drift_detected should be True — PSI must exceed 0.20 at drift_magnitude=0.30"
    )


def test_stale_model_degrades(sim_results):
    p1_auc = sim_results["phase1_baseline"]["test_auc"]
    p2_auc = sim_results["phase2_stale_model_on_drifted_data"]["test_auc"]
    assert p2_auc < p1_auc, (
        f"Stale model (AUC {p2_auc:.4f}) must underperform baseline "
        f"(AUC {p1_auc:.4f}) on drifted data"
    )


def test_retraining_recovers_performance(sim_results):
    p2_auc = sim_results["phase2_stale_model_on_drifted_data"]["test_auc"]
    p3_auc = sim_results["phase3_retrained_model"]["test_auc"]
    assert p3_auc > p2_auc, (
        f"Retrained model (AUC {p3_auc:.4f}) must outperform stale model "
        f"(AUC {p2_auc:.4f}) on drifted data"
    )


def test_psi_exceeds_threshold_on_drift(sim_results):
    max_psi = sim_results["phase2_stale_model_on_drifted_data"]["overall_max_psi"]
    assert max_psi > 0.20, (
        f"overall_max_psi ({max_psi:.4f}) must exceed 0.20 — "
        "PSI detector should fire at drift_magnitude=0.30"
    )


def test_drift_plot_saves_png(tmp_path, monkeypatch):
    import ml.local.visualise as vis
    monkeypatch.setattr(vis, "PLOT_DIR", str(tmp_path))

    synthetic = {
        "drift_magnitude": 0.30,
        "phase1_baseline": {
            "test_auc": 0.74, "lift_d1": 2.8,
            "psi_scores": {"recency_days": 0.01, "frequency": 0.01},
        },
        "phase2_stale_model_on_drifted_data": {
            "test_auc": 0.54, "lift_d1": 1.2,
            "psi_scores": {
                "recency_days": 0.38, "frequency": 0.25,
                "online_ratio": 0.31, "monetary": 0.04,
            },
            "overall_max_psi": 0.38,
            "features_above_threshold": ["recency_days", "frequency", "online_ratio"],
            "drift_detected": True,
        },
        "phase3_retrained_model": {"test_auc": 0.70, "lift_d1": 2.5},
        "phase4_recovery_metrics": {
            "auc_degradation": 0.20, "auc_recovery": 0.16,
            "lift_degradation": 1.6, "lift_recovery": 1.3,
            "full_recovery_achieved": False,
        },
    }
    path = plot_drift_simulation(synthetic)
    assert Path(path).exists(), f"drift_simulation.png not found at {path}"
