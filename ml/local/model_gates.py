"""
Model quality gates for the Tesco MLOps pipeline.

Each gate below corresponds to a specific production failure mode.
GateFailure is structured (not just a string) so CI/CD can log failures
to MLflow and post to Slack/Teams without string parsing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ml.config.thresholds import SELECTION_THRESHOLDS


# ── Exception ─────────────────────────────────────────────────────────────────

class GateFailure(Exception):
    """
    Raised when a model fails a quality gate.

    Attributes are machine-readable so callers can log structured metadata
    to MLflow or post to Slack without parsing the string message.
    """

    def __init__(
        self,
        gate_name: str,
        actual_value: float,
        threshold: float,
        business_impact: str,
    ) -> None:
        self.gate_name       = gate_name
        self.actual_value    = actual_value
        self.threshold       = threshold
        self.business_impact = business_impact
        super().__init__(
            f"Gate '{gate_name}' failed: actual={actual_value}, threshold={threshold}. "
            f"{business_impact}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name":       self.gate_name,
            "actual_value":    self.actual_value,
            "threshold":       self.threshold,
            "business_impact": self.business_impact,
        }


# ── Segmentation gates ────────────────────────────────────────────────────────

def run_segmentation_gates(
    silhouette_score: float,
    segment_sizes: list[float],
    thresholds: dict[str, float] = SELECTION_THRESHOLDS,
) -> dict[str, Any]:
    """
    Run quality gates on KMeans segmentation results.

    Gates
    -----
    silhouette_min      : segments must be meaningfully separated
    dominant_cluster_max: no single segment may dominate (> 60%)
    tiny_cluster_min    : no segment may be too small to target (< 1%)

    Raises GateFailure on the first failing gate.
    Returns a report dict if all gates pass.
    """
    gates_checked = 0
    gates_passed  = 0

    # Gate: silhouette score
    gates_checked += 1
    thr = thresholds["silhouette_min"]
    if silhouette_score < thr:
        raise GateFailure(
            gate_name="silhouette_score",
            actual_value=round(silhouette_score, 4),
            threshold=thr,
            business_impact=(
                f"Silhouette {silhouette_score:.2f} < {thr} — customer segments overlap. "
                "Campaign targeting would be no better than random."
            ),
        )
    gates_passed += 1

    # Gate: dominant cluster
    gates_checked += 1
    thr_dom = thresholds["dominant_cluster_max"]
    max_size = max(segment_sizes)
    if max_size > thr_dom:
        pct = round(max_size * 100)
        raise GateFailure(
            gate_name="dominant_cluster",
            actual_value=round(max_size, 4),
            threshold=thr_dom,
            business_impact=(
                f"Largest cluster contains {pct}% of customers (threshold {int(thr_dom*100)}%). "
                "Cannot run differentiated campaigns against one massive group."
            ),
        )
    gates_passed += 1

    # Gate: tiny cluster
    gates_checked += 1
    thr_tiny = thresholds["tiny_cluster_min"]
    min_size = min(segment_sizes)
    if min_size <= thr_tiny:
        pct = round(min_size * 100, 1)
        raise GateFailure(
            gate_name="tiny_cluster",
            actual_value=round(min_size, 4),
            threshold=thr_tiny,
            business_impact=(
                f"Smallest cluster contains {pct}% of customers "
                f"(minimum threshold {int(thr_tiny*100)}%). "
                "Segment is too small — model found noise, not a real pattern."
            ),
        )
    gates_passed += 1

    return {
        "passed":        True,
        "gates_checked": gates_checked,
        "gates_passed":  gates_passed,
        "metrics":       {"silhouette_score": silhouette_score, "segment_sizes": segment_sizes},
        "timestamp":     datetime.now(),
    }


# ── Propensity gates ──────────────────────────────────────────────────────────

def run_propensity_gates(
    test_auc: float,
    train_auc: float,
    previous_production_auc: float,
    lift_at_decile1: float,
    thresholds: dict[str, float] = SELECTION_THRESHOLDS,
) -> dict[str, Any]:
    """
    Run quality gates on propensity model evaluation results.

    Gates
    -----
    propensity_auc_min   : model must meaningfully outperform random
    auc_regression_max   : new model must not regress vs production
    overfit_gap_max      : train/test gap must be within tolerance
    lift_decile1_min     : top-decile lift must justify campaign cost

    Raises GateFailure on the first failing gate.
    Returns a report dict if all gates pass.
    """
    gates_checked = 0
    gates_passed  = 0

    # Gate: minimum AUC
    gates_checked += 1
    thr_auc = thresholds["propensity_auc_min"]
    if test_auc < thr_auc:
        raise GateFailure(
            gate_name="test_auc",
            actual_value=round(test_auc, 4),
            threshold=thr_auc,
            business_impact=(
                f"test_auc {test_auc:.4f} < {thr_auc} — model barely outperforms random. "
                "Campaign ROI would not justify personalisation cost."
            ),
        )
    gates_passed += 1

    # Gate: AUC regression vs production
    gates_checked += 1
    thr_reg = thresholds["auc_regression_max"]
    regression = round(previous_production_auc - test_auc, 4)
    if regression > thr_reg:
        raise GateFailure(
            gate_name="auc_regression",
            actual_value=round(regression, 4),
            threshold=thr_reg,
            business_impact=(
                f"AUC regression of {regression:.2f} vs production model. "
                "Customers correctly targeted will stop receiving relevant offers."
            ),
        )
    gates_passed += 1

    # Gate: overfitting
    gates_checked += 1
    thr_gap = thresholds["overfit_gap_max"]
    gap = round(train_auc - test_auc, 4)
    if gap > thr_gap:
        raise GateFailure(
            gate_name="overfitting",
            actual_value=round(gap, 4),
            threshold=thr_gap,
            business_impact=(
                f"Train-test gap {gap:.2f} > {thr_gap} — model memorised training data. "
                "Will perform poorly on new customers not seen during training."
            ),
        )
    gates_passed += 1

    # Gate: top-decile lift
    gates_checked += 1
    thr_lift = thresholds["lift_decile1_min"]
    if lift_at_decile1 < thr_lift:
        raise GateFailure(
            gate_name="lift_at_decile1",
            actual_value=round(lift_at_decile1, 4),
            threshold=thr_lift,
            business_impact=(
                f"Lift@D1 {lift_at_decile1:.2f} < {thr_lift} — "
                "top-decile customers do not convert enough to justify targeted promotion cost."
            ),
        )
    gates_passed += 1

    return {
        "passed":        True,
        "gates_checked": gates_checked,
        "gates_passed":  gates_passed,
        "metrics": {
            "test_auc":                test_auc,
            "train_auc":               train_auc,
            "previous_production_auc": previous_production_auc,
            "lift_at_decile1":         lift_at_decile1,
        },
        "timestamp": datetime.now(),
    }
