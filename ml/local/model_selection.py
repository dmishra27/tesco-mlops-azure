"""
Model selection logic for the Tesco propensity pipeline.

Gates are applied in strict priority order. Occam's razor tiebreaker
prefers simpler models when AUC difference is within tolerance.
Ensemble models must justify their complexity with a minimum AUC gain.
"""

from __future__ import annotations

from typing import Any

from ml.config.thresholds import SELECTION_THRESHOLDS

# Lower index = simpler (Occam's razor ordering)
COMPLEXITY_ORDER: list[str] = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "voting_ensemble",
    "stacking_ensemble",
]

ENSEMBLE_NAMES: set[str] = {"voting_ensemble", "stacking_ensemble"}


# ── Exceptions ────────────────────────────────────────────────────────────────

class NoModelApprovedError(Exception):
    """Raised when no model passes all quality gates."""


# ── ModelSelector ─────────────────────────────────────────────────────────────

class ModelSelector:
    """Applies 6 selection gates and returns a fully structured report."""

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self.thr = {**SELECTION_THRESHOLDS, **(thresholds or {})}

    # ── Public API ────────────────────────────────────────────────────────────

    def select(
        self,
        models: dict[str, Any],
        metrics: dict[str, dict],
    ) -> dict[str, Any]:
        """
        Apply all gates to each model and return the best approved model.

        Parameters
        ----------
        models  : {name: fitted_model_object}
        metrics : {name: {test_auc, train_auc, cv_std, lift_at_decile1}}

        Returns
        -------
        dict with keys: selected_model_name, selected_model_object,
            selection_reason, all_gate_results, rejected_models, metrics_table
        """
        lr_auc = metrics.get("logistic_regression", {}).get("test_auc", 0.0)
        best_single_auc = max(
            m["test_auc"] for name, m in metrics.items() if name not in ENSEMBLE_NAMES
        )

        approved:  list[tuple[str, dict]] = []
        rejected:  list[dict]             = []
        all_gates: dict[str, dict]        = {}

        for name, m in metrics.items():
            passed, reason, gate_results = self._apply_all_gates(
                name, m, lr_auc, best_single_auc
            )
            all_gates[name] = gate_results
            if passed:
                approved.append((name, m))
            else:
                rejected.append({"model_name": name, "reason": reason})

        if not approved:
            fail_summary = "; ".join(
                f"{r['model_name']}: {r['reason']}" for r in rejected
            )
            raise NoModelApprovedError(
                f"no model passed all quality gates. Failures: {fail_summary}"
            )

        selected_name, selected_metrics = self._apply_tiebreakers(approved)
        selected_obj = models[selected_name]

        return {
            "selected_model_name":   selected_name,
            "selected_model_object": selected_obj,
            "selection_reason":      self._selection_reason(selected_name, approved),
            "all_gate_results":      all_gates,
            "rejected_models":       rejected,
            "metrics_table": [
                {"model": n, **m} for n, m in metrics.items()
            ],
        }

    # ── Private gate runners ──────────────────────────────────────────────────

    def _apply_all_gates(
        self,
        name: str,
        m: dict,
        lr_auc: float,
        best_single_auc: float,
    ) -> tuple[bool, str, dict]:
        gate_results: dict[str, str] = {}

        # G1 — Baseline: must beat LR by 0.03 (LR auto-passes as the baseline itself)
        g1_pass, g1_reason = self._gate_baseline(name, m, lr_auc)
        gate_results["G1"] = "PASS" if g1_pass else f"FAIL: {g1_reason}"
        if not g1_pass:
            return False, f"baseline gate — {g1_reason}", gate_results

        # G2 — Overfit
        g2_pass, g2_reason = self._gate_overfit(m)
        gate_results["G2"] = "PASS" if g2_pass else f"FAIL: {g2_reason}"
        if not g2_pass:
            return False, f"overfit gate — {g2_reason}", gate_results

        # G3 — Lift
        g3_pass, g3_reason = self._gate_lift(m)
        gate_results["G3"] = "PASS" if g3_pass else f"FAIL: {g3_reason}"
        if not g3_pass:
            return False, f"lift gate — {g3_reason}", gate_results

        # G4 — Stability
        g4_pass, g4_reason = self._gate_stability(m)
        gate_results["G4"] = "PASS" if g4_pass else f"FAIL: {g4_reason}"
        if not g4_pass:
            return False, f"stability gate — {g4_reason}", gate_results

        # G6 — Ensemble justification (only for ensemble models)
        if name in ENSEMBLE_NAMES:
            g6_pass, g6_reason = self._gate_ensemble(name, m, best_single_auc)
            gate_results["G6"] = "PASS" if g6_pass else f"FAIL: {g6_reason}"
            if not g6_pass:
                return False, f"ensemble justification — {g6_reason}", gate_results

        return True, "", gate_results

    def _gate_baseline(self, name: str, m: dict, lr_auc: float) -> tuple[bool, str]:
        """G1: model must beat LR baseline by >= 0.03 (LR itself auto-passes)."""
        if name == "logistic_regression":
            return True, ""
        gain = round(m["test_auc"] - lr_auc, 4)
        thr  = self.thr["baseline_gate_min_gain"]
        if gain < thr:
            return False, f"gain {gain:.4f} < required {thr} over logistic_regression baseline"
        return True, ""

    def _gate_overfit(self, m: dict) -> tuple[bool, str]:
        """G2: train - test gap must be below threshold."""
        gap = round(m["train_auc"] - m["test_auc"], 4)
        thr = self.thr["overfit_gap_max"]
        if gap >= thr:
            return False, f"train-test gap {gap:.3f} >= {thr}"
        return True, ""

    def _gate_lift(self, m: dict) -> tuple[bool, str]:
        """G3: top-decile lift must exceed threshold."""
        lift = m.get("lift_at_decile1", 0.0)
        thr  = self.thr["lift_decile1_min"]
        if lift < thr:
            return False, f"lift {lift:.2f} < {thr}"
        return True, ""

    def _gate_stability(self, m: dict) -> tuple[bool, str]:
        """G4: cross-validation std must be within tolerance."""
        std = m.get("cv_std", 0.0)
        thr = self.thr["cv_std_max"]
        if std >= thr:
            return False, f"cv_std {std:.3f} >= {thr:.3f}"
        return True, ""

    def _gate_ensemble(self, name: str, m: dict, best_single_auc: float) -> tuple[bool, str]:
        """G6: ensembles must gain > 0.015 AUC over best single model."""
        gain = round(m["test_auc"] - best_single_auc, 4)
        thr  = self.thr["ensemble_justification_delta"]
        if gain < thr:
            return False, f"AUC gain {gain:.4f} < required {thr} over best single model"
        return True, ""

    # ── Tiebreaker (G5 — Occam's razor) ──────────────────────────────────────

    def _apply_tiebreakers(
        self, approved: list[tuple[str, dict]]
    ) -> tuple[str, dict]:
        """Among passing models, pick highest AUC; within 0.01, prefer simpler."""
        best_auc = max(m["test_auc"] for _, m in approved)
        thr      = self.thr["tiebreaker_delta"]

        close = [(n, m) for n, m in approved if m["test_auc"] >= best_auc - thr]

        def complexity(name: str) -> int:
            try:
                return COMPLEXITY_ORDER.index(name)
            except ValueError:
                return len(COMPLEXITY_ORDER)

        close.sort(key=lambda x: complexity(x[0]))
        return close[0]

    def _selection_reason(self, selected: str, approved: list[tuple[str, dict]]) -> str:
        if len(approved) == 1:
            return f"Only model passing all quality gates."

        best_auc = max(m["test_auc"] for _, m in approved)
        thr      = self.thr["tiebreaker_delta"]
        selected_auc = next(m["test_auc"] for n, m in approved if n == selected)
        gap = round(best_auc - selected_auc, 4)

        if gap <= thr and len(approved) > 1:
            return (
                f"Occam's razor tiebreaker: {selected} selected over higher-AUC models "
                f"because gap of {gap:.4f} is within {thr} tolerance. "
                f"Simpler model preferred for interpretability."
            )
        return f"Highest test AUC among approved models: {selected_auc:.4f}"
