"""
Model quality gate — runs between build and deploy.

Pulls the latest Staging model from the MLflow registry, validates its
training-run metrics against the same thresholds used by ModelSelector,
writes a structured summary to the GitHub Actions step summary, and either
transitions the model to Production (all gates pass) or exits with code 1
(any gate fails), which blocks the downstream deploy job.

Required environment variables:
  MLFLOW_TRACKING_URI  – set to "databricks" for Databricks-hosted MLflow
  DATABRICKS_HOST      – https://<workspace>.azuredatabricks.net
  DATABRICKS_TOKEN     – Databricks PAT
  REGISTERED_MODEL     – MLflow registered model name (default: tesco-propensity)

Thresholds mirror ml/local/model_selection.py::SELECTION_THRESHOLDS so there
is one source of truth.  Update both files together when requirements change.
"""

from __future__ import annotations

import os
import sys

import mlflow
from mlflow import MlflowClient

# ── Thresholds (mirror ml/local/model_selection.py::SELECTION_THRESHOLDS) ─────
THRESHOLDS = {
    "min_test_auc":     0.70,   # G1 – must beat this absolute floor
    "max_overfit_gap":  0.08,   # G2 – train_auc − test_auc
    "min_lift_decile1": 2.5,    # G3 – top-decile lift
    "max_cv_std":       0.03,   # G4 – cross-val AUC std
    "min_silhouette":   0.25,   # G5 – KMeans silhouette
    "min_segment_size": 0.05,   # G6 – no segment smaller than 5 %
    "max_segment_size": 0.70,   # G6 – no segment larger than 70 %
}

MODEL_NAME = os.environ.get("REGISTERED_MODEL", "tesco-propensity")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _append_summary(*lines: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    text = "\n".join(lines) + "\n"
    if path:
        with open(path, "a") as fh:
            fh.write(text)
    else:
        print(text, file=sys.stderr)


def _set_output(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT", "")
    line = f"{key}={value}\n"
    if path:
        with open(path, "a") as fh:
            fh.write(line)
    else:
        print(line, file=sys.stderr)


def _gate(label: str, value: float, threshold: float, op: str) -> tuple[bool, str]:
    """Return (passed, summary_row). op is '>=' or '<='."""
    if op == ">=":
        passed = value >= threshold
        sym = "≥"
    else:
        passed = value <= threshold
        sym = "≤"
    status = "✅ PASS" if passed else "❌ FAIL"
    row = f"| {label} | {value:.4f} | {sym} {threshold} | {status} |"
    return passed, row


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    client = MlflowClient()

    # 1 ── Pull latest Staging version ────────────────────────────────────────
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    except Exception as exc:
        _append_summary(
            "## ❌ Model Quality Gate — REGISTRY ERROR",
            "",
            f"Could not reach MLflow registry: `{exc}`",
            "",
            "**Deploy is blocked.**",
        )
        _set_output("gate_passed", "false")
        return 1

    if not versions:
        _append_summary(
            "## ❌ Model Quality Gate — NO STAGING MODEL",
            "",
            f"No version of **{MODEL_NAME}** found in stage `Staging`.",
            "Register and promote a model before deploying.",
            "",
            "**Deploy is blocked.**",
        )
        _set_output("gate_passed", "false")
        return 1

    mv = versions[0]
    model_version: str = mv.version
    run_id: str = mv.run_id
    _set_output("model_version", model_version)

    # 2 ── Fetch metrics from the training run ─────────────────────────────────
    run = client.get_run(run_id)
    m = run.data.metrics

    test_auc  = m.get("test_auc",        0.0)
    train_auc = m.get("train_auc",       test_auc)   # fallback avoids false G2 fail
    cv_std    = m.get("cv_std",          0.0)
    lift      = m.get("lift_at_decile1", 0.0)
    silhouette = m.get("silhouette_score", 0.0)
    overfit   = train_auc - test_auc

    segment_sizes = [
        m.get(f"segment_size_{i}") for i in range(3)
        if m.get(f"segment_size_{i}") is not None
    ]

    # 3 ── Run gates ───────────────────────────────────────────────────────────
    rows: list[str] = []
    failures: list[str] = []

    checks = [
        ("G1 – min test AUC",  test_auc,   THRESHOLDS["min_test_auc"],     ">="),
        ("G2 – overfit gap",   overfit,    THRESHOLDS["max_overfit_gap"],  "<="),
        ("G3 – lift @ D1",     lift,       THRESHOLDS["min_lift_decile1"], ">="),
        ("G4 – CV std",        cv_std,     THRESHOLDS["max_cv_std"],       "<="),
        ("G5 – silhouette",    silhouette, THRESHOLDS["min_silhouette"],   ">="),
    ]
    for label, value, threshold, op in checks:
        passed, row = _gate(label, value, threshold, op)
        rows.append(row)
        if not passed:
            failures.append(f"{label}: got {value:.4f}, required {op} {threshold}")

    for i, size in enumerate(segment_sizes):
        p_lo, row_lo = _gate(f"G6 – segment {i} min size", size,
                             THRESHOLDS["min_segment_size"], ">=")
        p_hi, row_hi = _gate(f"G6 – segment {i} max size", size,
                             THRESHOLDS["max_segment_size"], "<=")
        rows += [row_lo, row_hi]
        if not p_lo:
            failures.append(f"G6: segment {i} too small ({size:.1%})")
        if not p_hi:
            failures.append(f"G6: segment {i} too large ({size:.1%})")

    # 4 ── Write summary and exit ──────────────────────────────────────────────
    header = [
        f"**Model:** `{MODEL_NAME}` v{model_version}  ",
        f"**Run ID:** `{run_id}`  ",
        "",
        "| Gate | Value | Threshold | Status |",
        "|---|---|---|---|",
    ]

    if failures:
        _append_summary(
            "## ❌ Model Quality Gate — FAILED",
            "",
            *header,
            *rows,
            "",
            "### Failed gates",
            "",
            *[f"- {f}" for f in failures],
            "",
            "> Model **NOT** promoted. Deploy is blocked.",
        )
        _set_output("gate_passed", "false")
        return 1

    # 5 ── All gates passed — promote to Production ───────────────────────────
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Production",
        archive_existing_versions=True,
    )

    _append_summary(
        "## ✅ Model Quality Gate — PASSED",
        "",
        *header,
        *rows,
        "",
        f"> `{MODEL_NAME}` v{model_version} promoted **Staging → Production**.",
        "> Deploy proceeding.",
    )
    _set_output("gate_passed", "true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
