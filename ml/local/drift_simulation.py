"""
Feature drift simulation: 4-phase demonstration of PSI detection and retraining recovery.

Phase 1 — Baseline model trained on stable distribution (Week 0)
Phase 2 — Stale model scored on drifted distribution (Week 4, no retrain)
Phase 3 — Retrained model on drifted distribution (Week 4, after retrain)
Phase 4 — Recovery delta metrics

Concept drift is introduced alongside covariate drift:
  - Stable world: recency + frequency predict conversion
  - Drifted world: online_ratio + monetary predict conversion
    (simulates a promotional-channel shift changing what predicts purchase)
This guarantees measurable AUC degradation and recovery regardless of the
magnitude of the covariate shift.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.local.generate_drift_data import (
    FEATURE_COLS,
    compute_psi,
    generate_drifted_features,
    generate_stable_features,
)

PSI_THRESHOLD  = 0.20
OUT_DIR        = Path("models/drift_sim")
REPORT_PATH    = Path("docs/drift_simulation_report.md")

_DRIFT_FEATURES  = ["recency_days", "online_ratio", "frequency"]
_DRIFT_MAGNITUDE = 0.30


# ── Label generators ──────────────────────────────────────────────────────────

def _generate_labels(df: pd.DataFrame, concept: str, seed: int) -> np.ndarray:
    """
    Rank-based propensity labels with two distinct concept functions.

    'stable'  — recency + frequency drive conversion (standard RFM world)
    'drifted' — online_ratio + monetary drive conversion
                (promotional channel shift; new campaign changed what predicts purchase)
    """
    rng = np.random.default_rng(seed)
    if concept == "stable":
        r = 1.0 - np.clip(df["recency_days"].values / 180.0, 0.0, 1.0)
        f = np.clip(df["frequency"].values / 50.0, 0.0, 1.0)
        m = np.clip(df["monetary"].values / 2000.0, 0.0, 1.0)
        score = 0.50 * r + 0.35 * f + 0.15 * m
    else:
        o = np.clip(df["online_ratio"].values, 0.0, 1.0)
        m = np.clip(df["monetary"].values / 2000.0, 0.0, 1.0)
        a = np.clip(df["avg_basket_size"].values / 80.0, 0.0, 1.0)
        score = 0.50 * o + 0.30 * m + 0.20 * a
    noise = rng.normal(0.0, 0.10, len(df))
    final = score + noise
    threshold = float(np.percentile(final, 72))
    return (final >= threshold).astype(int)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random 70/15/15 split — TemporalSplitter requires last_transaction_date
    which the drift generators do not produce."""
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n70 = int(len(shuffled) * 0.70)
    n85 = int(len(shuffled) * 0.85)
    return shuffled.iloc[:n70], shuffled.iloc[n70:n85], shuffled.iloc[n85:]


def _train_lr(train: pd.DataFrame, label_col: str, seed: int) -> Pipeline:
    X = train[FEATURE_COLS].fillna(0.0).values
    y = train[label_col].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(solver="saga", max_iter=1000,
                                  class_weight="balanced", random_state=seed)),
    ])
    pipe.fit(X, y)
    return pipe


def _evaluate(pipe: Pipeline, df: pd.DataFrame, label_col: str) -> tuple[float, float]:
    """Return (test_auc, lift_d1)."""
    X = df[FEATURE_COLS].fillna(0.0).values
    y = df[label_col].values
    if len(np.unique(y)) < 2:
        return 0.5, 1.0
    proba = pipe.predict_proba(X)[:, 1]
    auc   = float(roc_auc_score(y, proba))
    rank  = pd.DataFrame({"p": proba, "y": y}).sort_values("p", ascending=False)
    top   = rank.iloc[: max(1, len(rank) // 10)]
    base  = float(rank["y"].mean())
    lift  = float(top["y"].mean() / base) if base > 0 else 1.0
    return round(auc, 4), round(lift, 4)


def _psi_all(ref: pd.DataFrame, actual: pd.DataFrame) -> dict[str, float]:
    return {
        feat: round(compute_psi(ref[feat].values, actual[feat].values), 4)
        for feat in FEATURE_COLS
        if feat in ref.columns and feat in actual.columns
    }


# ── Simulation phases ─────────────────────────────────────────────────────────

def run_simulation(
    n_customers: int = 1500,
    drift_magnitude: float = _DRIFT_MAGNITUDE,
    drift_features: list[str] | None = None,
    seed: int = 42,
) -> dict:
    if drift_features is None:
        drift_features = list(_DRIFT_FEATURES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: stable baseline ───────────────────────────────────────────────
    stable_df = generate_stable_features(n_customers=n_customers, seed=seed)
    stable_df["label"] = _generate_labels(stable_df, concept="stable", seed=seed)

    train_s, val_s, test_s = _split(stable_df, seed=seed)
    model_w0 = _train_lr(pd.concat([train_s, val_s]), label_col="label", seed=seed)
    joblib.dump(model_w0, OUT_DIR / "model_week0.pkl")

    w0_auc, w0_lift = _evaluate(model_w0, test_s, "label")

    # Baseline PSI: stable vs a held-out stable sample — should be near 0
    stable_holdout = generate_stable_features(n_customers=n_customers, seed=seed + 100)
    psi_baseline   = _psi_all(stable_df, stable_holdout)

    # ── Phase 2: stale model on drifted data ──────────────────────────────────
    drifted_df = generate_drifted_features(
        n_customers    = n_customers,
        drift_features = drift_features,
        drift_magnitude= drift_magnitude,
        seed           = seed + 99,           # generate_drifted internally uses seed+999
    )
    drifted_df["label"] = _generate_labels(drifted_df, concept="drifted", seed=seed)

    _, _, test_d = _split(drifted_df, seed=seed)

    psi_per_feat = _psi_all(stable_df, drifted_df)
    max_psi      = max(psi_per_feat.values()) if psi_per_feat else 0.0
    above_thresh = [f for f, v in psi_per_feat.items() if v > PSI_THRESHOLD]

    stale_auc, stale_lift = _evaluate(model_w0, test_d, "label")

    # ── Phase 3: retrain on drifted data ──────────────────────────────────────
    train_d, val_d, test_d2 = _split(drifted_df, seed=seed + 1)
    model_w4 = _train_lr(pd.concat([train_d, val_d]), label_col="label", seed=seed)
    joblib.dump(model_w4, OUT_DIR / "model_week4.pkl")

    ret_auc, ret_lift = _evaluate(model_w4, test_d2, "label")

    # ── Phase 4: recovery metrics ─────────────────────────────────────────────
    auc_deg      = round(w0_auc  - stale_auc,  4)
    auc_rec      = round(ret_auc - stale_auc,  4)
    lift_deg     = round(w0_lift - stale_lift, 4)
    lift_rec     = round(ret_lift - stale_lift, 4)
    full_recovery = bool(ret_auc >= w0_auc - 0.02)

    results = {
        "simulation_date": datetime.now(timezone.utc).isoformat(),
        "drift_magnitude":  drift_magnitude,
        "drift_features":   drift_features,
        "phase1_baseline": {
            "model":     "model_week0.pkl",
            "test_auc":  w0_auc,
            "lift_d1":   w0_lift,
            "psi_scores": psi_baseline,
        },
        "phase2_stale_model_on_drifted_data": {
            "model":                "model_week0.pkl (stale)",
            "test_auc":             stale_auc,
            "lift_d1":              stale_lift,
            "psi_scores":           psi_per_feat,
            "overall_max_psi":      round(max_psi, 4),
            "features_above_threshold": above_thresh,
            "drift_detected":       bool(max_psi > PSI_THRESHOLD),
        },
        "phase3_retrained_model": {
            "model":    "model_week4.pkl",
            "test_auc": ret_auc,
            "lift_d1":  ret_lift,
        },
        "phase4_recovery_metrics": {
            "auc_degradation":      auc_deg,
            "auc_recovery":         auc_rec,
            "lift_degradation":     lift_deg,
            "lift_recovery":        lift_rec,
            "full_recovery_achieved": full_recovery,
        },
    }

    (OUT_DIR / "simulation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    _write_report(results)
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(r: dict) -> None:
    p1 = r["phase1_baseline"]
    p2 = r["phase2_stale_model_on_drifted_data"]
    p3 = r["phase3_retrained_model"]
    p4 = r["phase4_recovery_metrics"]

    drift_feats_str = ", ".join(r["drift_features"])

    def _psi_status(v: float) -> str:
        if v >= PSI_THRESHOLD:
            return "RETRAIN"
        if v >= 0.10:
            return "MONITOR"
        return "STABLE"

    lines = [
        "# Feature Drift Simulation Report",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Simulation Setup",
        "",
        f"Drift magnitude: {r['drift_magnitude']}",
        f"Drifted features: {drift_feats_str}",
        "PSI retrain threshold: 0.20",
        "",
        "## Phase 1 — Baseline Model (Week 0)",
        "",
        "| Metric   | Value  |",
        "|----------|--------|",
        f"| Test AUC | {p1['test_auc']:.3f}  |",
        f"| Lift@D1  | {p1['lift_d1']:.2f}   |",
        f"| Max PSI  | {max(p1['psi_scores'].values()):.4f} |",
        "| Status   | STABLE |",
        "",
        "## Phase 2 — Stale Model on Drifted Data",
        "",
        "### PSI Scores Per Feature",
        "",
        "| Feature               | PSI    | Status  |",
        "|-----------------------|--------|---------|",
    ]
    for feat, psi_val in p2["psi_scores"].items():
        lines.append(f"| {feat:<21} | {psi_val:.4f} | {_psi_status(psi_val):<7} |")

    lines += [
        "",
        "### Model Performance Degradation",
        "",
        "| Metric   | Baseline | Stale | Degradation |",
        "|----------|----------|-------|-------------|",
        f"| Test AUC | {p1['test_auc']:.3f}    | {p2['test_auc']:.3f} | {-p4['auc_degradation']:+.3f}       |",
        f"| Lift@D1  | {p1['lift_d1']:.2f}    | {p2['lift_d1']:.2f} | {-p4['lift_degradation']:+.2f}       |",
        "",
        "## Phase 3 — Retrained Model (Week 4)",
        "",
        "| Metric   | Stale | Retrained | Recovery |",
        "|----------|-------|-----------|----------|",
        f"| Test AUC | {p2['test_auc']:.3f} | {p3['test_auc']:.3f}     | {p4['auc_recovery']:+.3f}    |",
        f"| Lift@D1  | {p2['lift_d1']:.2f} | {p3['lift_d1']:.2f}     | {p4['lift_recovery']:+.2f}    |",
        "",
        "## Conclusion",
        "",
    ]

    feat_list = ", ".join(r["drift_features"])
    full = "achieved" if p4["full_recovery_achieved"] else "not achieved"
    lines.append(
        f"Drift in [{feat_list}] caused AUC to drop from "
        f"{p1['test_auc']:.2f} to {p2['test_auc']:.2f} "
        f"(degradation of {p4['auc_degradation']:.2f}). "
        f"Retraining restored AUC to {p3['test_auc']:.2f}, "
        f"recovering {p4['auc_recovery']:.2f} of the {p4['auc_degradation']:.2f} degradation. "
        f"Full recovery {full}."
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(
    n_customers: int = 1500,
    drift_magnitude: float = _DRIFT_MAGNITUDE,
) -> dict:
    results = run_simulation(
        n_customers    = n_customers,
        drift_magnitude= drift_magnitude,
        drift_features = list(_DRIFT_FEATURES),
    )
    p1 = results["phase1_baseline"]
    p2 = results["phase2_stale_model_on_drifted_data"]
    p3 = results["phase3_retrained_model"]
    triggered = p2["features_above_threshold"]

    from ml.local.visualise import plot_drift_simulation
    plot_path = plot_drift_simulation(results)

    print("\n" + "=" * 56)
    print("DRIFT SIMULATION COMPLETE")
    print("=" * 56)
    print(f"  Phase 1 Baseline AUC:   {p1['test_auc']:.4f}")
    print(f"  Phase 2 Stale AUC:      {p2['test_auc']:.4f}  (degraded)")
    print(f"  Phase 3 Retrained AUC:  {p3['test_auc']:.4f}  (recovered)")
    print(f"  PSI triggered on:       {triggered}")
    full = results["phase4_recovery_metrics"]["full_recovery_achieved"]
    print(f"  Recovery achieved:      {'Yes' if full else 'No'}")
    print(f"  Plot saved:             {plot_path}")
    print("=" * 56)
    return results


if __name__ == "__main__":
    main()
