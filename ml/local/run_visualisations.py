"""
Demo script: generates all 10 visualisations from the Tesco propensity pipeline.

Runs the full pipeline on a fast synthetic dataset (FAST_CONFIG) and saves
all plots to docs/plots/. Also produces docs/plots/README.md as a gallery.
"""
from __future__ import annotations

import datetime
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.local.generate_drift_data import (
    FEATURE_COLS as DRIFT_FEATURE_COLS,
    compute_psi,
    generate_drifted_features,
    generate_stable_features,
)
from ml.local.run_pipeline import (
    DEFAULT_CONFIG,
    FEATURE_COLS,
    engineer_features,
    evaluate_and_select,
    generate_data,
    run_segmentation,
    train_all_models,
)
from ml.local.visualise import (
    plot_calibration_curve,
    plot_learning_curves,
    plot_lift_chart,
    plot_model_comparison,
    plot_oob_trajectory,
    plot_optuna_history,
    plot_overfitting_curve,
    plot_psi_heatmap,
    plot_segment_profiles,
    plot_shap_importance,
)

FAST_CONFIG = {
    "n_customers":     1500,
    "n_transactions":  15000,
    "n_optuna_trials": 20,
    "seed":            42,
    "nonlinear":       True,
}

PLOT_DIR = "docs/plots"


# ─── Helper: build synthetic PSI history ─────────────────────────────────────

def _psi_history(feat_cols: list[str]) -> tuple[dict, list[dict]]:
    today = datetime.date.today()
    reference = generate_stable_features(n_customers=400, seed=42)
    history = []

    for week in range(8, 0, -1):
        date_str = (today - datetime.timedelta(weeks=week)).strftime("%Y-%m-%d")
        entry = {"date": date_str}

        if week >= 6:
            snapshot = generate_stable_features(n_customers=400, seed=100 + week)
        elif week >= 4:
            snapshot = generate_stable_features(n_customers=400, seed=100 + week)
            snapshot["recency_days"] = (snapshot["recency_days"] * 1.5).clip(0, 360)
            snapshot["online_ratio"] = (snapshot["online_ratio"] + 0.15).clip(0, 1)
        else:
            multiplier = 2.0 + (4 - week) * 0.5
            snapshot = generate_stable_features(n_customers=400, seed=100 + week)
            snapshot["recency_days"] = (snapshot["recency_days"] * multiplier).clip(0, None)
            snapshot["online_ratio"] = (snapshot["online_ratio"] + 0.30).clip(0, 1)

        for feat in feat_cols:
            if feat in reference.columns and feat in snapshot.columns:
                entry[feat] = compute_psi(reference[feat].values, snapshot[feat].values)
        history.append(entry)

    psi_scores = {f: float(np.mean([e.get(f, 0.0) for e in history])) for f in feat_cols}
    return psi_scores, history


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = {**DEFAULT_CONFIG, **FAST_CONFIG}
    os.makedirs(PLOT_DIR, exist_ok=True)
    saved_plots = {}

    print("=" * 58)
    print("TESCO VISUALISATION DEMO")
    print(f"  n_customers={cfg['n_customers']}, "
          f"n_optuna_trials={cfg['n_optuna_trials']}")
    print("=" * 58)

    # ── 1. Generate data ──────────────────────────────────────────────────────
    print("\n[1/6] Generating synthetic data ...")
    txns_df, customers_df = generate_data(cfg)
    txns_df["date"] = pd.to_datetime(txns_df["date"])

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("[2/6] Engineering features ...")
    train_df, val_df, test_df, full_features = engineer_features(txns_df, customers_df, cfg)
    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df["label"].values
    X_val   = val_df[FEATURE_COLS].fillna(0).values
    y_val   = val_df["label"].values
    X_test  = test_df[FEATURE_COLS].fillna(0).values
    y_test  = test_df["label"].values

    # ── 3. Segmentation ───────────────────────────────────────────────────────
    print("[3/6] Running segmentation ...")
    seg_result = run_segmentation(full_features, cfg)

    # ── 4. Model training ─────────────────────────────────────────────────────
    print("[4/6] Training 5 models (Optuna) ...")
    trained = train_all_models(
        X_train, y_train, X_val, y_val,
        n_trials=cfg["n_optuna_trials"], seed=cfg["seed"],
    )

    # ── 5. Evaluation & selection ─────────────────────────────────────────────
    print("[5/6] Evaluating & selecting model ...")
    selection, metrics = evaluate_and_select(
        trained, X_train, y_train, X_val, y_val, X_test, y_test
    )
    selected_name = selection["selected_model_name"]
    selected_obj  = selection["selected_model_object"]
    print(f"      Selected: {selected_name}")

    # ── 6. Visualisations ─────────────────────────────────────────────────────
    print("[6/6] Generating 10 plots ...")

    from lightgbm import LGBMClassifier

    def _to_df(X):
        return pd.DataFrame(X, columns=FEATURE_COLS) if isinstance(selected_obj, LGBMClassifier) else X

    # Plot 1 — Learning curves (LR pipeline, 5 training-size checkpoints)
    lr_pipe = Pipeline([("sc", StandardScaler()), ("cls", LogisticRegression(max_iter=500, random_state=42))])
    tr_sizes, tr_scores, vl_scores = learning_curve(
        lr_pipe, X_train, y_train,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=3, scoring="roc_auc", n_jobs=-1,
    )
    saved_plots["learning_curve"] = plot_learning_curves(
        tr_sizes.tolist(), tr_scores.tolist(), vl_scores.tolist(), "Logistic Regression"
    )
    print(f"  1. learning_curve -> {saved_plots['learning_curve']}")

    # Plot 2 — Decision tree overfitting curve
    depths, tr_aucs, vl_aucs = list(range(2, 13)), [], []
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_aucs.append(roc_auc_score(y_train, dt.predict_proba(X_train)[:, 1]))
        vl_aucs.append(roc_auc_score(y_val,   dt.predict_proba(X_val)[:, 1]))
    saved_plots["overfitting_curve"] = plot_overfitting_curve(depths, tr_aucs, vl_aucs)
    print(f"  2. overfitting_curve -> {saved_plots['overfitting_curve']}")

    # Plot 3 — Random forest OOB trajectory
    n_trees_range = list(range(10, 211, 20))
    oob_scores = []
    for n in n_trees_range:
        rf_oob = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
        rf_oob.fit(X_train, y_train)
        oob_scores.append(rf_oob.oob_score_)
    saved_plots["oob_trajectory"] = plot_oob_trajectory(n_trees_range, oob_scores)
    print(f"  3. oob_trajectory -> {saved_plots['oob_trajectory']}")

    # Plot 4 — Optuna history (small inline study on XGBoost)
    from xgboost import XGBClassifier

    optuna_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    def _xgb_obj(trial):
        m = XGBClassifier(
            learning_rate=trial.suggest_float("lr", 0.05, 0.3, log=True),
            n_estimators=trial.suggest_int("n", 30, 120),
            random_state=42, eval_metric="auc",
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    optuna_study.optimize(_xgb_obj, n_trials=10)
    saved_plots["optuna_history"] = plot_optuna_history(optuna_study, "XGBoost")
    print(f"  4. optuna_history -> {saved_plots['optuna_history']}")

    # Plot 5 — Calibration curve (selected model + isotonic calibration)
    from sklearn.isotonic import IsotonicRegression

    val_in    = _to_df(X_val)
    val_proba = selected_obj.predict_proba(val_in)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_proba, y_val)
    cal_proba = np.clip(iso.predict(val_proba), 0.0, 1.0)
    saved_plots["calibration_curve"] = plot_calibration_curve(
        y_val.tolist(), val_proba.tolist(), cal_proba.tolist(), selected_name
    )
    print(f"  5. calibration_curve -> {saved_plots['calibration_curve']}")

    # Plot 6 — SHAP importance (feature–prediction correlation as proxy)
    train_in     = _to_df(X_train)
    train_proba  = selected_obj.predict_proba(train_in)[:, 1]
    pred_series  = pd.Series(train_proba)
    shap_vals = np.array([
        float(pd.Series(X_train[:, i]).corr(pred_series))
        for i in range(len(FEATURE_COLS))
    ])
    shap_vals = np.nan_to_num(shap_vals, nan=0.0)
    saved_plots["shap_importance"] = plot_shap_importance(FEATURE_COLS, shap_vals.tolist(), selected_name)
    print(f"  6. shap_importance -> {saved_plots['shap_importance']}")

    # Plot 7 — Lift chart (selected model on test set)
    test_in    = _to_df(X_test)
    test_proba = selected_obj.predict_proba(test_in)[:, 1]
    df_lift = (
        pd.DataFrame({"score": test_proba, "label": y_test})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    n_test = len(df_lift)
    actual_rates = [
        float(df_lift["label"].iloc[i * n_test // 10 : (i + 1) * n_test // 10].mean())
        for i in range(10)
    ]
    saved_plots["lift_chart"] = plot_lift_chart(
        list(range(1, 11)), actual_rates, float(y_test.mean())
    )
    print(f"  7. lift_chart -> {saved_plots['lift_chart']}")

    # Plot 8 — PSI heatmap (synthetic drift history)
    psi_feat_cols = [c for c in DRIFT_FEATURE_COLS if c != "has_promoted_category"]
    psi_scores, psi_history = _psi_history(psi_feat_cols)
    saved_plots["psi_heatmap"] = plot_psi_heatmap(psi_feat_cols, psi_scores, psi_history)
    print(f"  8. psi_heatmap -> {saved_plots['psi_heatmap']}")

    # Plot 9 — Segment profiles
    full_labeled = full_features.copy()
    full_labeled["segment_id"] = seg_result["labels"]
    seg_means = full_labeled.groupby("segment_id")[FEATURE_COLS].mean().reset_index()
    saved_plots["segment_profiles"] = plot_segment_profiles(seg_means)
    print(f"  9. segment_profiles -> {saved_plots['segment_profiles']}")

    # Plot 10 — Model comparison
    model_names    = list(metrics.keys())
    test_aucs_list = [metrics[n]["test_auc"]  for n in model_names]
    train_aucs_list = [metrics[n]["train_auc"] for n in model_names]
    saved_plots["model_comparison"] = plot_model_comparison(
        model_names, test_aucs_list, train_aucs_list, selected_name
    )
    print(f"  10. model_comparison -> {saved_plots['model_comparison']}")

    # ── Gallery README ────────────────────────────────────────────────────────
    _write_gallery_readme(saved_plots)
    print(f"\nGallery: {PLOT_DIR}/README.md")
    print("Done.")


def _write_gallery_readme(saved_plots: dict[str, str]) -> None:
    labels = {
        "learning_curve":    "Learning Curves — Logistic Regression",
        "overfitting_curve": "Bias-Variance Tradeoff — Decision Tree",
        "oob_trajectory":    "OOB Score vs Tree Count — Random Forest",
        "optuna_history":    "Optuna Optimisation History — XGBoost",
        "calibration_curve": "Calibration Curve — Selected Model",
        "shap_importance":   "SHAP Feature Importance — Selected Model",
        "lift_chart":        "Realised Lift by Propensity Decile",
        "psi_heatmap":       "PSI Drift Heatmap — Feature Stability (8 weeks)",
        "segment_profiles":  "Customer Segment Profiles",
        "model_comparison":  "7-Model AUC Comparison",
    }
    lines = ["# Plot Gallery\n",
             "Generated by `ml/local/run_visualisations.py`.\n"]
    for key, title in labels.items():
        fp = saved_plots.get(key, "")
        if fp:
            fname = os.path.basename(fp)
            lines.append(f"## {title}\n\n![{title}]({fname})\n")
    readme_path = os.path.join(PLOT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
