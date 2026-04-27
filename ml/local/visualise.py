"""
Visualisation functions for the Tesco MLOps pipeline.
Each function saves a PNG to docs/plots/ and returns the absolute filepath.
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_DIR = "docs/plots"


def _save(filename: str) -> str:
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    return path


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


# ─── 1. Learning curves ──────────────────────────────────────────────────────

def plot_learning_curves(
    train_sizes: list[int],
    train_scores: list[list[float]],
    val_scores: list[list[float]],
    model_name: str,
) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    tr = np.array(train_scores, dtype=float)
    vl = np.array(val_scores, dtype=float)
    tr_mean, tr_std = tr.mean(axis=1), tr.std(axis=1)
    vl_mean, vl_std = vl.mean(axis=1), vl.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, tr_mean, "b-o", label="Train AUC")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2, color="blue")
    ax.plot(train_sizes, vl_mean, color="orange", marker="o", label="Val AUC")
    ax.fill_between(train_sizes, vl_mean - vl_std, vl_mean + vl_std, alpha=0.2, color="orange")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("AUC score")
    ax.set_title(f"Learning Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(f"learning_curve_{_slug(model_name)}.png")


# ─── 2. Overfitting (bias-variance) curve ────────────────────────────────────

def plot_overfitting_curve(
    depths: list[int],
    train_aucs: list[float],
    val_aucs: list[float],
) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    optimal_depth = depths[int(np.argmax(val_aucs))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, train_aucs, "b-o", label="Train AUC")
    ax.plot(depths, val_aucs, color="orange", marker="o", label="Val AUC")
    ax.axvline(x=optimal_depth, color="green", linestyle="--",
               label=f"Optimal depth = {optimal_depth}")
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("AUC")
    ax.set_title("Decision Tree — Bias-Variance Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save("overfitting_curve.png")


# ─── 3. OOB score trajectory ─────────────────────────────────────────────────

def plot_oob_trajectory(
    n_trees_list: list[int],
    oob_scores: list[float],
) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    oob_arr = np.array(oob_scores, dtype=float)

    # Plateau: first index where per-step gain is < 5% of total gain
    if len(oob_arr) > 2:
        total_gain = max(abs(float(oob_arr[-1]) - float(oob_arr[0])), 1e-6)
        improvements = np.diff(oob_arr)
        small = np.where(improvements < total_gain * 0.05)[0]
        plateau_idx = int(small[0]) + 1 if len(small) > 0 else len(oob_arr) - 1
    else:
        plateau_idx = 0

    plateau_n = n_trees_list[min(plateau_idx, len(n_trees_list) - 1)]
    plateau_score = float(oob_arr[min(plateau_idx, len(oob_arr) - 1)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_trees_list, oob_scores, "b-o", markersize=4)
    ax.annotate(
        f"Plateau ~{plateau_n} trees",
        xy=(plateau_n, plateau_score),
        xytext=(plateau_n, plateau_score + (oob_arr.max() - oob_arr.min()) * 0.15 + 0.01),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=9,
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB Score")
    ax.set_title("Random Forest — OOB Score vs Tree Count")
    ax.grid(True, alpha=0.3)
    return _save("oob_trajectory.png")


# ─── 4. Optuna history ───────────────────────────────────────────────────────

def plot_optuna_history(study: Any, model_name: str) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    trials = [t for t in study.trials if t.value is not None]
    nums   = [t.number for t in trials]
    vals   = [t.value  for t in trials]

    best, best_so_far = -float("inf"), []
    for v in vals:
        best = max(best, v)
        best_so_far.append(best)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(nums, vals, alpha=0.6, s=20, label="Trial AUC")
    ax.plot(nums, best_so_far, "r-", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("AUC value")
    ax.set_title(f"Optuna Optimisation — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(f"optuna_{_slug(model_name)}.png")


# ─── 5. Calibration curve ────────────────────────────────────────────────────

def plot_calibration_curve(
    y_true: list[int],
    y_prob_uncal: list[float],
    y_prob_cal: list[float],
    model_name: str,
) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    from sklearn.calibration import calibration_curve as sk_cal_curve

    n = len(y_true)
    n_bins = max(3, min(10, n // 20))

    fpu, mpu = sk_cal_curve(y_true, y_prob_uncal, n_bins=n_bins, strategy="uniform")
    fpc, mpc = sk_cal_curve(y_true, y_prob_cal,   n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(mpu, fpu, "r-o", label="Uncalibrated")
    ax.plot(mpc, fpc, "g-o", label="Calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(f"calibration_{_slug(model_name)}.png")


# ─── 6. SHAP importance ──────────────────────────────────────────────────────

def plot_shap_importance(
    feature_names: list[str],
    shap_values: list[float],
    model_name: str,
) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    arr = np.array(shap_values, dtype=float)
    order = np.argsort(np.abs(arr))
    names  = [feature_names[i] for i in order]
    vals   = arr[order]
    colors = ["blue" if v >= 0 else "red" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.5)))
    ax.barh(names, vals, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean SHAP value")
    ax.set_title(f"SHAP Feature Importance — {model_name}")
    ax.grid(True, alpha=0.3, axis="x")
    return _save(f"shap_{_slug(model_name)}.png")


# ─── 7. Lift chart ───────────────────────────────────────────────────────────

def plot_lift_chart(
    deciles: list[int],
    actual_rates: list[float],
    baseline_rate: float,
) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    lifts = [r / baseline_rate if baseline_rate > 0 else 0.0 for r in actual_rates]
    colors = ["green" if l >= 2.0 else ("orange" if l >= 1.0 else "red") for l in lifts]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(deciles, lifts, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=1.0, color="gray",  linestyle="--", linewidth=1.5, label="Random baseline (1.0)")
    ax.axhline(y=2.5, color="navy",  linestyle="--", linewidth=1.5, label="Gate threshold (2.5)")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Lift  (actual rate / baseline rate)")
    ax.set_title("Realised Lift by Propensity Decile")
    ax.set_xticks(deciles)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return _save("lift_chart.png")


# ─── 8. PSI heatmap ──────────────────────────────────────────────────────────

def plot_psi_heatmap(
    feature_names: list[str],
    psi_scores: dict[str, float],
    psi_history: list[dict],
) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    recent = psi_history[-8:] if len(psi_history) > 8 else psi_history
    dates  = [str(e.get("date", i)) for i, e in enumerate(recent)]
    data   = np.array([
        [float(e.get(f, 0.0)) for e in recent]
        for f in feature_names
    ])

    cmap = mcolors.LinearSegmentedColormap.from_list("psi", [(0, "green"), (0.5, "yellow"), (1, "red")])
    fig, ax = plt.subplots(figsize=(max(8, len(dates)), max(4, len(feature_names) * 0.6)))
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=0.40, aspect="auto")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    plt.colorbar(im, ax=ax, label="PSI score")
    ax.set_title("PSI Drift Heatmap — Feature Stability")
    plt.tight_layout()
    return _save("psi_heatmap.png")


# ─── 9. Segment profiles ─────────────────────────────────────────────────────

def plot_segment_profiles(segment_stats: pd.DataFrame) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    feat_cols = [c for c in segment_stats.columns if c != "segment_id"]
    df = segment_stats[feat_cols].copy().astype(float)

    # Normalise each feature to [0, 1]
    for col in feat_cols:
        lo, hi = df[col].min(), df[col].max()
        df[col] = (df[col] - lo) / (hi - lo) if hi > lo else 0.5

    n_seg, n_feat = len(df), len(feat_cols)
    x = np.arange(n_feat)
    width = 0.8 / max(n_seg, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_feat * 1.2), 5))
    for i, (_, row) in enumerate(df.iterrows()):
        offset = (i - n_seg / 2 + 0.5) * width
        ax.bar(x + offset, row.values, width=width * 0.9, label=f"Segment {i}")

    ax.set_xticks(x)
    ax.set_xticklabels(feat_cols, rotation=45, ha="right")
    ax.set_ylabel("Normalised value (0–1)")
    ax.set_title("Customer Segment Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return _save("segment_profiles.png")


# ─── 10. Model comparison ────────────────────────────────────────────────────

def plot_model_comparison(
    model_names: list[str],
    test_aucs: list[float],
    train_aucs: list[float],
    selected_model: str,
) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    from ml.config.thresholds import SELECTION_THRESHOLDS

    n = len(model_names)
    y = np.arange(n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.9)))
    bars_test  = ax.barh(y - height / 2, test_aucs,  height, label="Test AUC",  color="steelblue")
    bars_train = ax.barh(y + height / 2, train_aucs, height, label="Train AUC", color="lightsteelblue")

    for i, name in enumerate(model_names):
        if name == selected_model:
            bars_test[i].set_color("gold")
            bars_train[i].set_color("khaki")

    # Vertical line at LR baseline + gate threshold
    lr_auc = next(
        (auc for name, auc in zip(model_names, test_aucs) if "logistic" in name.lower()),
        None,
    )
    if lr_auc is not None:
        gate_line = lr_auc + SELECTION_THRESHOLDS["baseline_gate_min_gain"]
        ax.axvline(x=gate_line, color="red", linestyle="--", linewidth=1.5,
                   label=f"LR + {SELECTION_THRESHOLDS['baseline_gate_min_gain']:.2f} gate")

    ax.set_yticks(y)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("AUC score")
    ax.set_title("7-Model Comparison — AUC Scores")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _save("model_comparison.png")
