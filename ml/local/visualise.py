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

# ─── 11. XGBoost boosting-round loss curve ───────────────────────────────────

def plot_xgb_loss_curve(evals_result: dict, model_name: str = "XGBoost") -> str:
    """
    evals_result format:
      {"train": {"logloss": [...]}, "val": {"logloss": [...]}}
    Keys are positional (train = index 0, val = index 1).
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    keys = list(evals_result.keys())
    train_key, val_key = keys[0], keys[1] if len(keys) > 1 else keys[0]
    metric_key = list(evals_result[train_key].keys())[0]

    train_loss = evals_result[train_key][metric_key]
    val_loss   = evals_result[val_key][metric_key]
    rounds     = list(range(len(train_loss)))
    best_round = int(np.argmin(val_loss))
    best_val   = val_loss[best_round]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, train_loss, "b-",  linewidth=1.5, label="Train Loss")
    ax.plot(rounds, val_loss,   color="orange", linewidth=1.5, label="Val Loss")
    ax.axvline(x=best_round, color="red", linestyle="--", linewidth=1.5,
               label=f"Best: round {best_round}, loss {best_val:.4f}")
    if best_round < len(rounds) - 1:
        ax.axvspan(best_round, rounds[-1], alpha=0.08, color="red", label="Overfitting zone")
    ax.annotate(
        f"Best: round {best_round}\nloss {best_val:.4f}",
        xy=(best_round, best_val),
        xytext=(best_round + max(1, len(rounds) // 10), best_val + (max(val_loss) - min(val_loss)) * 0.15),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red", fontsize=9,
    )
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(f"{model_name} — Train vs Val Loss ({metric_key} per Boosting Round)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save("loss_curve_xgboost.png")


# ─── 12. LightGBM boosting-round loss curve ──────────────────────────────────

def plot_lgbm_loss_curve(evals_result: dict, model_name: str = "LightGBM") -> str:
    """
    evals_result format:
      {"train": {"binary_logloss": [...]}, "val": {"binary_logloss": [...]}}
    Keys are positional (train = index 0, val = index 1).
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    keys = list(evals_result.keys())
    train_key, val_key = keys[0], keys[1] if len(keys) > 1 else keys[0]
    metric_key = list(evals_result[train_key].keys())[0]

    train_loss = evals_result[train_key][metric_key]
    val_loss   = evals_result[val_key][metric_key]
    rounds     = list(range(len(train_loss)))
    best_round = int(np.argmin(val_loss))
    best_val   = val_loss[best_round]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, train_loss, "b-",  linewidth=1.5, label="Train Loss")
    ax.plot(rounds, val_loss,   color="orange", linewidth=1.5, label="Val Loss")
    ax.axvline(x=best_round, color="red", linestyle="--", linewidth=1.5,
               label=f"Best: round {best_round}, loss {best_val:.4f}")
    if best_round < len(rounds) - 1:
        ax.axvspan(best_round, rounds[-1], alpha=0.08, color="red", label="Overfitting zone")
    ax.annotate(
        f"Best: round {best_round}\nloss {best_val:.4f}",
        xy=(best_round, best_val),
        xytext=(best_round + max(1, len(rounds) // 10), best_val + (max(val_loss) - min(val_loss)) * 0.15),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red", fontsize=9,
    )
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(f"{model_name} — Train vs Val Loss ({metric_key} per Boosting Round)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save("loss_curve_lightgbm.png")


# ─── 13. All-model bias-variance diagnosis ────────────────────────────────────

def plot_all_models_bias_variance(
    metrics: dict[str, dict],
    selected_model: str,
) -> str:
    """
    metrics format per model:
      {"train_auc": float, "val_auc": float, "test_auc": float, "cv_std": float}
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    def _diagnose(train_auc: float, test_auc: float, val_auc: float) -> str:
        gap = train_auc - test_auc
        if gap > 0.08:
            return "HIGH VARIANCE (overfit)"
        if test_auc < 0.70:
            return "HIGH BIAS (underfit)"
        if gap < 0.02:
            return "WELL BALANCED"
        return "MODERATE VARIANCE"

    model_names = list(metrics.keys())
    n = len(model_names)
    y = np.arange(n)
    h = 0.25

    train_aucs = [metrics[m].get("train_auc", 0.0) for m in model_names]
    val_aucs   = [metrics[m].get("val_auc",   0.0) for m in model_names]
    test_aucs  = [metrics[m].get("test_auc",  0.0) for m in model_names]

    fig, ax = plt.subplots(figsize=(12, max(5, n * 0.9)))

    # Base colours — gold for selected model, muted otherwise
    def _bar_color(i: int, base: str) -> str:
        if model_names[i] == selected_model:
            return "gold" if base == "train" else ("orange" if base == "val" else "goldenrod")
        return {"train": "steelblue", "val": "sandybrown", "test": "mediumseagreen"}[base]

    for i in range(n):
        ax.barh(y[i] + h, train_aucs[i], h * 0.9, color=_bar_color(i, "train"), label="Train AUC" if i == 0 else "")
        ax.barh(y[i],     val_aucs[i],   h * 0.9, color=_bar_color(i, "val"),   label="Val AUC"   if i == 0 else "")
        ax.barh(y[i] - h, test_aucs[i],  h * 0.9, color=_bar_color(i, "test"),  label="Test AUC"  if i == 0 else "")
        diag = _diagnose(train_aucs[i], test_aucs[i], val_aucs[i])
        ax.text(max(train_aucs[i], val_aucs[i], test_aucs[i]) + 0.005, y[i],
                diag, va="center", fontsize=8, color="dimgray")

    ax.axvline(x=0.70, color="red", linestyle="--", linewidth=1.2, label="AUC threshold (0.70)")
    ax.set_yticks(y)
    ax.set_yticklabels(model_names)
    ax.set_xlabel("AUC score")
    ax.set_title("All Models — Bias-Variance Diagnosis (Train / Val / Test AUC)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _save("bias_variance_summary.png")


# ─── 14. Learning curves — multi-model comparison ────────────────────────────

def plot_learning_curves_all_models(results: dict[str, dict]) -> str:
    """
    results format per model:
      {"train_sizes": [...], "train_scores": [[...]], "val_scores": [[...]]}
    Each row of train_scores / val_scores is one CV fold.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    model_names = list(results.keys())
    n_models = len(model_names)

    all_means = []
    for name in model_names:
        tr = np.array(results[name]["train_scores"])
        vl = np.array(results[name]["val_scores"])
        all_means.extend(tr.mean(axis=1).tolist())
        all_means.extend(vl.mean(axis=1).tolist())
    y_min = max(0.0, min(all_means) - 0.05)
    y_max = min(1.0, max(all_means) + 0.05)

    fig, axes = plt.subplots(n_models, 1, figsize=(9, 4 * n_models), sharex=False)
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Learning Curves — Bias-Variance Comparison Across Models", fontsize=13, y=1.01)

    for ax, name in zip(axes, model_names):
        sizes      = results[name]["train_sizes"]
        tr_arr     = np.array(results[name]["train_scores"])
        vl_arr     = np.array(results[name]["val_scores"])
        tr_mean, tr_std = tr_arr.mean(axis=1), tr_arr.std(axis=1)
        vl_mean, vl_std = vl_arr.mean(axis=1), vl_arr.std(axis=1)

        ax.plot(sizes, tr_mean, "b-o", label="Train AUC")
        ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="blue")
        ax.plot(sizes, vl_mean, color="orange", marker="o", label="Val AUC")
        ax.fill_between(sizes, vl_mean - vl_std, vl_mean + vl_std, alpha=0.15, color="orange")

        # Diagnosis at max training size
        gap = float(tr_mean[-1] - vl_mean[-1])
        if vl_mean[-1] < 0.70 and tr_mean[-1] < 0.70:
            diag = "Both low — underfitting"
        elif gap > 0.08:
            diag = "Diverging — overfitting"
        elif gap < 0.03:
            diag = "Converged — well fitted"
        else:
            diag = f"Gap {gap:.2f} — moderate variance"

        ax.text(0.97, 0.05, diag, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Training set size")
        ax.set_ylabel("AUC")
        ax.set_title(name.replace("_", " ").title())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return _save("learning_curves_comparison.png")


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
