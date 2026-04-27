"""
Unit tests for ml.local.visualise — all 14 plot functions.
"""
from __future__ import annotations

import os

import numpy as np
import optuna
import pandas as pd
import pytest

import ml.local.visualise as vis_mod
from ml.local.visualise import (
    plot_all_models_bias_variance,
    plot_calibration_curve,
    plot_learning_curves,
    plot_learning_curves_all_models,
    plot_lgbm_loss_curve,
    plot_lift_chart,
    plot_model_comparison,
    plot_oob_trajectory,
    plot_optuna_history,
    plot_overfitting_curve,
    plot_psi_heatmap,
    plot_segment_profiles,
    plot_shap_importance,
    plot_xgb_loss_curve,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _redirect_plot_dir(tmp_path, monkeypatch):
    """Redirect PLOT_DIR to a temp directory so tests don't write to docs/."""
    plot_dir = str(tmp_path / "plots")
    monkeypatch.setattr(vis_mod, "PLOT_DIR", plot_dir)
    return plot_dir


@pytest.fixture
def simple_study():
    """Optuna study with 5 trivial trials — fast to create."""
    study = optuna.create_study(direction="maximize")
    def objective(trial):
        return trial.suggest_float("x", 0.5, 1.0)
    study.optimize(objective, n_trials=5)
    return study


@pytest.fixture
def segment_df():
    return pd.DataFrame({
        "segment_id":     [0, 1, 2],
        "recency_days":   [10.0, 50.0, 120.0],
        "frequency":      [20.0, 8.0,  2.0],
        "monetary":       [500.0, 200.0, 40.0],
        "avg_basket_size":[30.0, 18.0, 10.0],
    })


# ── Test 1: all 10 functions return a filepath to an existing .png ─────────

class TestAllPlotsReturnFilepath:

    def test_plot_learning_curves(self):
        fp = plot_learning_curves(
            train_sizes=[100, 200, 300],
            train_scores=[[0.70, 0.72], [0.75, 0.77], [0.78, 0.80]],
            val_scores=  [[0.65, 0.67], [0.68, 0.70], [0.70, 0.72]],
            model_name="Logistic Regression",
        )
        assert os.path.exists(fp), f"File not found: {fp}"
        assert fp.endswith(".png")

    def test_plot_overfitting_curve(self):
        fp = plot_overfitting_curve(
            depths=[2, 4, 6, 8, 10],
            train_aucs=[0.70, 0.80, 0.90, 0.95, 0.98],
            val_aucs=  [0.68, 0.75, 0.73, 0.70, 0.67],
        )
        assert os.path.exists(fp)
        assert fp.endswith(".png")

    def test_plot_oob_trajectory(self):
        fp = plot_oob_trajectory(
            n_trees_list=[10, 50, 100, 150, 200],
            oob_scores=  [0.62, 0.70, 0.72, 0.73, 0.73],
        )
        assert os.path.exists(fp)
        assert fp.endswith(".png")

    def test_plot_optuna_history(self, simple_study):
        fp = plot_optuna_history(simple_study, "XGBoost")
        assert os.path.exists(fp)
        assert fp.endswith(".png")

    def test_plot_calibration_curve(self):
        rng = np.random.default_rng(0)
        n = 200
        y_true = ([0] * 100 + [1] * 100)
        y_prob_uncal = rng.uniform(0.1, 0.9, n).tolist()
        y_prob_cal   = np.clip(np.array(y_prob_uncal) * 0.9 + 0.05, 0, 1).tolist()
        fp = plot_calibration_curve(y_true, y_prob_uncal, y_prob_cal, "LightGBM")
        assert os.path.exists(fp)
        assert fp.endswith(".png")

    def test_plot_shap_importance(self):
        fp = plot_shap_importance(
            feature_names=["recency_days", "frequency", "monetary"],
            shap_values=  [0.45, -0.30, 0.18],
            model_name="LightGBM",
        )
        assert os.path.exists(fp)
        assert fp.endswith(".png")

    def test_plot_lift_chart(self):
        fp = plot_lift_chart(
            deciles=list(range(1, 11)),
            actual_rates=[0.80, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
            baseline_rate=0.30,
        )
        assert os.path.exists(fp)
        assert fp.endswith(".png")

    def test_plot_psi_heatmap(self):
        feature_names = ["recency_days", "online_ratio"]
        psi_scores = {"recency_days": 0.08, "online_ratio": 0.25}
        psi_history = [
            {"date": f"2026-0{i+1}-01", "recency_days": 0.02 * i, "online_ratio": 0.05 * i}
            for i in range(8)
        ]
        fp = plot_psi_heatmap(feature_names, psi_scores, psi_history)
        assert os.path.exists(fp)
        assert fp.endswith(".png")

    def test_plot_segment_profiles(self, segment_df):
        fp = plot_segment_profiles(segment_df)
        assert os.path.exists(fp)
        assert fp.endswith(".png")

    def test_plot_model_comparison(self):
        fp = plot_model_comparison(
            model_names=["logistic_regression", "decision_tree", "random_forest"],
            test_aucs=  [0.75, 0.72, 0.77],
            train_aucs= [0.78, 0.85, 0.82],
            selected_model="logistic_regression",
        )
        assert os.path.exists(fp)
        assert fp.endswith(".png")


# ── Test 2: plots are saved to the configured directory ──────────────────────

def test_plots_saved_to_correct_directory(tmp_path, monkeypatch):
    """docs/plots/ is created on first use even when the directory is missing."""
    new_dir = str(tmp_path / "fresh_plots_dir")
    assert not os.path.exists(new_dir), "Pre-condition: directory must not exist"

    monkeypatch.setattr(vis_mod, "PLOT_DIR", new_dir)
    fp = plot_overfitting_curve(
        depths=[2, 4, 6],
        train_aucs=[0.70, 0.80, 0.85],
        val_aucs=  [0.68, 0.74, 0.72],
    )

    assert os.path.isdir(new_dir), "Directory must be created by the plot function"
    assert os.path.exists(fp), "Plot file must exist"
    assert os.path.dirname(fp) == new_dir, "Plot must be saved in the configured PLOT_DIR"


# ── Tests 3-6: new loss-curve and bias-variance functions ────────────────────

def _fake_loss(n: int, start: float, fast_drop: float, overfit_after: int, slow_rise: float):
    """Produce realistic-looking train or val loss vectors."""
    return [max(0.01, start - i * fast_drop + max(0, (i - overfit_after) * slow_rise))
            for i in range(n)]


def test_xgb_loss_curve_saves_png():
    rounds = 25
    evals = {
        "train": {"logloss": _fake_loss(rounds, 0.70, 0.020, 99, 0.000)},
        "val":   {"logloss": _fake_loss(rounds, 0.72, 0.014, 12, 0.006)},
    }
    fp = plot_xgb_loss_curve(evals, "XGBoost")
    assert os.path.exists(fp)
    assert fp.endswith(".png")


def test_lgbm_loss_curve_saves_png():
    rounds = 25
    evals = {
        "train":   {"binary_logloss": _fake_loss(rounds, 0.68, 0.018, 99, 0.000)},
        "valid_1": {"binary_logloss": _fake_loss(rounds, 0.70, 0.012, 14, 0.005)},
    }
    fp = plot_lgbm_loss_curve(evals, "LightGBM")
    assert os.path.exists(fp)
    assert fp.endswith(".png")


def test_bias_variance_summary_saves_png():
    metrics = {
        "logistic_regression": {"train_auc": 0.78, "val_auc": 0.75, "test_auc": 0.76, "cv_std": 0.02},
        "decision_tree":       {"train_auc": 0.95, "val_auc": 0.70, "test_auc": 0.71, "cv_std": 0.05},
        "random_forest":       {"train_auc": 0.86, "val_auc": 0.77, "test_auc": 0.78, "cv_std": 0.03},
    }
    fp = plot_all_models_bias_variance(metrics, "logistic_regression")
    assert os.path.exists(fp)
    assert fp.endswith(".png")


def test_learning_curves_comparison_saves_png():
    results = {
        "logistic_regression": {
            "train_sizes":  [100, 200, 300],
            "train_scores": [[0.70, 0.72], [0.75, 0.77], [0.78, 0.80]],
            "val_scores":   [[0.65, 0.67], [0.68, 0.70], [0.70, 0.72]],
        },
        "random_forest": {
            "train_sizes":  [100, 200, 300],
            "train_scores": [[0.82, 0.84], [0.88, 0.90], [0.92, 0.94]],
            "val_scores":   [[0.68, 0.70], [0.73, 0.75], [0.75, 0.77]],
        },
    }
    fp = plot_learning_curves_all_models(results)
    assert os.path.exists(fp)
    assert fp.endswith(".png")
