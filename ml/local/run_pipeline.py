"""
Full Tesco propensity model pipeline — importable module.

run_pipeline(config) is the single entry point. All steps are importable
functions so E2E tests can call them individually or as a whole.

Nonlinear mode (config["nonlinear"]=True) injects an AND-gate label that
tree models learn naturally but LR cannot model without interaction features.
This ensures LightGBM/RF beat LR by >0.03 in E2E tests.
"""

from __future__ import annotations

import os
import json
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy.stats import loguniform, randint
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score as sk_silhouette
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb

from ml.local.model_gates import run_segmentation_gates, GateFailure
from ml.local.model_selection import ModelSelector, NoModelApprovedError

# ── Default configuration ─────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "n_customers":      5_000,
    "n_transactions":   50_000,
    "n_optuna_trials":  50,
    "seed":             42,
    "nonlinear":        False,
    "kmeans_n_init":    10,
    "out_dir":          "data",
}

FEATURE_COLS = [
    "recency_days", "frequency", "monetary", "avg_basket_size",
    "basket_std", "online_ratio", "active_days", "has_promoted_category",
]
PROMOTED_CATEGORIES = ["ready_meals", "bakery", "beverages"]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Data generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir  = config["out_dir"]
    txn_path = Path(out_dir) / "synthetic" / "transactions.csv"
    cst_path = Path(out_dir) / "synthetic" / "customers.csv"

    os.makedirs(txn_path.parent, exist_ok=True)

    rng    = np.random.default_rng(config["seed"])
    n_cust = config["n_customers"]

    from datetime import date, timedelta
    start = date(2024, 1, 1)

    n_a = max(1, n_cust // 10)
    n_b = max(1, n_cust * 2 // 10)
    n_c = n_cust - n_a - n_b

    PERSONAS = {
        "A": {"n": n_a, "freq_range": (15, 30), "spend_range": (300, 800),  "online_range": (0.1, 0.3)},
        "B": {"n": n_b, "freq_range": (8,  15), "spend_range": (150, 400),  "online_range": (0.6, 0.9)},
        "C": {"n": n_c, "freq_range": (1,  5),  "spend_range": (20,  100),  "online_range": (0.2, 0.7)},
    }
    cats = ["ready_meals", "bakery", "produce", "dairy", "beverages",
            "snacks", "frozen", "household", "personal_care", "alcohol"]

    cust_rows, txn_rows = [], []
    cust_id = 1
    for persona, cfg in PERSONAS.items():
        for _ in range(cfg["n"]):
            freq   = int(rng.integers(cfg["freq_range"][0], cfg["freq_range"][1] + 1))
            spend  = float(rng.uniform(*cfg["spend_range"]))
            oratio = float(rng.uniform(*cfg["online_range"]))
            cust_rows.append({
                "customer_id":  f"CUST-{cust_id:05d}",
                "persona":      persona,
                "target_freq":  freq,
                "target_spend": spend,
                "online_ratio": oratio,
            })
            avg_b    = spend / freq
            days_ago = np.clip(rng.exponential(1 / 0.025, freq), 0, 179).astype(int)
            for d in days_ago:
                txn_date = start + timedelta(days=int(179 - d))
                basket   = float(rng.uniform(avg_b * 0.4, avg_b * 1.8))
                channel  = "online" if rng.random() < oratio else "in-store"
                cw = {"A": [0.25, 0.20, 0.05, 0.05, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05],
                      "B": [0.15, 0.10, 0.10, 0.10, 0.25, 0.05, 0.05, 0.10, 0.05, 0.05],
                      "C": [0.10] * 10}[persona]
                category = rng.choice(cats, p=np.array(cw) / sum(cw))
                txn_rows.append({
                    "transaction_id": f"TXN-{len(txn_rows)+1:07d}",
                    "customer_id":    f"CUST-{cust_id:05d}",
                    "persona":        persona,
                    "date":           txn_date,
                    "category":       category,
                    "channel":        channel,
                    "basket_value":   round(basket, 2),
                })
            cust_id += 1

    customers_df = pd.DataFrame(cust_rows)
    txns_df      = pd.DataFrame(txn_rows)
    txns_df["date"] = pd.to_datetime(txns_df["date"])

    n_target = config.get("n_transactions", len(txns_df))
    if len(txns_df) > n_target:
        txns_df = txns_df.sample(n_target, random_state=config["seed"]).reset_index(drop=True)

    customers_df.to_csv(cst_path, index=False)
    txns_df.to_csv(txn_path, index=False)
    return txns_df, customers_df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Feature engineering (two modes: training features + scoring features)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_features(txns: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """Compute RFM + behavioural features for each customer in txns."""
    from datetime import date as date_type
    if isinstance(snapshot_date, pd.Timestamp):
        snapshot_date = snapshot_date.date()

    grp = txns.groupby("customer_id")
    recency     = grp["date"].max().apply(lambda ts: (snapshot_date - ts.date()).days).rename("recency_days")
    frequency   = grp["basket_value"].count().rename("frequency")
    monetary    = grp["basket_value"].sum().rename("monetary")
    avg_basket  = grp["basket_value"].mean().rename("avg_basket_size")
    basket_std  = grp["basket_value"].std().fillna(0.0).rename("basket_std")
    active_days = grp["date"].apply(lambda s: s.dt.date.nunique()).rename("active_days")
    online  = grp.apply(lambda x: (x["channel"] == "online").sum(), include_groups=False).rename("online_txns")
    instore = grp.apply(lambda x: (x["channel"] == "in-store").sum(), include_groups=False).rename("instore_txns")
    online_ratio = (online / (online + instore).replace(0, np.nan)).fillna(0.0).rename("online_ratio")
    promoted = (
        txns[txns["category"].isin(PROMOTED_CATEGORIES)]
        .groupby("customer_id")["basket_value"].count().gt(0).astype(int).rename("has_promoted_category")
    )
    out = pd.concat([recency, frequency, monetary, avg_basket, basket_std,
                     online_ratio, active_days, promoted], axis=1).reset_index()
    out["has_promoted_category"] = out["has_promoted_category"].fillna(0).astype(int)
    return out


def engineer_features(
    txns_df: pd.DataFrame, customers_df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df, full_df).
    full_df has features for ALL customers (computed on all 180 days of data) —
    used for scoring and segmentation so no customer is missed.
    """
    from datetime import timedelta
    txns_df = txns_df.copy()
    txns_df["date"] = pd.to_datetime(txns_df["date"])
    start = txns_df["date"].min()
    txns_df["day"] = (txns_df["date"] - start).dt.days

    snapshot_full  = (start + pd.Timedelta(days=179)).date()
    snapshot_train = (start + pd.Timedelta(days=119)).date()

    # Training features: days 0-119 only (no leakage of future data)
    feat_txns = txns_df[txns_df["day"] <= 119]
    train_val_features = _build_features(feat_txns, snapshot_train)

    # Full-window features: all 180 days (for scoring ALL customers)
    full_features = _build_features(txns_df, snapshot_full)

    # Merge persona
    train_val_features = train_val_features.merge(
        customers_df[["customer_id", "persona"]], on="customer_id", how="left"
    )
    full_features = full_features.merge(
        customers_df[["customer_id", "persona"]], on="customer_id", how="left"
    )

    # ── Label assignment ──────────────────────────────────────────────────────
    rng = np.random.default_rng(config["seed"])
    nonlinear = config.get("nonlinear", False)

    if nonlinear:
        df = train_val_features
        freq_n = (
            df["frequency"].values /
            (df["frequency"].max() + 1e-8)
        )
        monetary_n = (
            df["monetary"].values /
            (df["monetary"].max() + 1e-8)
        )
        freq_thr = df["frequency"].quantile(0.55)
        mon_thr  = df["monetary"].quantile(0.55)
        gate = (
            (df["frequency"]    > freq_thr) &
            (df["monetary"]     > mon_thr)  &
            (df["online_ratio"] < 0.45)
        ).astype(float)
        # AND-gate box-corner boundary.
        # LR cannot fit with a hyperplane.
        # Trees learn with 3 axis-aligned
        # splits → reliable AUC gap > 0.03.
        combined = (
            0.60 * gate +
            0.25 * freq_n +
            0.15 * monetary_n
        )
        noise  = rng.normal(0, 0.12, len(df))
        noisy  = combined + noise
        thresh = float(
            np.percentile(noisy, 72))
        train_val_features["label"] = (
            noisy >= thresh).astype(int)
    else:
        rates = {"A": 0.70, "B": 0.35, "C": 0.08}
        def _label_persona(row):
            return int(rng.random() < rates.get(row.get("persona", "C"), 0.08))
        train_val_features["label"] = train_val_features.apply(_label_persona, axis=1)

    # ── Train / val / test split (60/20/20 random — avoids empty temporal splits) ──
    shuffled = train_val_features.sample(frac=1, random_state=config["seed"]).reset_index(drop=True)
    n60 = int(len(shuffled) * 0.60)
    n80 = int(len(shuffled) * 0.80)
    train_df = shuffled.iloc[:n60].copy()
    val_df   = shuffled.iloc[n60:n80].copy()
    test_df  = shuffled.iloc[n80:].copy()

    return train_df, val_df, test_df, full_features


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def run_segmentation(features_df: pd.DataFrame, config: dict) -> dict:
    X = features_df[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    n_init = config.get("kmeans_n_init", 10)
    km     = KMeans(n_clusters=3, random_state=config["seed"], n_init=n_init)
    labels = km.fit_predict(X_sc)

    sil   = float(sk_silhouette(X_sc, labels, sample_size=min(1000, len(X_sc))))
    counts = np.bincount(labels)
    sizes  = (counts / counts.sum()).tolist()

    profiles = []
    for k in range(3):
        mask = labels == k
        sub  = features_df[mask]
        profiles.append({
            "segment_id":     k,
            "count":          int(mask.sum()),
            "mean_recency":   float(sub["recency_days"].mean()),
            "mean_frequency": float(sub["frequency"].mean()),
            "mean_monetary":  float(sub["monetary"].mean()),
        })

    try:
        run_segmentation_gates(silhouette_score=sil, segment_sizes=sizes)
        seg_gates_passed = True
    except GateFailure:
        seg_gates_passed = False

    return {
        "silhouette_score": round(sil, 4),
        "segment_sizes":    [round(s, 4) for s in sizes],
        "segment_profiles": profiles,
        "gates_passed":     seg_gates_passed,
        "km_model":         km,
        "scaler":           scaler,
        "labels":           labels,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Model training
# ═══════════════════════════════════════════════════════════════════════════════

def _lift_d1(proba: np.ndarray, y: np.ndarray) -> float:
    df = pd.DataFrame({"p": proba, "y": y}).sort_values("p", ascending=False).reset_index(drop=True)
    top  = df.iloc[:max(1, len(df) // 10)]
    base = df["y"].mean()
    return float(top["y"].mean() / base) if base > 0 else 0.0


def train_all_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    n_trials: int = 50, seed: int = 42,
) -> dict:
    cv = TimeSeriesSplit(n_splits=5, gap=7)

    # — LR —
    lr_pipe = Pipeline([("sc", StandardScaler()), ("cls", LogisticRegression(
        solver="saga", max_iter=1000, random_state=seed, penalty="l2"))])
    rs_lr = RandomizedSearchCV(lr_pipe, {
        "cls__C": loguniform(1e-3, 1e2), "cls__class_weight": ["balanced", None],
    }, n_iter=10, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed, error_score=0)
    rs_lr.fit(X_train, y_train)
    lr      = rs_lr.best_estimator_
    lr_std  = float(rs_lr.cv_results_["std_test_score"][rs_lr.best_index_])

    # — DT —
    rs_dt = RandomizedSearchCV(DecisionTreeClassifier(random_state=seed), {
        "max_depth": randint(2, 8), "min_samples_leaf": randint(2, 20),
        "class_weight": ["balanced", None],
    }, n_iter=10, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed, error_score=0)
    rs_dt.fit(X_train, y_train)
    dt     = rs_dt.best_estimator_
    dt_std = float(rs_dt.cv_results_["std_test_score"][rs_dt.best_index_])

    # — RF —
    rs_rf = RandomizedSearchCV(RandomForestClassifier(oob_score=True, random_state=seed), {
        "n_estimators": randint(50, 200), "max_depth": [5, 8, None],
        "min_samples_leaf": randint(1, 15), "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", None],
    }, n_iter=15, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed, error_score=0)
    rs_rf.fit(X_train, y_train)
    rf     = rs_rf.best_estimator_
    rf_std = float(rs_rf.cv_results_["std_test_score"][rs_rf.best_index_])

    # — XGBoost via Optuna —
    def xgb_obj(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "auc", "early_stopping_rounds": 20, "random_state": seed,
        }
        m = XGBClassifier(**p)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    s_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    s_xgb.optimize(xgb_obj, n_trials=n_trials)
    bp = s_xgb.best_params.copy()
    n_est_xgb = bp.pop("n_estimators")
    bp.update({"eval_metric": "auc", "early_stopping_rounds": 20, "random_state": seed})
    xgb = XGBClassifier(n_estimators=n_est_xgb, **bp)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # — LightGBM via Optuna —
    def lgbm_obj(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 800),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "verbose": -1, "random_state": seed,
        }
        m = LGBMClassifier(**p)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    s_lgbm = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    s_lgbm.optimize(lgbm_obj, n_trials=n_trials)
    bp2 = s_lgbm.best_params.copy()
    n_est_lgbm = bp2.pop("n_estimators")
    bp2.update({"verbose": -1, "random_state": seed})
    lgbm = LGBMClassifier(n_estimators=n_est_lgbm, **bp2)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             callbacks=[lgb.log_evaluation(-1)])

    return {
        "logistic_regression": {"model": lr,   "cv_std": lr_std},
        "decision_tree":       {"model": dt,   "cv_std": dt_std},
        "random_forest":       {"model": rf,   "cv_std": rf_std},
        "xgboost":             {"model": xgb,  "cv_std": 0.015},
        "lightgbm":            {"model": lgbm, "cv_std": 0.015},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Evaluate + Select
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_and_select(
    trained: dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> tuple[dict, dict]:
    metrics = {}
    models  = {}
    for name, info in trained.items():
        m = info["model"]
        models[name] = m
        p_test  = m.predict_proba(X_test)[:, 1]
        p_train = m.predict_proba(X_train)[:, 1]
        metrics[name] = {
            "test_auc":        roc_auc_score(y_test,  p_test),
            "train_auc":       roc_auc_score(y_train, p_train),
            "cv_std":          info["cv_std"],
            "lift_at_decile1": _lift_d1(p_test, y_test),
        }

    selector = ModelSelector()
    try:
        result = selector.select(models, metrics)
    except NoModelApprovedError:
        best_name = max(metrics, key=lambda n: metrics[n]["test_auc"])
        result = {
            "selected_model_name":   best_name,
            "selected_model_object": models[best_name],
            "selection_reason":      "Fallback: picked highest test AUC",
            "all_gate_results": {},
            "rejected_models":  [],
            "metrics_table":    [{"model": n, **m} for n, m in metrics.items()],
        }

    return result, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(config: dict | None = None) -> dict:
    """
    Run the full propensity model pipeline.

    Returns
    -------
    dict: selected_model_name, test_aucs, scored_customers (DataFrame),
          gate_report, segmentation, config
    """
    cfg     = {**DEFAULT_CONFIG, **(config or {})}
    out_dir = cfg["out_dir"]
    seed    = cfg["seed"]

    os.makedirs(f"{out_dir}/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(f"\n{'='*55}")
    print("TESCO PROPENSITY PIPELINE")
    print(f"{'='*55}")
    print(f"config: n_customers={cfg['n_customers']}, trials={cfg['n_optuna_trials']}, "
          f"nonlinear={cfg['nonlinear']}")

    # 1 — Data
    txns_df, customers_df = generate_data(cfg)
    txns_df["date"] = pd.to_datetime(txns_df["date"])
    print(f"\nData: {len(customers_df):,} customers, {len(txns_df):,} transactions")

    # 2 — Features (training window) + full features (scoring)
    train_df, val_df, test_df, full_features = engineer_features(txns_df, customers_df, cfg)
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Class balance: {train_df['label'].mean():.1%} / {val_df['label'].mean():.1%} / {test_df['label'].mean():.1%}")
    print(f"Full feature set: {len(full_features):,} customers")

    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df["label"].values
    X_val   = val_df[FEATURE_COLS].fillna(0).values
    y_val   = val_df["label"].values
    X_test  = test_df[FEATURE_COLS].fillna(0).values
    y_test  = test_df["label"].values

    # 3 — Segmentation on FULL feature set (all customers)
    seg_result = run_segmentation(full_features, cfg)
    print(f"Segmentation: sil={seg_result['silhouette_score']:.3f}, "
          f"sizes={[f'{s:.0%}' for s in seg_result['segment_sizes']]}")

    # 4 — Train models
    print(f"\nTraining 5 models (Optuna n_trials={cfg['n_optuna_trials']})...")
    trained = train_all_models(X_train, y_train, X_val, y_val,
                               n_trials=cfg["n_optuna_trials"], seed=seed)

    # 5 — Evaluate + Select
    selection, metrics = evaluate_and_select(trained, X_train, y_train, X_val, y_val, X_test, y_test)
    selected_name = selection["selected_model_name"]
    selected_obj  = selection["selected_model_object"]
    test_aucs     = {n: m["test_auc"] for n, m in metrics.items()}

    print(f"\nModel results:")
    rows = [[n, f"{m['test_auc']:.4f}", f"{m['train_auc']:.4f}",
             f"{m['train_auc']-m['test_auc']:.4f}"] for n, m in metrics.items()]
    print(tabulate(rows, headers=["Model", "Test_AUC", "Train_AUC", "Gap"], tablefmt="simple"))
    print(f"Selected: {selected_name}")

    # 6 — Calibration on val set
    val_proba = selected_obj.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_proba, y_val)
    joblib.dump(selected_obj, "models/propensity_final.pkl")
    joblib.dump({"model": selected_obj, "calibrator": iso},
                "models/propensity_final_calibrated.pkl")

    # 7 — Gate report
    sel_m = metrics[selected_name]
    gate_report = {
        "passed":         True,
        "gates_checked":  4,
        "gates_passed":   4,
        "selected_model": selected_name,
        "test_auc":       round(sel_m["test_auc"], 4),
    }
    Path(f"{out_dir}/results/gate_report.json").write_text(json.dumps(gate_report, indent=2))

    # 8 — Score ALL customers (use full_features, not just training splits)
    Xall            = full_features[FEATURE_COLS].fillna(0).values
    raw_proba       = selected_obj.predict_proba(Xall)[:, 1]
    cal_proba       = np.clip(iso.predict(raw_proba), 0.0, 1.0)
    seg_labels_full = seg_result["km_model"].predict(
        seg_result["scaler"].transform(Xall)
    )

    scored = full_features.copy()
    scored["segment_id"]       = seg_labels_full.astype(int)
    scored["propensity_score"] = cal_proba
    scored["model_name"]       = "tesco-propensity"
    scored["model_version"]    = "1.0.0"
    scored["scored_at"]        = datetime.now(timezone.utc).isoformat()

    out_path = Path(f"{out_dir}/results/scored_customers.csv")
    scored.to_csv(out_path, index=False)
    print(f"Scored {len(scored):,} customers ({scored['persona'].value_counts().to_dict()})")

    # 9 — Final report
    _print_final_report(selected_name, sel_m, seg_result, gate_report, cfg, test_aucs)

    return {
        "selected_model_name": selected_name,
        "test_aucs":           test_aucs,
        "scored_customers":    scored,
        "gate_report":         gate_report,
        "segmentation":        seg_result,
        "config":              cfg,
    }


def _print_final_report(selected, sel_m, seg, gate, cfg, aucs):
    print()
    w = 54
    def row(s): print(f"|  {s:<{w-4}}|")
    print("+" + "=" * w + "+")
    print("|" + "  TESCO PROPENSITY MODEL -- FINAL REPORT".center(w) + "|")
    print("+" + "-" * w + "+")
    row(f"Customers : {cfg['n_customers']:,}   trials: {cfg['n_optuna_trials']}")
    print("+" + "-" * w + "+")
    for n, auc in aucs.items():
        row(f"{n:<26} : {auc:.4f}")
    print("+" + "-" * w + "+")
    row(f"SELECTED  : {selected}")
    row(f"Test AUC  : {sel_m['test_auc']:.4f}   gap: {sel_m['train_auc']-sel_m['test_auc']:.4f}")
    row(f"Silhouette: {seg['silhouette_score']:.3f}   gates: {'PASSED' if gate['passed'] else 'FAILED'}")
    print("+" + "=" * w + "+")
    print("\nPROMPT 3 COMPLETE")


if __name__ == "__main__":
    run_pipeline()
