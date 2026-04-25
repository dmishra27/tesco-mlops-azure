# Tesco Propensity Model — Selection Results

**Run date:** 25 April 2026 (Session 3 recovery)
**Pipeline:** `ml/local/run_pipeline.py` (default config, n=5000, 50 Optuna trials)
**Python:** 3.14 · scikit-learn · XGBoost · LightGBM · Optuna

---

## Pipeline Configuration

| Parameter | Value |
|---|---|
| Customers | 5,000 |
| Transactions | 33,495 (after temporal sampling) |
| Feature window | Days 0–119 (training) |
| Scoring window | Days 0–179 (all customers) |
| Optuna trials (XGB/LGBM) | 50 |
| CV | TimeSeriesSplit(n_splits=5, gap=7) |
| Seed | 42 |

---

## Model Comparison

| Model | Test AUC | Train AUC | Gap | Diagnosis |
|---|---|---|---|---|
| Logistic Regression | 0.7682 | 0.7657 | −0.002 | WELL BALANCED |
| Decision Tree | 0.7632 | 0.7683 | +0.005 | WELL BALANCED |
| Random Forest | 0.7750 | 0.8156 | +0.041 | WELL BALANCED |
| XGBoost | 0.7728 | 0.8093 | +0.037 | WELL BALANCED |
| LightGBM | 0.7483 | 0.9315 | +0.183 | HIGH VARIANCE |

---

## Selected Model

**Logistic Regression** — selected by Occam's razor tiebreaker
- Test AUC: **0.7682**
- Train–test gap: −0.0024 (no overfitting)
- Passes all quality gates (G1–G4)

Selection rationale: Random Forest and XGBoost also pass all gates but are within 0.01 AUC of Logistic Regression. Occam's razor tiebreaker (G5) prefers the simplest interpretable model within the tolerance band.

---

## Segmentation

| Metric | Value |
|---|---|
| Silhouette score | 0.297 |
| Segment sizes | 29% / 43% / 27% |
| Gates passed | Yes |

---

## Persona Recovery (Ground Truth Check)

| Persona | Description | Recovery |
|---|---|---|
| A (Loyalists) | High-value, high-frequency | **95.6% in top decile** |
| C (At-risk) | Low-frequency, low-spend | **71.4% in bottom half** |

---

## Calibration

Isotonic regression calibration applied on validation set.
Output: `models/propensity_final_calibrated.pkl`
All propensity scores constrained to [0.0, 1.0].

---

## Output Files

| File | Description |
|---|---|
| `models/propensity_final.pkl` | Uncalibrated Logistic Regression model |
| `models/propensity_final_calibrated.pkl` | Isotonic-calibrated model + calibrator |
| `data/results/scored_customers.csv` | 5,000 customers with propensity + segment |
| `data/results/gate_report.json` | Model quality gate report |
