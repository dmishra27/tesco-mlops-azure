SELECTION_THRESHOLDS: dict[str, float] = {
    "propensity_auc_min":          0.70,
    "overfit_gap_max":             0.08,
    "lift_decile1_min":            2.5,
    "cv_std_max":                  0.03,
    "silhouette_min":              0.25,
    "dominant_cluster_max":        0.60,
    "tiny_cluster_min":            0.01,
    "tiebreaker_delta":            0.01,
    "ensemble_justification_delta": 0.015,
    "baseline_gate_min_gain":      0.03,
    "auc_regression_max":          0.05,
}
