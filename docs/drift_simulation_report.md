# Feature Drift Simulation Report
Generated: 2026-05-01 11:04:13 UTC

## Simulation Setup

Drift magnitude: 0.3
Drifted features: recency_days, online_ratio, frequency
PSI retrain threshold: 0.20

## Phase 1 — Baseline Model (Week 0)

| Metric   | Value  |
|----------|--------|
| Test AUC | 0.944  |
| Lift@D1  | 3.75   |
| Max PSI  | 0.1869 |
| Status   | STABLE |

## Phase 2 — Stale Model on Drifted Data

### PSI Scores Per Feature

| Feature               | PSI    | Status  |
|-----------------------|--------|---------|
| recency_days          | 22.5963 | RETRAIN |
| frequency             | 21.8538 | RETRAIN |
| monetary              | 0.1064 | MONITOR |
| avg_basket_size       | 0.0544 | STABLE  |
| basket_std            | 0.0575 | STABLE  |
| online_ratio          | 13.3329 | RETRAIN |
| active_days           | 0.0431 | STABLE  |
| has_promoted_category | 0.0044 | STABLE  |

### Model Performance Degradation

| Metric   | Baseline | Stale | Degradation |
|----------|----------|-------|-------------|
| Test AUC | 0.944    | 0.282 | -0.662       |
| Lift@D1  | 3.75    | 0.00 | -3.75       |

## Phase 3 — Retrained Model (Week 4)

| Metric   | Stale | Retrained | Recovery |
|----------|-------|-----------|----------|
| Test AUC | 0.282 | 0.632     | +0.350    |
| Lift@D1  | 0.00 | 1.41     | +1.41    |

## Conclusion

Drift in [recency_days, online_ratio, frequency] caused AUC to drop from 0.94 to 0.28 (degradation of 0.66). Retraining restored AUC to 0.63, recovering 0.35 of the 0.66 degradation. Full recovery not achieved.