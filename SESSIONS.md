Last updated: 24 April 2026 (Session 3)

## Session 1 — 23 April 2026 (completed)

### Infrastructure (Terraform)
- [x] providers.tf — AzureRM + Databricks providers, remote backend
- [x] variables.tf — all input variables with validation
- [x] main.tf — full Azure resource definitions
- [x] outputs.tf — resource outputs with sensitive=true

### Bug Fixes Applied (all 13)
- [x] Fix 1: azurerm_databricks_workspace replaces broken MWS placeholder
- [x] Fix 2: silver + gold containers added
- [x] Fix 3: ACR SKU upgraded Basic → Premium
- [x] Fix 4: Key Vault resource added
- [x] Fix 5: sensitive=true on eventhub output
- [x] Fix 6: all <STORAGE_ACCOUNT> replaced with dbutils.secrets.get()
- [x] Fix 7: Flask → FastAPI with /health /ready /predict endpoints
- [x] Fix 8: requirements.txt updated to fastapi + uvicorn[standard] + httpx
- [x] Fix 9: Dockerfile CMD updated to uvicorn
- [x] Fix 10: MLFLOW_TRACKING_URI injected via K8s Secret
- [x] Fix 11: kubectl apply k8s/service.yaml added to CI/CD
- [x] Fix 12: Databricks CLI job trigger added to CI/CD
- [x] Fix 13: .github/workflows/ moved to repo root

### Databricks Notebooks
- [x] 01_ingest.py — Event Hub → bronze Delta streaming
- [x] 02_feature_engineering.py — bronze → silver RFM
- [x] 03_train_segmentation.py — KMeans + MLflow
- [x] 04_propensity_model.py — LightGBM + MLflow

### ML Serving
- [x] ml/score.py — FastAPI scoring service
- [x] ml/train.py — local training entry point
- [x] ml/requirements.txt
- [x] ml/Dockerfile — uvicorn, non-root user, HEALTHCHECK
- [x] ml/mlflow_projects/MLproject

### Infrastructure and Deployment
- [x] k8s/deployment.yaml — AKS Deployment + HPA
- [x] k8s/service.yaml — Azure LoadBalancer service
- [x] .github/workflows/ci-cd.yml — 4-job pipeline
- [x] databricks/jobs/run_job.json — multi-task DAG

### Documentation
- [x] docs/architecture.md
- [x] docs/runbook.md
- [x] CLAUDE.md
- [x] README.md — portfolio-quality with Mermaid diagrams, badges, 13 design decisions
- [x] .gitignore
- [x] Repo pushed to github.com/dmishra27/tesco-mlops-azure

### Stats
- Total files created: 26
- Total lines of code: ~1,905
- All 13 architectural bugs fixed
- Build time: ~6 minutes

---

## Session 2 — 24 April 2026 (completed)

### Gap 1 — Power BI / Synapse
- [x] docs/powerbi_integration.md
- [x] infra/terraform/main.tf — azurerm_synapse_workspace + filesystem + firewall + RBAC
- [x] infra/terraform/variables.tf — synapse_sql_admin_username/password (sensitive)
- [x] infra/terraform/outputs.tf — synapse_workspace_name + synapse_serverless_endpoint
- [x] README.md — Analytics/BI row added to Tech Stack table

### Gap 2 — Airflow DAGs (verified exist)
- [x] airflow/dags/tesco_ml_pipeline.py — daily training DAG ✓
- [x] airflow/dags/tesco_batch_scoring.py — weekly scoring DAG ✓
- [x] airflow/requirements.txt ✓

### Gap 3 — Tests (28 tests, all passing)
- [x] tests/conftest.py — fixed: timestamp fixture now produces pd.Timestamp
- [x] tests/unit/test_feature_engineering.py — 13 tests; fixed include_groups FutureWarning
- [x] tests/unit/test_score_api.py — 15 tests
- [x] pytest: 28 passed, 0 warnings

### Gap 4 — Fix SIGIR DOI (CRITICAL)
- [x] README.md — DOI fixed: 3657967 → 3657765

### Gap 5 — Terraform tfvars example
- [x] infra/terraform/terraform.tfvars.example
- [x] README.md Quickstart — cp command added

### Gap 6 — Architecture diagrams (verified exist)
- [x] docs/architecture_diagram.md — three Mermaid diagrams present ✓

### Gap 7 — Commit and push
- [x] Committed: "feat: Session 2 gaps — Synapse/Power BI, tests, tfvars, DOI fix"
- [x] Pushed to origin/master (056f848)

### Stats
- Files created: 2 (powerbi_integration.md, terraform.tfvars.example)
- Files modified: 6 (README, main.tf, outputs.tf, variables.tf, conftest.py, test_feature_engineering.py)
- Tests: 28 passed (was 0 passing due to conftest bug)

---

## Session 3 — 24 April 2026 (completed)

### Propensity Model Pipeline

- [x] Full model progression (7 models)
- [x] Optuna hyperparameter tuning (XGBoost + LightGBM, 50 trials each)
- [x] TimeSeriesSplit cross-validation (n_splits=5, gap=7) for all sklearn models
- [x] Learning curves and bias-variance analysis (Logistic Regression)
- [x] Overfitting curve scan depths 2-12 (Decision Tree)
- [x] OOB score trajectory n_trees 10-500 (Random Forest)
- [x] Feature importance stability across 5 seeds (Random Forest)
- [x] Stacking ensemble with temporal forward-chaining CV (5 folds)
- [x] Soft voting ensemble (weights proportional to val AUC)
- [x] Model selection with 6 gate criteria
- [x] Calibration check with isotonic fallback (max gap 0.361 reduced)
- [x] SHAP business interpretability table (LightGBM + XGBoost)
- [x] Persona recovery ground truth check: 3/3
- [x] Results saved to data/results/scored_customers.csv
- [x] docs/model_selection_results.md

### Data pipeline

- [x] ml/local/generate.py — 5,000 customers, 50,000 transactions, 3 personas
- [x] ml/local/feature_engineering.py — RFM + behavioural features, temporal splits
- [x] data/synthetic/transactions.csv + customers.csv
- [x] data/features/customer_features.csv
- [x] data/splits/train.csv + val.csv + test.csv (22-27% positive rate)

### Model artefacts

- [x] models/propensity_final.pkl (Logistic Regression, selected model)
- [x] models/propensity_final_calibrated.pkl (isotonic calibrated)

### Key results

| Model         | Test_AUC | Gap    | Diagnosis           |
|---------------|----------|--------|---------------------|
| Logistic Reg  | 0.7706   | -0.023 | WELL BALANCED       |
| Decision Tree | 0.7356   | 0.012  | WELL BALANCED       |
| Random Forest | 0.7631   | 0.037  | WELL BALANCED       |
| XGBoost       | 0.7299   | 0.166  | HIGH VARIANCE       |
| LightGBM      | 0.7661   | 0.021  | WELL BALANCED       |
| Stacking Ens  | 0.7302   | 0.133  | HIGH VARIANCE       |
| Voting Ens    | 0.7544   | 0.097  | MODERATE VARIANCE   |

Selected: **Logistic Regression** (Test AUC 0.7706, Lift@D1=3.05, passes G1/G2/G3)  
Persona recovery: **3/3** (A in top decile 76.2%, B in decile 2-3 66.4%, C in bottom half 75.8%)

### Stretch goals (future)

- [ ] Add pytest coverage report to CI/CD
- [ ] Add terraform fmt and tflint to CI/CD
- [ ] Add pre-commit hooks (.pre-commit-config.yaml)
- [ ] Consider Azure ML pipeline as alternative to Databricks Jobs

---

## Project Reference
- GitHub: https://github.com/dmishra27/tesco-mlops-azure
- Stack: Azure + Databricks + MLflow + FastAPI + Airflow + Terraform + GitHub Actions
- Python: 3.11
- Total sessions planned: 3
