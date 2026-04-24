Last updated: 24 April 2026

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

## Session 3 — Future (stretch goals)

- [ ] monitoring/drift_detector.py — PSI drift detection
- [ ] Add Airflow drift monitoring task to DAG
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
