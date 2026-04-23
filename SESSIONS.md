Last updated: 23 April 2026

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

## Session 2 — 24 April 2026 (to do)

### Gap 1 — Power BI / Synapse (MISSING from spec)
- [ ] docs/powerbi_integration.md
- [ ] infra/terraform/main.tf — add azurerm_synapse_workspace
- [ ] README.md — add Power BI row to Tech Stack table

### Gap 2 — Airflow DAGs (verify exist, create if missing)
- [ ] airflow/dags/tesco_ml_pipeline.py — daily training DAG
- [ ] airflow/dags/tesco_batch_scoring.py — weekly scoring DAG
- [ ] airflow/requirements.txt

### Gap 3 — Tests (verify 25 tests exist and pass)
- [ ] tests/conftest.py — shared fixtures
- [ ] tests/unit/test_feature_engineering.py — 12 tests
- [ ] tests/unit/test_score_api.py — 13 tests
- [ ] Run pytest and confirm 25 tests pass

### Gap 4 — Fix SIGIR DOI (CRITICAL)
- [ ] README.md — fix DOI from 3657967 to 3657765

### Gap 5 — Terraform tfvars example (MISSING)
- [ ] infra/terraform/terraform.tfvars.example
- [ ] README.md Quickstart — add cp command for tfvars setup

### Gap 6 — Architecture diagrams (verify exist)
- [ ] docs/architecture_diagram.md — three Mermaid diagrams

### Gap 7 — Commit and push all Session 2 changes
- [ ] git add .
- [ ] git commit -m "feat: Session 2 gaps — Airflow, tests, Synapse, tfvars"
- [ ] git push origin master

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
