# CLAUDE.md — tesco-mlops-azure

## Project Overview

Production Azure MLOps platform for Tesco customer segmentation and propensity modelling.
Medallion data lake (bronze/silver/gold), Databricks ML pipelines, FastAPI scoring on AKS.

## Repo Layout

| Path | Purpose |
|---|---|
| `infra/terraform/` | Azure infrastructure (Databricks, ADLS, ACR Premium, Key Vault, Event Hub) |
| `databricks/notebooks/` | PySpark + MLflow notebooks — run in order 01→04 |
| `databricks/jobs/` | Databricks Jobs API JSON definition |
| `ml/` | `train.py` (local/MLflow Projects), `score.py` (FastAPI), `Dockerfile`, `requirements.txt` |
| `ml/mlflow_projects/` | MLproject entry-point for `mlflow run` |
| `k8s/` | Kubernetes deployment and service manifests |
| `.github/workflows/` | CI/CD pipeline (test → build → deploy → Databricks trigger) |
| `producer/` | Async Event Hub producer for synthetic transaction data |
| `docs/` | Architecture overview and operational runbook |

## Common Commands

```bash
# Terraform
cd infra/terraform && terraform init && terraform plan -var-file=environments/prod.tfvars

# Local scoring API
cd ml && uvicorn score:app --host 0.0.0.0 --port 8080 --reload

# Run training locally via MLflow Projects
mlflow run ml/mlflow_projects/ -P features_path=data/customer_features.parquet

# Lint
ruff check ml/ producer/

# Tests
pytest tests/ -v
```

## Key Design Decisions

- **FastAPI over Flask**: production readiness, async, native Pydantic validation, /health + /ready K8s probes
- **ACR Premium SKU**: required for geo-replication and private endpoints (Basic unsuitable for prod)
- **Databricks secrets scope `adls-scope`**: all notebooks retrieve secrets at runtime — no hardcoded values
- **`sensitive = true` on Terraform outputs**: prevents connection strings appearing in CI logs
- **Silver and Gold ADLS containers**: full medallion architecture; original design only had bronze

## Secrets Required (GitHub Actions)

| Secret | Value |
|---|---|
| `AZURE_CREDENTIALS` | JSON output of `az ad sp create-for-rbac --sdk-auth` |
| `DATABRICKS_HOST` | `https://<workspace>.azuredatabricks.net` |
| `DATABRICKS_TOKEN` | Databricks PAT |

## Python Version

Python 3.11 throughout (scoring API, training, producer).
