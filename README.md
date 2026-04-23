# tesco-mlops-azure

Production Azure MLOps platform for Tesco customer segmentation and propensity modelling.

## Architecture

Event Hub → ADLS Gen2 (bronze/silver/gold) → Databricks ML pipelines → MLflow Registry → AKS FastAPI scoring

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and component inventory.

## Repo Structure

```
tesco-mlops-azure/
├── infra/terraform/          # Azure infrastructure (IaC)
│   ├── main.tf               # Resources: Databricks, ADLS, ACR Premium, Key Vault, Event Hub
│   ├── variables.tf
│   ├── outputs.tf
│   └── providers.tf
├── databricks/
│   ├── notebooks/            # Run in order: 01→02→03→04
│   │   ├── 01_ingest.py      # Event Hub → bronze Delta
│   │   ├── 02_feature_engineering.py  # bronze → silver RFM features
│   │   ├── 03_train_segmentation.py   # KMeans + MLflow
│   │   └── 04_propensity_model.py     # LightGBM + MLflow
│   └── jobs/run_job.json     # Databricks Jobs API pipeline definition
├── ml/
│   ├── train.py              # Local / MLflow Projects training entry-point
│   ├── score.py              # FastAPI scoring service (/health, /ready, /predict)
│   ├── requirements.txt      # fastapi, uvicorn, lightgbm, mlflow, ...
│   ├── Dockerfile            # uvicorn-based production image
│   └── mlflow_projects/MLproject
├── k8s/
│   ├── deployment.yaml       # AKS deployment + HPA
│   └── service.yaml          # Internal LoadBalancer service
├── .github/workflows/
│   └── ci-cd.yml             # test → build → deploy → Databricks trigger
├── producer/
│   └── send_event.py         # Async synthetic transaction producer
└── docs/
    ├── architecture.md
    └── runbook.md
```

## Quick Start

1. **Bootstrap infra** — see [`docs/runbook.md`](docs/runbook.md) §1
2. **Configure Databricks secret scope** — see runbook §2
3. **Upload notebooks and register the job** — runbook §3
4. **Run training pipeline** — runbook §4
5. **Deploy scoring API** — push to `main`; GitHub Actions handles build + deploy

## GitHub Actions Secrets Required

| Secret | Description |
|---|---|
| `AZURE_CREDENTIALS` | `az ad sp create-for-rbac --sdk-auth` JSON |
| `DATABRICKS_HOST` | `https://<workspace>.azuredatabricks.net` |
| `DATABRICKS_TOKEN` | Databricks personal access token |

## Python Version

Python 3.11 throughout.
