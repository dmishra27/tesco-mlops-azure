# Runbook — Tesco MLOps Azure

## Prerequisites

- Azure CLI ≥ 2.55 (`az --version`)
- Terraform ≥ 1.5 (`terraform --version`)
- Databricks CLI (`pip install databricks-cli`)
- kubectl ≥ 1.28
- Docker

---

## 1. Bootstrap Infrastructure

```bash
cd infra/terraform

# Create tfstate backend first (one-time)
az group create -n tesco-mlops-tfstate-rg -l uksouth
az storage account create -n tescomlopstfstate -g tesco-mlops-tfstate-rg --sku Standard_LRS
az storage container create -n tfstate --account-name tescomlopstfstate

# Deploy
terraform init
terraform plan -var-file=environments/prod.tfvars -out=tfplan
terraform apply tfplan
```

Required `prod.tfvars` values:
- `subscription_id` — Azure subscription GUID
- `tenant_id` — Azure AD tenant GUID
- `deployer_object_id` — Object ID of the CI service principal

---

## 2. Configure Databricks Secret Scope

```bash
# After terraform apply, get storage account name
SA_NAME=$(terraform output -raw datalake_storage_account_name)
SA_KEY=$(az storage account keys list --account-name $SA_NAME --query '[0].value' -o tsv)
EH_CS=$(terraform output -raw eventhub_connection_string)  # sensitive, handle carefully

databricks secrets create-scope --scope adls-scope
databricks secrets put --scope adls-scope --key STORAGE_ACCOUNT  --string-value "$SA_NAME"
databricks secrets put --scope adls-scope --key EVENTHUB_CONNECTION_STRING --string-value "$EH_CS"
```

---

## 3. Deploy Databricks Notebooks

```bash
# Upload notebooks to workspace
for nb in databricks/notebooks/*.py; do
    name=$(basename "$nb" .py)
    databricks workspace import \
        --language PYTHON \
        --overwrite \
        "$nb" \
        "/Shared/tesco-mlops/${name}"
done

# Register the pipeline job
databricks jobs create --json @databricks/jobs/run_job.json
```

---

## 4. Run the Training Pipeline

```bash
JOB_ID=$(databricks jobs list --output json \
    | python3 -c "import sys,json; print(next(j['job_id'] for j in json.load(sys.stdin)['jobs'] if j['settings']['name']=='tesco-mlops-training-pipeline'))")

databricks jobs run-now --job-id $JOB_ID
```

---

## 5. Build and Deploy the Scoring API

```bash
# Build image locally (for testing)
cd ml/
docker build -t tesco-mlops-scoring:local .
docker run -p 8080:8080 \
    -e MLFLOW_TRACKING_URI=<databricks-workspace-url> \
    -e MLFLOW_TRACKING_TOKEN=<databricks-pat> \
    tesco-mlops-scoring:local

# Verify
curl http://localhost:8080/health
curl http://localhost:8080/ready

# Push via CI/CD (normal path — push to main branch)
```

---

## 6. AKS K8s Secrets Setup (one-time per cluster)

```bash
kubectl create namespace mlops

kubectl create secret generic mlops-secrets \
    --namespace mlops \
    --from-literal=MLFLOW_TRACKING_URI=<databricks-workspace-url> \
    --from-literal=DATABRICKS_TOKEN=<databricks-pat>

kubectl create secret docker-registry acr-pull-secret \
    --namespace mlops \
    --docker-server=tescomlopscr.azurecr.io \
    --docker-username=<sp-client-id> \
    --docker-password=<sp-client-secret>
```

---

## 7. Promote a Model to Production

```bash
# Via MLflow Python client
python3 - <<'EOF'
import mlflow
mlflow.set_tracking_uri("<databricks-workspace-url>")
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="tesco-customer-segmentation",
    version=<version>,
    stage="Production",
    archive_existing_versions=True,
)
EOF
```

---

## Operational Alerts

| Alert | Cause | Action |
|---|---|---|
| `/ready` returns 503 | Model failed to load from MLflow | Check `MLFLOW_TRACKING_URI` secret; verify model version is in Production stage |
| Databricks job failing at ingest | Event Hub SAS token expired | Rotate via Key Vault and update Databricks secret scope |
| ACR pull failure in AKS | `acr-pull-secret` expired | Rotate SP credentials, recreate K8s secret |
| Silhouette score < 0.3 | Feature drift | Re-run feature engineering with extended lookback window |
