# Architecture — Tesco MLOps Azure

## High-Level Design

```
Tesco POS / Online    Event Hub        ADLS Gen2 (medallion)      Databricks
─────────────────     ─────────        ──────────────────────     ──────────
Transactions ──────►  transactions ──► bronze/ (raw Delta)  ──►  01_ingest
                                  ──► silver/ (features)    ──►  02_feature_engineering
                                  ──► gold/   (segments,         03_train_segmentation
                                              propensity)    ──►  04_propensity_model
                                                                       │
                                                               MLflow Registry
                                                                       │
                                                    ┌──────────────────┘
                                                    ▼
                                          AKS (scoring service)
                                          FastAPI /predict
```

## Component Inventory

| Component | Azure Service | Purpose |
|---|---|---|
| Event ingestion | Azure Event Hub (Standard, 4 partitions) | Real-time transaction streaming |
| Data lake | ADLS Gen2 (GRS) | Medallion architecture — bronze/silver/gold |
| Feature engineering | Databricks (Premium) | PySpark RFM + behavioural feature pipeline |
| Model training | Databricks ML cluster + MLflow | KMeans segmentation, LightGBM propensity |
| Model registry | MLflow (Databricks-managed) | Versioned model artefacts + stage promotion |
| Scoring API | AKS + FastAPI | Low-latency `/predict` endpoint |
| Container registry | ACR (Premium SKU) | Geo-replicated image store with private endpoints |
| Secrets management | Azure Key Vault | Connection strings, tokens, storage account name |
| CI/CD | GitHub Actions | Build, push, deploy, Databricks trigger |
| Monitoring | Azure Application Insights | API latency, error rate, throughput |

## Medallion Architecture

```
bronze/transactions/   — raw, schema-on-read Delta, partitioned by ingest_date
silver/customer_features/ — cleaned RFM + channel mix features, partitioned by snapshot_date
gold/customer_segments/   — KMeans segment labels per customer
gold/propensity_scores/   — LightGBM propensity scores, partitioned by target_category
```

## Security Posture

- ACR Premium: no public pull; AKS uses managed identity + ACR pull role
- ADLS Gen2: no public blob access; Databricks accesses via service principal + Key Vault secret scope (`adls-scope`)
- Key Vault: purge protection enabled; RBAC via access policies (not public)
- Event Hub: send-only SAS token for producer; listen-only for consumer (least privilege)
- Kubernetes: non-root container user (UID 1000); secrets injected from K8s Secret, not env literals
- Terraform state: stored in Azure Blob with soft-delete and versioning

## Scaling

- Databricks clusters auto-terminate after 30 min idle
- AKS HPA scales scoring pods 2→10 on CPU>70% or memory>80%
- Event Hub partitions (4) match Databricks streaming parallelism
