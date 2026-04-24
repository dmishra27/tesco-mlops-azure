# Power BI Integration — Tesco MLOps Azure

## Overview

The gold layer of the medallion data lake is the authoritative source for operational
reporting. Azure Synapse Analytics sits between ADLS Gen2 and Power BI, providing a
serverless SQL query layer that avoids loading raw Delta files directly into Power BI
Desktop.

```
ADLS Gen2 (gold/)
    └── customer_segments/    ← KMeans segment labels + RFM features
    └── propensity_scores/    ← LightGBM per-category scores
         │
    Azure Synapse Analytics (Serverless SQL pool)
         │  CREATE EXTERNAL TABLE / OPENROWSET on Delta
         │
    Power BI (DirectQuery or Import)
         │
    Dashboards → Merchandising, Marketing, Finance
```

---

## Synapse Workspace

The Synapse workspace (`azurerm_synapse_workspace`) is provisioned by Terraform in
`infra/terraform/main.tf`. It uses the same ADLS Gen2 storage account as the data lake
and creates a dedicated filesystem container (`synapse`) for internal workspace data.

| Resource | Name pattern | Notes |
|---|---|---|
| `azurerm_synapse_workspace` | `<prefix>-synapse` | System-assigned identity; linked to ADLS |
| `azurerm_synapse_firewall_rule` | `AllowAzureServices` | Allows Power BI gateway outbound |
| ADLS container | `synapse` | Workspace scratch / temp storage |

---

## External Tables (Serverless SQL)

Connect to the Synapse Serverless endpoint from Synapse Studio or SSMS, then create
views over the gold Delta files:

```sql
-- customer_segments view
CREATE OR ALTER VIEW gold.customer_segments AS
SELECT *
FROM OPENROWSET(
    BULK 'https://<storage_account>.dfs.core.windows.net/gold/customer_segments/**',
    FORMAT = 'DELTA'
) AS [result];

-- propensity_scores view
CREATE OR ALTER VIEW gold.propensity_scores AS
SELECT *
FROM OPENROWSET(
    BULK 'https://<storage_account>.dfs.core.windows.net/gold/propensity_scores/**',
    FORMAT = 'DELTA'
) AS [result];
```

Replace `<storage_account>` with the value of `terraform output datalake_storage_account_name`.

Grant the Synapse managed identity **Storage Blob Data Reader** on the ADLS account so
it can read gold Delta files without SAS tokens.

---

## Connecting Power BI

1. Open Power BI Desktop → **Get Data** → **Azure** → **Azure Synapse Analytics (SQL)**.
2. Enter the Serverless SQL endpoint:
   `<workspace_name>-ondemand.sql.azuresynapse.net`
3. Select **DirectQuery** for live results (recommended for dashboards refreshed by the
   Airflow scoring DAG), or **Import** for snapshots requiring complex DAX.
4. Choose `gold.customer_segments` and `gold.propensity_scores`.

### Recommended DirectQuery dataset structure

| Table | Key columns | Refresh cadence |
|---|---|---|
| `customer_segments` | `customer_id`, `segment_id`, `recency_days`, `monetary`, `snapshot_date` | Daily (post Airflow training DAG) |
| `propensity_scores` | `customer_id`, `category`, `propensity_score`, `scored_at` | Weekly (post batch scoring DAG) |

---

## Key Dashboards

| Dashboard | Audience | Primary metric |
|---|---|---|
| Segment Distribution | Merchandising | Customer count and revenue share per KMeans cluster |
| Propensity Heatmap | Marketing | Top-propensity customers per category (ready_meals, bakery, etc.) |
| Drift Monitor | ML Platform | PSI trend per feature; flag when PSI > 0.2 triggers retraining |
| API Latency | Engineering | p50 / p95 `/predict` latency from Application Insights |

---

## Access Control

- Power BI workspace service principal must be granted **Reader** on the Synapse workspace.
- Synapse managed identity must be granted **Storage Blob Data Reader** on the ADLS account.
- Row-level security (RLS) is not required for the current internal-only deployment; add
  Power BI RLS if the dataset is shared outside the ML platform team.

---

## Operational Notes

- Gold layer is written by Airflow DAGs (`tesco_ml_pipeline.py` daily,
  `tesco_batch_scoring.py` weekly). Power BI DirectQuery reflects the latest partition
  automatically; Import mode requires a scheduled dataset refresh aligned to the DAG
  completion time (+30 min buffer recommended).
- Delta Lake schema evolution is transparent to Serverless SQL — new columns appear in
  OPENROWSET results automatically without DDL changes.
- For production, add a Synapse Linked Service connection string as a Key Vault secret
  (`synapse-serverless-endpoint`) and reference it from Airflow or downstream consumers.
