locals {
  prefix = "${var.project}-${var.environment}"
}

# ── Resource Group ────────────────────────────────────────────────────────────
resource "azurerm_resource_group" "this" {
  name     = "${local.prefix}-rg"
  location = var.location
  tags     = var.tags
}

# ── ADLS Gen2 (Data Lake) ─────────────────────────────────────────────────────
resource "azurerm_storage_account" "datalake" {
  name                     = replace("${local.prefix}adls", "-", "")
  resource_group_name      = azurerm_resource_group.this.name
  location                 = azurerm_resource_group.this.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true # ADLS Gen2 hierarchical namespace
  min_tls_version          = "TLS1_2"

  tags = var.tags
}

# Bug fix #2: bronze, silver AND gold containers (original only had bronze)
resource "azurerm_storage_container" "bronze" {
  name                  = "bronze"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "silver" {
  name                  = "silver"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "gold" {
  name                  = "gold"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

# ── Azure Databricks Workspace ─────────────────────────────────────────────────
# Bug fix #1: replaced broken databricks_mws_workspaces placeholder with
# the correct azurerm_databricks_workspace resource
resource "azurerm_databricks_workspace" "this" {
  name                        = "${local.prefix}-dbx"
  resource_group_name         = azurerm_resource_group.this.name
  location                    = azurerm_resource_group.this.location
  sku                         = var.databricks_sku
  managed_resource_group_name = "${local.prefix}-dbx-managed-rg"

  tags = var.tags
}

# ── Azure Machine Learning Workspace ─────────────────────────────────────────
resource "azurerm_application_insights" "this" {
  name                = "${local.prefix}-appinsights"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  application_type    = "web"
  tags                = var.tags
}

resource "azurerm_machine_learning_workspace" "this" {
  name                    = "${local.prefix}-aml"
  location                = azurerm_resource_group.this.location
  resource_group_name     = azurerm_resource_group.this.name
  application_insights_id = azurerm_application_insights.this.id
  key_vault_id            = azurerm_key_vault.this.id
  storage_account_id      = azurerm_storage_account.datalake.id
  container_registry_id   = azurerm_container_registry.this.id

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# ── Azure Container Registry ──────────────────────────────────────────────────
# Bug fix #3: upgraded SKU from Basic to Premium for geo-replication and
# private endpoint support required in production
resource "azurerm_container_registry" "this" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  sku                 = "Premium"
  admin_enabled       = false

  tags = var.tags
}

# ── Key Vault ─────────────────────────────────────────────────────────────────
# Bug fix #4: Key Vault resource was missing entirely in the original design
resource "azurerm_key_vault" "this" {
  name                        = "${local.prefix}-kv"
  location                    = azurerm_resource_group.this.location
  resource_group_name         = azurerm_resource_group.this.name
  tenant_id                   = var.tenant_id
  sku_name                    = "standard"
  purge_protection_enabled    = true
  soft_delete_retention_days  = 90
  enable_rbac_authorization   = false

  access_policy {
    tenant_id = var.tenant_id
    object_id = var.deployer_object_id

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Recover", "Backup", "Restore", "Purge"
    ]
    key_permissions = [
      "Get", "List", "Create", "Delete", "Recover", "Backup", "Restore"
    ]
  }

  # Grant AML system-assigned identity read access
  access_policy {
    tenant_id = var.tenant_id
    object_id = azurerm_machine_learning_workspace.this.identity[0].principal_id

    secret_permissions = ["Get", "List"]
    key_permissions    = ["Get", "List"]
  }

  tags = var.tags
}

# ── Event Hub (real-time transaction streaming) ───────────────────────────────
resource "azurerm_eventhub_namespace" "this" {
  name                = "${local.prefix}-ehns"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  sku                 = "Standard"
  capacity            = 2
  tags                = var.tags
}

resource "azurerm_eventhub" "transactions" {
  name                = "transactions"
  namespace_name      = azurerm_eventhub_namespace.this.name
  resource_group_name = azurerm_resource_group.this.name
  partition_count     = var.eventhub_partition_count
  message_retention   = var.eventhub_message_retention_days
}

resource "azurerm_eventhub_authorization_rule" "producer" {
  name                = "producer"
  namespace_name      = azurerm_eventhub_namespace.this.name
  eventhub_name       = azurerm_eventhub.transactions.name
  resource_group_name = azurerm_resource_group.this.name
  listen              = false
  send                = true
  manage              = false
}

resource "azurerm_eventhub_authorization_rule" "consumer" {
  name                = "consumer"
  namespace_name      = azurerm_eventhub_namespace.this.name
  eventhub_name       = azurerm_eventhub.transactions.name
  resource_group_name = azurerm_resource_group.this.name
  listen              = true
  send                = false
  manage              = false
}

# Store Event Hub connection strings in Key Vault
resource "azurerm_key_vault_secret" "eventhub_producer_cs" {
  name         = "eventhub-producer-connection-string"
  value        = azurerm_eventhub_authorization_rule.producer.primary_connection_string
  key_vault_id = azurerm_key_vault.this.id
  depends_on   = [azurerm_key_vault.this]
}

resource "azurerm_key_vault_secret" "eventhub_consumer_cs" {
  name         = "eventhub-consumer-connection-string"
  value        = azurerm_eventhub_authorization_rule.consumer.primary_connection_string
  key_vault_id = azurerm_key_vault.this.id
  depends_on   = [azurerm_key_vault.this]
}

resource "azurerm_key_vault_secret" "storage_account_name" {
  name         = "STORAGE-ACCOUNT"
  value        = azurerm_storage_account.datalake.name
  key_vault_id = azurerm_key_vault.this.id
  depends_on   = [azurerm_key_vault.this]
}

# ── Azure Synapse Analytics ───────────────────────────────────────────────────
# Provides a serverless SQL query layer over the gold Delta containers,
# consumed by Power BI DirectQuery dashboards.
resource "azurerm_storage_container" "synapse" {
  name                  = "synapse"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

resource "azurerm_synapse_workspace" "this" {
  name                                 = "${local.prefix}-synapse"
  resource_group_name                  = azurerm_resource_group.this.name
  location                             = azurerm_resource_group.this.location
  storage_data_lake_gen2_filesystem_id = azurerm_storage_data_lake_gen2_filesystem.synapse.id
  sql_administrator_login              = var.synapse_sql_admin_username
  sql_administrator_login_password     = var.synapse_sql_admin_password

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

resource "azurerm_storage_data_lake_gen2_filesystem" "synapse" {
  name               = "synapse"
  storage_account_id = azurerm_storage_account.datalake.id
}

resource "azurerm_synapse_firewall_rule" "allow_azure_services" {
  name                 = "AllowAzureServices"
  synapse_workspace_id = azurerm_synapse_workspace.this.id
  start_ip_address     = "0.0.0.0"
  end_ip_address       = "0.0.0.0"
}

# Grant Synapse managed identity read access to the gold layer in ADLS
resource "azurerm_role_assignment" "synapse_adls_reader" {
  scope                = azurerm_storage_account.datalake.id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_synapse_workspace.this.identity[0].principal_id
}
