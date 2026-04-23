output "resource_group_name" {
  description = "Name of the primary resource group"
  value       = azurerm_resource_group.this.name
}

output "datalake_storage_account_name" {
  description = "ADLS Gen2 storage account name"
  value       = azurerm_storage_account.datalake.name
}

output "databricks_workspace_url" {
  description = "Databricks workspace URL (used as MLFLOW_TRACKING_URI base)"
  value       = "https://${azurerm_databricks_workspace.this.workspace_url}"
}

output "databricks_workspace_id" {
  description = "Databricks workspace resource ID"
  value       = azurerm_databricks_workspace.this.id
}

output "aml_workspace_name" {
  description = "Azure Machine Learning workspace name"
  value       = azurerm_machine_learning_workspace.this.name
}

output "acr_login_server" {
  description = "Azure Container Registry login server (Premium SKU)"
  value       = azurerm_container_registry.this.login_server
}

output "key_vault_uri" {
  description = "Key Vault URI for secret references"
  value       = azurerm_key_vault.this.vault_uri
}

output "key_vault_name" {
  description = "Key Vault name"
  value       = azurerm_key_vault.this.name
}

output "eventhub_namespace_name" {
  description = "Event Hub namespace name"
  value       = azurerm_eventhub_namespace.this.name
}

# Bug fix #5: marked sensitive=true — connection strings must never appear in
# plain-text plan output or CI logs
output "eventhub_connection_string" {
  description = "Event Hub producer connection string (sensitive)"
  value       = azurerm_eventhub_authorization_rule.producer.primary_connection_string
  sensitive   = true
}

output "storage_containers" {
  description = "Medallion architecture containers (bronze / silver / gold)"
  value = {
    bronze = azurerm_storage_container.bronze.name
    silver = azurerm_storage_container.silver.name
    gold   = azurerm_storage_container.gold.name
  }
}
