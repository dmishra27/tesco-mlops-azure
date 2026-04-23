variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "uksouth"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging, or prod."
  }
}

variable "project" {
  description = "Project name prefix for resource naming"
  type        = string
  default     = "tesco-mlops"
}

variable "tenant_id" {
  description = "Azure AD tenant ID (used for Key Vault access policy)"
  type        = string
}

variable "deployer_object_id" {
  description = "Object ID of the service principal or user running Terraform"
  type        = string
}

variable "databricks_sku" {
  description = "Databricks workspace SKU"
  type        = string
  default     = "premium"
  validation {
    condition     = contains(["standard", "premium", "trial"], var.databricks_sku)
    error_message = "databricks_sku must be standard, premium, or trial."
  }
}

variable "eventhub_partition_count" {
  description = "Number of Event Hub partitions"
  type        = number
  default     = 4
}

variable "eventhub_message_retention_days" {
  description = "Event Hub message retention in days"
  type        = number
  default     = 7
}

variable "acr_name" {
  description = "Azure Container Registry name (globally unique, alphanumeric only)"

  type        = string
  default     = "tescomlopscr"
}

variable "tags" {
  description = "Common tags applied to all resources"
  type        = map(string)
  default = {
    project     = "tesco-mlops"
    managed_by  = "terraform"
    cost_centre = "ml-platform"
  }
}
