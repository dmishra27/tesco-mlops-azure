terraform {
  required_version = ">= 1.5"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.40"
    }
  }

  backend "azurerm" {
    resource_group_name  = "tesco-mlops-tfstate-rg"
    storage_account_name = "tescomlopstfstate"
    container_name       = "tfstate"
    key                  = "tesco-mlops.terraform.tfstate"
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = false
      recover_soft_deleted_key_vaults = true
    }
  }
  subscription_id = var.subscription_id
}

provider "databricks" {
  host = azurerm_databricks_workspace.this.workspace_url
}
