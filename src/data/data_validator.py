"""
Data Validator Module
SaaS Revenue Risk & Retention Intelligence System
Week 1 – Schema-aware validation with proper boolean/int handling
"""

import logging
import pandas as pd
from typing import Dict


class DataValidator:
    
    EXPECTED_DATASETS = [
        "accounts",
        "subscriptions",
        "feature_usage",
        "support_tickets",
        "churn_events",
    ]
    
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)
        self.logger.info("DataValidator initialized.")
    
    def _setup_logger(self, log_level):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    # ---------------------------------------------------
    # RAW DATA VALIDATION (Schema-aware)
    # ---------------------------------------------------
    def validate_raw_datasets(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """
        Validate raw datasets for required structure and schema.
        
        Checks:
        - All required datasets present
        - No empty datasets
        - Required columns exist
        - Foreign key columns present
        - No duplicate account_ids in accounts table
        """
        
        if not datasets:
            raise ValueError("Datasets dictionary is empty.")
        
        # Check all required datasets present
        missing = [k for k in self.EXPECTED_DATASETS if k not in datasets]
        if missing:
            raise ValueError(f"Missing required datasets: {missing}")
        
        # Check for empty datasets
        for name, df in datasets.items():
            if df.empty:
                raise ValueError(f"Dataset '{name}' is empty.")
        
        # Schema validation - required columns
        self.logger.info("Validating schema...")
        
        if "account_id" not in datasets["accounts"].columns:
            raise ValueError("accounts must contain 'account_id'")
        
        if "account_id" not in datasets["subscriptions"].columns:
            raise ValueError("subscriptions must contain 'account_id'")
        
        if "subscription_id" not in datasets["feature_usage"].columns:
            raise ValueError("feature_usage must contain 'subscription_id'")
        
        if "account_id" not in datasets["support_tickets"].columns:
            raise ValueError("support_tickets must contain 'account_id'")
        
        if "account_id" not in datasets["churn_events"].columns:
            raise ValueError("churn_events must contain 'account_id'")
        
        # Uniqueness check for primary key
        if datasets["accounts"]["account_id"].duplicated().any():
            raise ValueError("Duplicate account_id found in accounts dataset.")
        
        self.logger.info("Raw dataset validation passed.")
    
    # ---------------------------------------------------
    # ACCOUNT-LEVEL DATASET VALIDATION
    # ---------------------------------------------------
    def validate_account_level_dataset(self, df: pd.DataFrame) -> None:
        """
        Validate the final account-level feature dataset.
        
        Checks:
        - Required columns present
        - No duplicate account_ids
        - Churn flag is binary (0/1 or True/False)
        - No nulls in critical columns
        """
        
        # Check required columns
        required_columns = ["account_id", "churn_flag", "tenure_months"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Check for duplicates
        if df["account_id"].duplicated().any():
            raise ValueError("Duplicate account_id found in account-level dataset.")
        
        # Validate churn_flag is binary (handles both int and bool)
        unique_values = set(df["churn_flag"].unique())
        valid_values = {0, 1, True, False}
        if not unique_values.issubset(valid_values):
            raise ValueError(
                f"churn_flag must contain only 0, 1, True, or False. "
                f"Found: {unique_values}"
            )
        
        # Check for nulls in critical columns
        for col in required_columns:
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                raise ValueError(
                    f"Null values found in critical column '{col}': {null_count} nulls"
                )
        
        # Optional: Check data shape
        if len(df) < 100:
            self.logger.warning(
                f"Dataset has only {len(df)} rows. Consider if this is sufficient."
            )
        
        self.logger.info("Account-level dataset validation passed.")
        self.logger.info(f"Dataset shape: {df.shape}")
        self.logger.info(f"Churn rate: {df['churn_flag'].mean()*100:.2f}%")


# Standalone validation functions
def validate_raw_data(datasets: Dict[str, pd.DataFrame]) -> None:
    """Convenience function for raw data validation."""
    validator = DataValidator()
    validator.validate_raw_datasets(datasets)


def validate_account_data(df: pd.DataFrame) -> None:
    """Convenience function for account-level data validation."""
    validator = DataValidator()
    validator.validate_account_level_dataset(df)


if __name__ == "__main__":
    print("Data Validator module ready.")
    print("Use: validator = DataValidator()")
    print("     validator.validate_raw_datasets(datasets)")
    print("     validator.validate_account_level_dataset(account_df)")
