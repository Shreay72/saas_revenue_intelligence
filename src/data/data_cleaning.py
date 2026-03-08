"""
Data Cleaning Module for SaaS Revenue Risk & Retention Intelligence System

Week 1 – Data Engineering Phase
Single Row Per Account Strategy
"""

import logging
import pandas as pd
import re
from typing import Dict


class DataCleaner:
    """
    Cleans raw SaaS datasets before feature engineering.
    """

    DATE_COLUMNS = {
        "accounts": ["created_date"],
        "subscriptions": ["start_date", "end_date"],
        "feature_usage": ["usage_date"],
        "support_tickets": ["ticket_created_date"],
        "churn_events": ["churn_date"],
    }

    ID_COLUMNS = ["account_id", "subscription_id", "ticket_id"]

    def __init__(self, log_level: int = logging.INFO):
        self.logger = self._setup_logger(log_level)
        self.logger.info("DataCleaner initialized.")

    def _setup_logger(self, log_level: int) -> logging.Logger:
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

    def _to_snake_case(self, name: str) -> str:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
        s3 = re.sub("[\s\-]+", "_", s2)
        return re.sub("_+", "_", s3.lower()).strip("_")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [self._to_snake_case(col) for col in df.columns]
        return df

    def _convert_ids_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.ID_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df

    def _convert_dates(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        for col in self.DATE_COLUMNS.get(dataset_name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def _strip_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        string_cols = df.select_dtypes(include=["object"]).columns
        for col in string_cols:
            df[col] = df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
        return df

    def _remove_duplicates(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)

        if removed > 0:
            self.logger.warning(
                f"[{dataset_name}] Removed {removed} duplicate rows."
            )

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Fill numeric with median
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

        # Fill categorical with 'unknown'
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna("unknown")

        return df

    def clean_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        self.logger.info(f"[{dataset_name}] Cleaning started. Shape: {df.shape}")

        df = df.copy()

        df = self._standardize_columns(df)
        df = self._strip_string_columns(df)
        df = self._convert_ids_to_string(df)
        df = self._convert_dates(df, dataset_name)
        df = self._remove_duplicates(df, dataset_name)
        df = self._handle_missing_values(df)

        self.logger.info(
            f"[{dataset_name}] Cleaning completed. Final Shape: {df.shape}"
        )

        return df

    def clean_all(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not datasets:
            raise ValueError("Input datasets dictionary cannot be empty.")

        cleaned = {}

        for name, df in datasets.items():
            cleaned[name] = self.clean_dataset(df, name)

        self.logger.info("All datasets cleaned successfully.")

        return cleaned


def clean_saas_data(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    cleaner = DataCleaner()
    return cleaner.clean_all(datasets)


if __name__ == "__main__":
    print("DataCleaning module ready.")
