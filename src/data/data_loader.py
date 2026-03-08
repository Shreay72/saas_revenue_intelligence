"""
Data Loader Module for SaaS Revenue Risk & Retention Intelligence System

This module provides functionality to load raw SaaS datasets from CSV files
with validation, error handling, and logging capabilities.

Author: Data Engineering Team
Date: 2026-02-11
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict


class DataLoader:
    """
    DataLoader class for loading multiple SaaS datasets from CSV files.
    """

    DATASET_FILES = {
        "accounts": "accounts.csv",
        "subscriptions": "subscriptions.csv",
        "feature_usage": "feature_usage.csv",
        "support_tickets": "support_tickets.csv",
        "churn_events": "churn_events.csv",
    }

    def __init__(self, base_path: str, log_level: int = logging.INFO):
        if not base_path:
            raise ValueError("base_path cannot be None or empty")

        self.base_path = Path(base_path)

        if not self.base_path.exists():
            raise FileNotFoundError(
                f"Base path does not exist: {self.base_path.resolve()}"
            )

        if not self.base_path.is_dir():
            raise NotADirectoryError(
                f"Base path is not a directory: {self.base_path.resolve()}"
            )

        self.logger = self._setup_logger(log_level)
        self.logger.info(f"DataLoader initialized at: {self.base_path.resolve()}")

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

    def _validate_file(self, file_path: Path, dataset_name: str) -> None:
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(
                f"Required dataset '{dataset_name}' not found at: {file_path.resolve()}"
            )

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        if dataset_name not in self.DATASET_FILES:
            raise ValueError(
                f"Invalid dataset name '{dataset_name}'. "
                f"Expected one of: {list(self.DATASET_FILES.keys())}"
            )

        file_path = self.base_path / self.DATASET_FILES[dataset_name]
        self._validate_file(file_path, dataset_name)

        try:
            self.logger.info(f"Loading dataset: {dataset_name}")
            df = pd.read_csv(file_path, low_memory=False)

            self.logger.info(
                f"{dataset_name} loaded successfully | Shape: {df.shape}"
            )

            if df.empty:
                self.logger.warning(f"Dataset '{dataset_name}' is empty.")

            return df

        except Exception as e:
            self.logger.error(f"Error loading dataset '{dataset_name}': {str(e)}")
            raise

    def load_all(self) -> Dict[str, pd.DataFrame]:
        self.logger.info("Starting bulk dataset loading...")

        datasets = {}

        for dataset_name in self.DATASET_FILES:
            datasets[dataset_name] = self.load_dataset(dataset_name)

        total_memory_mb = self._calculate_total_memory(datasets)

        self.logger.info(
            f"All datasets loaded successfully | Total memory: {total_memory_mb:.2f} MB"
        )

        return datasets

    @staticmethod
    def _calculate_total_memory(datasets: Dict[str, pd.DataFrame]) -> float:
        total_bytes = sum(
            df.memory_usage(deep=True).sum() for df in datasets.values()
        )
        return total_bytes / (1024 * 1024)

    def get_summary(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        summary = []

        for name, df in datasets.items():
            summary.append(
                {
                    "dataset": name,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "memory_mb": round(
                        df.memory_usage(deep=True).sum() / (1024 * 1024), 2
                    ),
                }
            )

        return pd.DataFrame(summary)


def load_saas_data(base_path: str) -> Dict[str, pd.DataFrame]:
    loader = DataLoader(base_path=base_path)
    return loader.load_all()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    print(f"\nLoading SaaS datasets from: {path}")
    print("=" * 60)

    try:
        loader = DataLoader(base_path=path)
        datasets = loader.load_all()
        summary = loader.get_summary(datasets)

        print("\nDATASET SUMMARY")
        print("=" * 60)
        print(summary.to_string(index=False))

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
