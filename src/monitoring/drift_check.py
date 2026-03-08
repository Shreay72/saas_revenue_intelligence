"""
Simple Drift Monitoring Placeholder
SaaS Revenue Intelligence System

Purpose:
    - Track basic distribution shifts over time
    - Starting point for full monitoring in later weeks
"""

from typing import Dict

import pandas as pd


def compute_basic_drift_metrics(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute simple drift indicators for key columns.

    Metrics:
        - MRR mean shift
        - churn_probability mean shift (if present)
    """
    metrics: Dict[str, float] = {}

    for col in ["total_mrr", "churn_probability"]:
        if col in current_df.columns and col in baseline_df.columns:
            current_mean = float(current_df[col].mean())
            baseline_mean = float(baseline_df[col].mean())
            metrics[f"{col}_mean_current"] = current_mean
            metrics[f"{col}_mean_baseline"] = baseline_mean
            metrics[f"{col}_mean_shift"] = current_mean - baseline_mean

    return metrics


if __name__ == "__main__":
    print("Drift monitoring placeholder ready. Integrate in Week 5–6.")
