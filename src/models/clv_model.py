"""
Customer Lifetime Value (CLV) — Deterministic Model
SaaS Revenue Intelligence System - Week 3 (Revised)

Design:
    CLV is computed via analytical SaaS formula, not ML.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from src.utils.metrics import calculate_clv

logger = logging.getLogger(__name__)


class CLVModel:
    """
    Deterministic Customer Lifetime Value calculator.

    Uses:
        CLV = (MRR × gross_margin) / (monthly_churn + monthly_discount)
    """

    def __init__(
        self,
        gross_margin: float = 0.70,
        discount_rate: float = 0.10,
        output_dir: str = "models/revenue",
    ):
        self.gross_margin  = gross_margin
        self.discount_rate = discount_rate
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_fitted     = True  # deterministic, no training step
        self.feature_cols  = ["total_mrr", "churn_probability"]

        logger.info("CLVModel initialized (deterministic formula).")

    # ─────────────────────────────────────────────
    # FIT — no-op (used only to compute metrics)
    # ─────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> dict:
        logger.info("=" * 60)
        logger.info("CLV Model — Deterministic Calculation")
        logger.info("=" * 60)

        clv_values = self._compute_clv_series(df)

        metrics = {
            "method":        "deterministic_formula",
            "formula":       "CLV = (MRR × gross_margin) / (monthly_churn + monthly_discount)",
            "gross_margin":  self.gross_margin,
            "discount_rate": self.discount_rate,
            "clv_min":       round(float(clv_values.min()),    2),
            "clv_max":       round(float(clv_values.max()),    2),
            "clv_mean":      round(float(clv_values.mean()),   2),
            "clv_median":    round(float(clv_values.median()), 2),
        }

        logger.info("\n📊 CLV Distribution:")
        logger.info(f"  Min:    ${metrics['clv_min']:,.2f}")
        logger.info(f"  Max:    ${metrics['clv_max']:,.2f}")
        logger.info(f"  Mean:   ${metrics['clv_mean']:,.2f}")
        logger.info(f"  Median: ${metrics['clv_median']:,.2f}")
        logger.info("  ✅ No overfitting risk (formula-based)")

        return metrics

    # ─────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────

    def _compute_clv_series(self, df: pd.DataFrame) -> pd.Series:
        mrr = df["total_mrr"] if "total_mrr" in df.columns else pd.Series(
            np.zeros(len(df)), index=df.index
        )

        if "churn_probability" in df.columns:
            churn_prob = df["churn_probability"]
        elif "churn_flag" in df.columns:
            churn_prob = df["churn_flag"].astype(float)  # 0 or 1
        else:
            churn_prob = pd.Series(np.full(len(df), 0.10), index=df.index)

        clv_values = [
            calculate_clv(
                mrr=float(m),
                churn_probability=float(c),
                gross_margin=self.gross_margin,
                discount_rate=self.discount_rate,
            )
            for m, c in zip(mrr, churn_prob)
        ]
        return pd.Series(clv_values, index=df.index)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._compute_clv_series(df).values

    def predict_with_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        clv_predictions = self._compute_clv_series(df)

        results = pd.DataFrame(
            {
                "account_id":   df["account_id"] if "account_id" in df.columns else range(len(df)),
                "account_name": df.get("account_name", "Unknown"),
                "total_mrr":    df.get("total_mrr", 0),
                "predicted_clv": clv_predictions.round(2),
            }
        )

        results["clv_segment"]   = self._assign_segment(results["predicted_clv"])
        results["predicted_arr"] = results["total_mrr"] * 12

        return results.sort_values("predicted_clv", ascending=False).reset_index(drop=True)

    def _assign_segment(self, clv_series: pd.Series) -> pd.Series:
        p80 = clv_series.quantile(0.80)
        p60 = clv_series.quantile(0.60)
        p40 = clv_series.quantile(0.40)

        def seg(v: float) -> str:
            if v >= p80:
                return "Champions"
            if v >= p60:
                return "Loyalists"
            if v >= p40:
                return "At-Risk"
            return "Lost Causes"

        return clv_series.apply(seg)

    # ─────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────

    def save(self, metrics: dict = None) -> None:
        metadata = {
            "model_name":    "CLV Model (Deterministic)",
            "model_type":    "deterministic_formula",
            "formula":       "CLV = (MRR * gross_margin) / (monthly_churn + monthly_discount)",
            "train_date":    datetime.now().isoformat(),
            "feature_cols":  self.feature_cols,
            "gross_margin":  self.gross_margin,
            "discount_rate": self.discount_rate,
            "metrics":       metrics or {},
            "version":       "v2",
        }

        meta_path = self.output_dir / "clv_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ CLV metadata saved: {meta_path}")

        model_path = self.output_dir / "clv_model_v1.pkl"
        joblib.dump(
            {"type": "deterministic", "gross_margin": self.gross_margin,
             "discount_rate": self.discount_rate},
            model_path,
        )
        logger.info(f"✅ CLV model saved: {model_path}")

    @classmethod
    def load(cls, output_dir: str = "models/revenue") -> "CLVModel":
        meta_path = Path(output_dir) / "clv_metadata.json"
        gross_margin  = 0.70
        discount_rate = 0.10

        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            gross_margin  = metadata.get("gross_margin", 0.70)
            discount_rate = metadata.get("discount_rate", 0.10)

        instance = cls(
            gross_margin=gross_margin,
            discount_rate=discount_rate,
            output_dir=output_dir,
        )
        logger.info("✅ CLV model loaded.")
        return instance


if __name__ == "__main__":
    print("CLV Model (Deterministic) ready.")
