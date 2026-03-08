"""
Revenue Intelligence Pipeline
SaaS Revenue Intelligence System - Week 3 (Revised)
"""

import sys
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.metrics import (
    calculate_clv,
    calculate_revenue_at_risk,
    calculate_health_score,
)
from src.models.clv_model import CLVModel

logger = get_logger(__name__)


def load_revenue_pipeline(output_dir: str = "models/revenue") -> "RevenuePipeline":
    pipeline = RevenuePipeline(output_dir=output_dir)
    pipeline.load()
    return pipeline


class RevenuePipeline:
    """
    Revenue Intelligence Pipeline.

    Components:
        - MRR prediction (ML model)
        - CLV calculation (deterministic)
        - Revenue at risk (expected value)
        - Health scoring
        - Account intelligence report
    """

    def __init__(self, output_dir: str = "models/revenue"):
        self.output_dir   = Path(output_dir)
        self.mrr_model    = None
        self.clv_model    = None
        self.feature_cols = None
        self.metadata: Dict = {}
        self.is_loaded    = False

    # ─────────────────────────────────────────────
    # LOAD MODELS
    # ─────────────────────────────────────────────

    def load(self) -> None:
        mrr_path = self.output_dir / "revenue_model_v1.pkl"
        if not mrr_path.exists():
            raise FileNotFoundError(f"MRR model not found: {mrr_path}")

        artifact = joblib.load(mrr_path)
        self.mrr_model    = artifact["model"]
        self.feature_cols = artifact["feature_cols"]
        logger.info(f"✅ MRR model loaded: {mrr_path}")

        self.clv_model = CLVModel.load(str(self.output_dir))

        meta_path = self.output_dir / "revenue_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        logger.info("✅ Revenue metadata loaded.")

        self.is_loaded = True
        logger.info("✅ Revenue pipeline loaded.")

    # ─────────────────────────────────────────────
    # PREDICTIONS
    # ─────────────────────────────────────────────

    def predict_mrr(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        df = df.copy()
        # Ensure state/signal features exist
        for col in ["engagement_state", "revenue_change_signal", "support_pressure_signal"]:
            if col not in df.columns:
                df[col] = 0

        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].fillna(0)
        X = X.reindex(columns=self.feature_cols, fill_value=0)

        predictions = self.mrr_model.predict(X)

        # MRR cannot be negative — clip to 0
        return np.clip(predictions, 0, None)

    def predict_clv(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        return self.clv_model.predict(df)

    def calculate_revenue_at_risk(self, df: pd.DataFrame) -> np.ndarray:
        mrr = df["total_mrr"].values if "total_mrr" in df.columns else np.zeros(len(df))

        if "churn_probability" in df.columns:
            churn_prob = df["churn_probability"].values
        elif "churn_flag" in df.columns:
            churn_prob = df["churn_flag"].astype(float).values
        else:
            churn_prob = np.full(len(df), 0.10)

        return np.array(
            [calculate_revenue_at_risk(m, c) for m, c in zip(mrr, churn_prob)]
        )

    def calculate_health_scores(self, df: pd.DataFrame) -> np.ndarray:
        scores = []
        for _, row in df.iterrows():
            score = calculate_health_score(
                engagement_score=row.get("engagement_score", 50),
                support_risk_score=row.get("support_risk_score", 50),
                churn_probability=row.get(
                    "churn_probability",
                    float(row.get("churn_flag", 0)) if "churn_flag" in row else 0.10,
                ),
                tenure_months=row.get("tenure_months", 12),
                auto_renew_ratio=row.get("auto_renew_ratio", 0.5),
            )
            scores.append(score)
        return np.array(scores)

    # ─────────────────────────────────────────────
    # ACCOUNT INTELLIGENCE REPORT
    # ─────────────────────────────────────────────

    def generate_account_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating revenue intelligence...")

        base_cols = ["account_id", "account_name", "total_mrr"]
        base_cols = [c for c in base_cols if c in df.columns]
        result = df[base_cols].copy()

        result["predicted_clv"]    = self.predict_clv(df)
        result["revenue_at_risk"]  = self.calculate_revenue_at_risk(df)
        result["health_score"]     = self.calculate_health_scores(df)

        if "churn_probability" in df.columns:
            result["churn_probability"] = df["churn_probability"].values
        elif "churn_flag" in df.columns:
            result["churn_probability"] = df["churn_flag"].astype(float).values

        clv_series = pd.Series(result["predicted_clv"].values)
        result["clv_segment"]  = self.clv_model._assign_segment(clv_series).values
        result["priority_tier"] = result["health_score"].apply(self._priority_tier)

        return result.sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)

    def _priority_tier(self, health_score: float) -> str:
        if health_score < 30:
            return "P1 - Critical"
        if health_score < 50:
            return "P2 - High"
        if health_score < 70:
            return "P3 - Medium"
        return "P4 - Low"

    def get_pipeline_info(self) -> Dict:
        return {
            "mrr_model_type": self.metadata.get("model_type", "GradientBoosting"),
            "clv_model_type": "deterministic_formula",
            "feature_count":  len(self.feature_cols) if self.feature_cols else 0,
            "version":        self.metadata.get("version", "v3"),
        }
