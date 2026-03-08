"""
Scoring Logic
SaaS Revenue Intelligence System - Week 4

Defines risk tier boundaries and risk type classification.
All thresholds configurable via config/model_config.yaml.
"""

import sys
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

VALID_TIERS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
VALID_RISK_TYPES = {
    "Strategic Risk",
    "Revenue Risk",
    "Support Risk",
    "Usage Risk",
    "Composite Risk",
    "Healthy",
}


def load_scoring_config(config_path: str = "config/model_config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ScoringLogic:
    """
    Handles risk tier assignment and risk type classification
    for the full account portfolio.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        cfg = load_scoring_config(config_path)
        re  = cfg.get("risk_engine", {})

        self.tier_thresholds = re.get("tier_thresholds", {
            "critical": 75, "high": 60, "medium": 35
        })
        self.p90_override = re.get("normalization", {}).get(
            "revenue_percentile_override", 90
        )
        logger.info("ScoringLogic initialized.")

    # ─────────────────────────────────────────────
    # TIER ASSIGNMENT
    # ─────────────────────────────────────────────

    def assign_tier(
        self,
        risk_score: float,
        revenue_at_risk: float,
        p90_revenue: float,
    ) -> str:
        """
        Assign risk tier.

        Revenue override: top 10% revenue accounts are always CRITICAL
        regardless of risk score — large accounts cannot be ignored.
        """
        if risk_score >= self.tier_thresholds["critical"] \
                or revenue_at_risk >= p90_revenue:
            return "CRITICAL"
        elif risk_score >= self.tier_thresholds["high"]:
            return "HIGH"
        elif risk_score >= self.tier_thresholds["medium"]:
            return "MEDIUM"
        else:
            return "LOW"

    # ─────────────────────────────────────────────
    # RISK TYPE CLASSIFICATION
    # ─────────────────────────────────────────────

    def classify_risk_type(
        self,
        clv: float,
        churn_probability: float,
        revenue_at_risk: float,
        engagement_state: int,
        support_pressure_signal: int,
        clv_p75: float,
        rev_risk_p75: float,
    ) -> str:
        """
        Deterministic risk type (first match wins):
            1. Strategic Risk  — high CLV + high churn
            2. Revenue Risk    — high revenue at risk
            3. Support Risk    — support pressure active
            4. Usage Risk      — negative engagement state
            5. Composite Risk  — 2+ non-strategic signals
            6. Healthy         — no signals
        """
        strategic = (clv >= clv_p75) and (churn_probability > 0.70)
        revenue   = revenue_at_risk >= rev_risk_p75
        support   = support_pressure_signal == 1
        usage     = engagement_state == -1

        if strategic:
            return "Strategic Risk"
        if revenue:
            return "Revenue Risk"
        if support:
            return "Support Risk"
        if usage:
            return "Usage Risk"
        if sum([revenue, support, usage]) >= 2:
            return "Composite Risk"
        return "Healthy"

    # ─────────────────────────────────────────────
    # PORTFOLIO-LEVEL SCORING
    # ─────────────────────────────────────────────

    def apply_to_portfolio(
        self,
        df: pd.DataFrame,
        risk_scores: pd.Series,
    ) -> pd.DataFrame:
        """
        Apply tier and risk type classification to full portfolio.

        Args:
            df:          account-level features (needs clv, churn_probability,
                         revenue_at_risk, engagement_state,
                         support_pressure_signal)
            risk_scores: pre-computed risk scores from RiskEngine

        Returns:
            DataFrame with risk_tier and risk_type columns
        """
        rev  = df["revenue_at_risk"] if "revenue_at_risk" in df.columns \
               else pd.Series([0.0] * len(df))
        clv  = df["clv"] if "clv" in df.columns \
               else pd.Series([0.0] * len(df))

        p90_rev  = max(float(rev.quantile(self.p90_override / 100)), 1.0)
        p75_rev  = float(rev.quantile(0.75))
        p75_clv  = float(clv.quantile(0.75))

        result = pd.DataFrame(index=df.index)

        result["risk_tier"] = [
            self.assign_tier(
                risk_score=float(risk_scores.iloc[i]),
                revenue_at_risk=float(rev.iloc[i]),
                p90_revenue=p90_rev,
            )
            for i in range(len(df))
        ]

        result["risk_type"] = [
            self.classify_risk_type(
                clv=float(clv.iloc[i]),
                churn_probability=float(df["churn_probability"].iloc[i])
                    if "churn_probability" in df.columns else 0.0,
                revenue_at_risk=float(rev.iloc[i]),
                engagement_state=int(df["engagement_state"].iloc[i])
                    if "engagement_state" in df.columns else 0,
                support_pressure_signal=int(df["support_pressure_signal"].iloc[i])
                    if "support_pressure_signal" in df.columns else 0,
                clv_p75=p75_clv,
                rev_risk_p75=p75_rev,
            )
            for i in range(len(df))
        ]

        return result
