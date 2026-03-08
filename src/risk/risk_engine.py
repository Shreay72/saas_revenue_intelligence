"""
Risk Engine
SaaS Revenue Intelligence System - Week 4

Computes composite risk score for every account using:
    - churn_probability  (Week 2 output)
    - revenue_at_risk    (Week 3 output)
    - health_score       (Week 3 output)

All weights and thresholds are read from config/model_config.yaml.
"""

import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.metrics import calculate_composite_risk_score
from src.utils.helpers import (
    assign_risk_tier,
    assign_risk_type,
    assign_velocity_flag,
    calculate_priority_score,
)

logger = logging.getLogger(__name__)


def load_risk_config(config_path: str = "config/model_config.yaml") -> Dict:
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    return full_config


class RiskEngine:
    """
    Composite Risk Scoring Engine.

    Reads all weights, thresholds, and normalization parameters
    from config/model_config.yaml — no hardcoded values.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config     = load_risk_config(config_path)
        self.re_config  = self.config.get("risk_engine", {})
        self.weights    = self.re_config.get("weights", {
            "churn_probability": 0.40,
            "revenue_at_risk":   0.35,
            "health_score":      0.25,
        })
        self.norm_cfg   = self.re_config.get("normalization", {})
        self.tier_cfg   = self.re_config.get("tier_thresholds", {
            "critical": 75, "high": 60, "medium": 35
        })
        self.vel_cfg    = self.re_config.get("velocity_thresholds", {
            "accelerating": 15, "improving": -15
        })
        self.p95_cap    = self.norm_cfg.get("revenue_percentile_cap", 95)
        self.p90_override = self.norm_cfg.get("revenue_percentile_override", 90)

        logger.info("RiskEngine initialized.")
        logger.info(f"  Weights: {self.weights}")
        logger.info(f"  Tier thresholds: {self.tier_cfg}")

    # ─────────────────────────────────────────────
    # CORE SCORING
    # ─────────────────────────────────────────────

    def _compute_percentiles(self, df: pd.DataFrame) -> Dict[str, float]:
        """Pre-compute all portfolio-level percentiles needed for scoring."""
        rev  = df["revenue_at_risk"]
        clv  = df["clv"] if "clv" in df.columns else pd.Series([0.0] * len(df))

        p95_rev = max(float(rev.quantile(self.p95_cap / 100)), 1.0)
        p90_rev = max(float(rev.quantile(self.p90_override / 100)), 1.0)
        p75_rev = float(rev.quantile(0.75))
        p75_clv = float(clv.quantile(0.75))

        return {
            "p95_revenue":     p95_rev,
            "p90_revenue":     p90_rev,
            "p75_revenue":     p75_rev,
            "p75_clv":         p75_clv,
        }

    def score_account(
        self,
        churn_probability: float,
        revenue_at_risk: float,
        health_score: float,
        p95_revenue: float,
    ) -> float:
        """Compute composite risk score for a single account."""
        return calculate_composite_risk_score(
            churn_probability=churn_probability,
            revenue_at_risk=revenue_at_risk,
            health_score=health_score,
            p95_revenue=p95_revenue,
            weights=self.weights,
        )

    # ─────────────────────────────────────────────
    # BATCH SCORING
    # ─────────────────────────────────────────────

    def score_portfolio(
        self,
        df: pd.DataFrame,
        previous_scores: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Score all accounts in portfolio.

        Args:
            df:               account-level features dataframe
            previous_scores:  risk scores from previous run (for velocity)
                              Pass None for first run — velocity defaults to 0.

        Returns:
            DataFrame with risk_score, risk_tier, risk_type,
            risk_velocity, velocity_flag, priority_score
        """
        logger.info(f"Scoring portfolio: {len(df)} accounts...")

        # Pre-compute portfolio percentiles
        pcts = self._compute_percentiles(df)
        logger.info(
            f"  P95 revenue: ${pcts['p95_revenue']:,.2f} | "
            f"  P90 revenue: ${pcts['p90_revenue']:,.2f}"
        )

        result = df[["account_id"]].copy() if "account_id" in df.columns \
                 else pd.DataFrame(index=df.index)

        # ── Risk Scores ──
        result["risk_score"] = df.apply(
            lambda r: self.score_account(
                churn_probability=r.get("churn_probability", 0.0),
                revenue_at_risk=r.get("revenue_at_risk", 0.0),
                health_score=r.get("health_score", 50.0),
                p95_revenue=pcts["p95_revenue"],
            ),
            axis=1,
        )

        # ── Risk Tier ──
        result["risk_tier"] = result.apply(
            lambda r: assign_risk_tier(
                risk_score=r["risk_score"],
                revenue_at_risk=df.loc[r.name, "revenue_at_risk"]
                    if "revenue_at_risk" in df.columns else 0.0,
                p90_revenue=pcts["p90_revenue"],
                thresholds=self.tier_cfg,
            ),
            axis=1,
        )

        # ── Risk Type ──
        result["risk_type"] = df.apply(
            lambda r: assign_risk_type(
                clv=r.get("clv", 0.0),
                churn_probability=r.get("churn_probability", 0.0),
                revenue_at_risk=r.get("revenue_at_risk", 0.0),
                engagement_state=int(r.get("engagement_state", 0)),
                support_pressure_signal=int(r.get("support_pressure_signal", 0)),
                clv_p75=pcts["p75_clv"],
                rev_risk_p75=pcts["p75_revenue"],
            ),
            axis=1,
        )

        # ── Risk Velocity ──
        if previous_scores is not None:
            result["risk_velocity"] = (
                result["risk_score"].values - previous_scores.values
            ).round(2)
        else:
            result["risk_velocity"] = 0.0

        result["velocity_flag"] = result["risk_velocity"].apply(
            lambda v: assign_velocity_flag(v, thresholds=self.vel_cfg)
        )

        # ── Priority Score ──
        result["priority_score"] = df.apply(
            lambda r: calculate_priority_score(
                risk_score=result.loc[r.name, "risk_score"],
                revenue_at_risk=r.get("revenue_at_risk", 0.0),
            ),
            axis=1,
        )

        logger.info("Portfolio scoring complete.")
        logger.info(
            f"  CRITICAL: {(result['risk_tier']=='CRITICAL').sum()} | "
            f"  HIGH: {(result['risk_tier']=='HIGH').sum()} | "
            f"  MEDIUM: {(result['risk_tier']=='MEDIUM').sum()} | "
            f"  LOW: {(result['risk_tier']=='LOW').sum()}"
        )

        return result
