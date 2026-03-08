"""
Recommendation Engine
SaaS Revenue Intelligence System - Week 4

Combines:
    - Risk Engine scores
    - Scoring Logic tiers + types
    - Business Rules recommendations
    - Recovery calculations
    - Portfolio summary

Produces the final account intelligence table.
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

from src.risk.risk_engine import RiskEngine
from src.risk.scoring_logic import ScoringLogic
from src.recommendation.business_rules import BusinessRules
from src.utils.metrics import calculate_expected_recovery
from src.utils.helpers import calculate_priority_score

logger = logging.getLogger(__name__)


def load_rec_config(config_path: str = "config/model_config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class RecommendationEngine:
    """
    Full Revenue Intelligence Recommendation Engine.

    Produces per-account recommendations and portfolio summary.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config        = load_rec_config(config_path)
        self.risk_engine   = RiskEngine(config_path)
        self.scoring_logic = ScoringLogic(config_path)
        self.rules         = BusinessRules(config_path)
        self.success_rates = self.config.get("intervention_success_rates", {
            "VP_ESCALATION":  0.50,
            "EXECUTIVE_QBR":  0.30,
            "TAM_ASSIGNMENT": 0.40,
            "UPSELL":         0.20,
            "WINBACK":        0.25,
            "MONITOR":        0.05,
        })
        logger.info("RecommendationEngine initialized.")

    # ─────────────────────────────────────────────
    # BUILD REASON TEXT
    # ─────────────────────────────────────────────

    def _build_reason(self, row: pd.Series, rule: str) -> str:
        """Human-readable explanation for the recommendation."""
        cp   = row.get("churn_probability", 0)
        clv  = row.get("clv", 0)
        rar  = row.get("revenue_at_risk", 0)
        hs   = row.get("health_score", 0)
        es   = int(row.get("engagement_state", 0))
        sp   = int(row.get("support_pressure_signal", 0))
        cf   = int(row.get("churn_flag", 0))

        reasons = {
            "VP_ESCALATION":  (
                f"churn_probability={cp:.2f}, CLV=${clv:,.0f} — "
                f"strategic account at critical risk"
            ),
            "EXECUTIVE_QBR":  (
                f"engagement_state={es} (declining), "
                f"tenure={row.get('tenure_months', 0):.0f} months — "
                f"long-term customer disengaging"
            ),
            "TAM_ASSIGNMENT": (
                f"support_pressure_signal={sp}, "
                f"ticket_count={row.get('ticket_count', 0):.0f} — "
                f"high support burden requires dedicated resource"
            ),
            "UPSELL":         (
                f"churn_probability={cp:.2f} (low risk), "
                f"revenue_change_signal={row.get('revenue_change_signal', 0):.2f} — "
                f"healthy account with expansion potential"
            ),
            "WINBACK":        (
                f"churn_flag={cf}, CLV=${clv:,.0f} — "
                f"high-value churned account worth recovering"
            ),
            "MONITOR":        (
                f"risk_score={row.get('risk_score', 0):.1f} (low), "
                f"health_score={hs:.1f} — no immediate intervention needed"
            ),
        }
        return reasons.get(rule, "No specific reason identified.")

    # ─────────────────────────────────────────────
    # GENERATE FULL INTELLIGENCE TABLE
    # ─────────────────────────────────────────────

    def generate_intelligence(
        self,
        df: pd.DataFrame,
        previous_scores: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Produce the full account intelligence table.

        Columns:
            account_id, account_name, total_mrr,
            risk_score, risk_tier, risk_type,
            risk_velocity, velocity_flag, priority_score,
            churn_probability, revenue_at_risk, health_score, clv,
            recommended_action, action_owner, urgency,
            confidence_level, reason, expected_recovery, rule_triggered
        """
        logger.info("=" * 60)
        logger.info("Generating Account Intelligence Table")
        logger.info("=" * 60)

        df = df.copy()

        # Step 1: Compute risk scores
        risk_df = self.risk_engine.score_portfolio(df, previous_scores)
        df["risk_score"]    = risk_df["risk_score"].values
        df["risk_tier"]     = risk_df["risk_tier"].values
        df["risk_type"]     = risk_df["risk_type"].values
        df["risk_velocity"] = risk_df["risk_velocity"].values
        df["velocity_flag"] = risk_df["velocity_flag"].values
        df["priority_score"] = risk_df["priority_score"].values

        # Step 2: Evaluate business rules
        rules_df = self.rules.evaluate_portfolio(df)
        df["rule_triggered"]     = rules_df["rule_triggered"].values
        df["recommended_action"] = rules_df["recommended_action"].values
        df["action_owner"]       = rules_df["action_owner"].values
        df["urgency"]            = rules_df["urgency"].values
        df["confidence_level"]   = rules_df["confidence_level"].values

        # Step 3: Build reason text
        df["reason"] = df.apply(
            lambda r: self._build_reason(r, r["rule_triggered"]),
            axis=1,
        )

        # Step 4: Expected recovery
        df["expected_recovery"] = df.apply(
            lambda r: calculate_expected_recovery(
                revenue_at_risk=r.get("revenue_at_risk", 0.0),
                rule_triggered=r["rule_triggered"],
                success_rates=self.success_rates,
            ),
            axis=1,
        )

        # Step 5: Build final output table
        base_cols = [
            "account_id", "account_name", "total_mrr",
        ]
        risk_cols = [
            "risk_score", "risk_tier", "risk_type",
            "risk_velocity", "velocity_flag", "priority_score",
        ]
        ml_cols = [
            "churn_probability", "revenue_at_risk",
            "health_score", "clv",
        ]
        rec_cols = [
            "recommended_action", "action_owner", "urgency",
            "confidence_level", "reason", "expected_recovery",
            "rule_triggered",
        ]

        all_cols = (
            [c for c in base_cols if c in df.columns] +
            [c for c in risk_cols if c in df.columns] +
            [c for c in ml_cols if c in df.columns] +
            rec_cols
        )

        result = df[all_cols].copy()

        # Sort by priority_score descending
        if "priority_score" in result.columns:
            result = result.sort_values("priority_score", ascending=False)
        result = result.reset_index(drop=True)

        logger.info(f"Intelligence table generated: {len(result)} accounts")
        return result

    # ─────────────────────────────────────────────
    # PORTFOLIO SUMMARY
    # ─────────────────────────────────────────────

    def generate_portfolio_summary(self, intelligence_df: pd.DataFrame) -> Dict:
        """
        Aggregate portfolio-level intelligence.
        Feeds Week 6 dashboard.
        """
        df = intelligence_df

        total       = len(df)
        tier_counts = df["risk_tier"].value_counts().to_dict()
        type_counts = df["risk_type"].value_counts().to_dict()
        rule_counts = df["rule_triggered"].value_counts().to_dict()
        vel_counts  = df["velocity_flag"].value_counts().to_dict() \
            if "velocity_flag" in df.columns else {}

        total_mrr       = float(df["total_mrr"].sum()) \
            if "total_mrr" in df.columns else 0.0
        total_at_risk   = float(df["revenue_at_risk"].sum()) \
            if "revenue_at_risk" in df.columns else 0.0
        total_recovery  = float(df["expected_recovery"].sum())
        pct_recoverable = round(
            (total_recovery / total_at_risk * 100) if total_at_risk > 0 else 0, 1
        )

        top20 = df.nlargest(20, "priority_score")[
            [c for c in [
                "account_id", "account_name", "total_mrr",
                "risk_score", "risk_tier", "risk_type",
                "revenue_at_risk", "expected_recovery",
                "recommended_action", "urgency", "priority_score",
            ] if c in df.columns]
        ]

        return {
            "total_accounts":     total,
            "tier_distribution":  tier_counts,
            "type_distribution":  type_counts,
            "action_distribution": rule_counts,
            "velocity_distribution": vel_counts,
            "total_mrr":          round(total_mrr, 2),
            "total_revenue_at_risk": round(total_at_risk, 2),
            "total_recoverable":  round(total_recovery, 2),
            "pct_recoverable":    pct_recoverable,
            "top_20_accounts":    top20,
        }
