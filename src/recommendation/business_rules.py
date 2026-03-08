"""
Business Rules
SaaS Revenue Intelligence System - Week 4

Defines all 5 intervention rules + default Monitor.
Rules are evaluated in priority order — first match wins.
All thresholds configurable.
"""

import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    rule_name:    str
    priority:     int
    action:       str
    action_owner: str
    urgency:      str
    confidence:   str
    fired:        bool = False


def load_rules_config(config_path: str = "config/model_config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BusinessRules:
    """
    Rule-based recommendation engine.

    Rules evaluated top-down by priority.
    First matching rule determines the recommendation.
    Default (Monitor) fires when no rule matches.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        cfg = load_rules_config(config_path)
        self.confidence_cfg = cfg.get("intervention_confidence", {
            "VP_ESCALATION":  "HIGH",
            "EXECUTIVE_QBR":  "HIGH",
            "TAM_ASSIGNMENT": "HIGH",
            "UPSELL":         "MEDIUM",
            "WINBACK":        "MEDIUM",
            "MONITOR":        "LOW",
        })
        logger.info("BusinessRules initialized.")

    # ─────────────────────────────────────────────
    # RULE DEFINITIONS
    # ─────────────────────────────────────────────

    def _rule_vp_escalation(self, row: pd.Series, median_tickets: float) -> bool:
        """P1: High churn + high CLV → escalate to VP."""
        return (
            row.get("churn_probability", 0) > 0.80
            and row.get("clv", 0) > 500_000
        )

    def _rule_executive_qbr(self, row: pd.Series, median_tickets: float) -> bool:
        """P2: Declining engagement + long tenure → QBR."""
        return (
            int(row.get("engagement_state", 0)) == -1
            and row.get("tenure_months", 0) > 12
        )

    def _rule_tam_assignment(self, row: pd.Series, median_tickets: float) -> bool:
        """P3: High support pressure + high ticket volume → TAM."""
        return (
            int(row.get("support_pressure_signal", 0)) == 1
            and row.get("ticket_count", 0) >= median_tickets
        )

    def _rule_upsell(self, row: pd.Series, median_tickets: float) -> bool:
        """P4: Low churn + positive revenue signal → upsell."""
        return (
            row.get("churn_probability", 1) < 0.20
            and float(row.get("revenue_change_signal", 0)) > 0
        )

    def _rule_winback(self, row: pd.Series, median_tickets: float) -> bool:
        """P5: Churned account with high CLV → win-back."""
        return (
            int(row.get("churn_flag", 0)) == 1
            and row.get("clv", 0) > 100_000
        )

    # ─────────────────────────────────────────────
    # MONITOR SUB-ACTIONS
    # ─────────────────────────────────────────────

    def _monitor_sub_action(self, row: pd.Series, median_usage: float) -> str:
        """
        Low-risk accounts are not ignored.
        Sub-action depends on account profile.
        """
        if row.get("unique_features_used", 0) < median_usage:
            return "Send product adoption email"
        elif row.get("total_usage", 0) < median_usage:
            return "Send automated usage tips"
        else:
            return "Schedule periodic check-in"

    # ─────────────────────────────────────────────
    # EVALUATE RULES FOR ONE ACCOUNT
    # ─────────────────────────────────────────────

    def evaluate(
        self,
        row: pd.Series,
        median_tickets: float,
        median_usage: float,
    ) -> Dict:
        """
        Evaluate all rules for one account.
        Returns the highest-priority matching rule result.
        """
        rules = [
            (1, "VP_ESCALATION",  self._rule_vp_escalation,
             "Escalate to VP of Customer Success",
             "VP_CUSTOMER_SUCCESS",      "IMMEDIATE"),
            (2, "EXECUTIVE_QBR",  self._rule_executive_qbr,
             "Schedule Executive QBR",
             "ACCOUNT_MANAGER",          "HIGH"),
            (3, "TAM_ASSIGNMENT", self._rule_tam_assignment,
             "Assign dedicated Technical Account Manager",
             "TECHNICAL_ACCOUNT_MANAGER","HIGH"),
            (4, "UPSELL",         self._rule_upsell,
             "Initiate upsell conversation",
             "ACCOUNT_MANAGER",          "MEDIUM"),
            (5, "WINBACK",        self._rule_winback,
             "Trigger win-back campaign",
             "CSM",                      "HIGH"),
        ]

        for priority, rule_key, rule_fn, action, owner, urgency in rules:
            if rule_fn(row, median_tickets):
                return {
                    "rule_triggered": rule_key,
                    "priority":       priority,
                    "recommended_action": action,
                    "action_owner":   owner,
                    "urgency":        urgency,
                    "confidence_level": self.confidence_cfg.get(rule_key, "MEDIUM"),
                }

        # Default — Monitor
        sub_action = self._monitor_sub_action(row, median_usage)
        return {
            "rule_triggered":     "MONITOR",
            "priority":           99,
            "recommended_action": sub_action,
            "action_owner":       "CSM",
            "urgency":            "LOW",
            "confidence_level":   self.confidence_cfg.get("MONITOR", "LOW"),
        }

    # ─────────────────────────────────────────────
    # BATCH EVALUATION
    # ─────────────────────────────────────────────

    def evaluate_portfolio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate rules for all accounts.
        Returns DataFrame with rule columns.
        """
        median_tickets = float(df["ticket_count"].median()) \
            if "ticket_count" in df.columns else 0.0
        median_usage = float(df["unique_features_used"].median()) \
            if "unique_features_used" in df.columns else 0.0

        results = df.apply(
            lambda row: self.evaluate(row, median_tickets, median_usage),
            axis=1,
        )

        return pd.DataFrame(list(results), index=df.index)
