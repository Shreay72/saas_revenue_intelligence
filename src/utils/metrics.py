"""
Shared Metrics & Business Calculations
SaaS Revenue Intelligence System
Week 3 + Week 4
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or (isinstance(denominator, float) and np.isnan(denominator)):
        return default
    return numerator / denominator


def normalize_series(series: pd.Series) -> pd.Series:
    if series.std() == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def calculate_health_score(
    engagement_score: float,
    support_risk_score: float,
    churn_probability: float,
    tenure_months: float,
    auto_renew_ratio: float
) -> float:
    engagement_norm = min(max(engagement_score, 0), 100) / 100
    support_health  = 1.0 - min(max(support_risk_score, 0), 100) / 100
    churn_health    = 1.0 - min(max(churn_probability, 0.0), 1.0)
    tenure_norm     = min(tenure_months / 36, 1.0)
    auto_renew_norm = min(max(auto_renew_ratio, 0.0), 1.0)

    score = (
        0.30 * engagement_norm +
        0.25 * support_health +
        0.25 * churn_health +
        0.10 * tenure_norm +
        0.10 * auto_renew_norm
    ) * 100

    return round(score, 2)


# ─────────────────────────────────────────────
# CLV — Deterministic Formula
# ─────────────────────────────────────────────

def calculate_clv(
    mrr: float,
    churn_probability: float,
    gross_margin: float = 0.70,
    discount_rate: float = 0.10
) -> float:
    """
    Deterministic SaaS CLV:
        CLV = (MRR × gross_margin) / (monthly_churn + monthly_discount)
    """
    if mrr <= 0:
        return 0.0
    churn_probability = min(max(churn_probability, 0.001), 0.999)
    monthly_churn    = churn_probability / 12
    monthly_discount = discount_rate / 12
    return round((mrr * gross_margin) / (monthly_churn + monthly_discount), 2)


# ─────────────────────────────────────────────
# Revenue at Risk — Expected Value
# ─────────────────────────────────────────────

def calculate_revenue_at_risk(mrr: float, churn_probability: float) -> float:
    """Revenue at Risk = MRR × churn_probability"""
    return round(mrr * churn_probability, 2)


def calculate_expansion_revenue(
    mrr: float,
    plan_tier: str,
    upgrade_probability: float = 0.10
) -> float:
    tier_multipliers = {
        "starter":    1.5,
        "basic":      1.5,
        "pro":        1.25,
        "enterprise": 1.1,
    }
    multiplier = tier_multipliers.get(plan_tier.lower(), 1.2)
    return round(mrr * (multiplier - 1) * upgrade_probability, 2)


def calculate_portfolio_metrics(df: pd.DataFrame) -> Dict[str, float]:
    total_accounts = len(df)
    churned        = int(df["churn_flag"].sum()) if "churn_flag" in df.columns else 0
    active         = total_accounts - churned
    churn_rate     = safe_divide(churned, total_accounts)

    total_mrr = float(df["total_mrr"].sum()) if "total_mrr" in df.columns else 0.0
    avg_mrr   = safe_divide(total_mrr, total_accounts)
    total_arr = total_mrr * 12

    metrics = {
        "total_accounts":   total_accounts,
        "active_accounts":  active,
        "churned_accounts": churned,
        "churn_rate":       round(churn_rate, 4),
        "total_mrr":        round(total_mrr, 2),
        "total_arr":        round(total_arr, 2),
        "avg_mrr":          round(avg_mrr, 2),
    }

    if "churn_probability" in df.columns:
        total_risk = df.apply(
            lambda r: calculate_revenue_at_risk(r["total_mrr"], r["churn_probability"]),
            axis=1,
        ).sum()
        metrics["total_revenue_at_risk"] = round(total_risk, 2)

    if "clv" in df.columns:
        metrics["total_clv"]  = round(float(df["clv"].sum()), 2)
        metrics["avg_clv"]    = round(float(df["clv"].mean()), 2)
        metrics["median_clv"] = round(float(df["clv"].median()), 2)

    return metrics


# ─────────────────────────────────────────────
# WEEK 4 — Composite Risk Score
# ─────────────────────────────────────────────

def calculate_composite_risk_score(
    churn_probability: float,
    revenue_at_risk: float,
    health_score: float,
    p95_revenue: float,
    weights: Dict[str, float] = None,
) -> float:
    """
    Composite risk score (0–100).

    Formula:
        risk = (w1 × churn_prob
              + w2 × clip(rev_at_risk / p95, 0, 1)
              + w3 × (1 - health_score/100)) × 100

    Revenue normalization uses p95 as ceiling (outlier-safe).
    p95 is floored at 1.0 to prevent division by zero.
    Result is always clipped to [0, 100].
    """
    if weights is None:
        weights = {
            "churn_probability": 0.40,
            "revenue_at_risk":   0.35,
            "health_score":      0.25,
        }

    # Safe p95 floor
    p95_safe = max(float(p95_revenue), 1.0)

    churn_norm  = min(max(float(churn_probability), 0.0), 1.0)
    rev_norm    = float(np.clip(revenue_at_risk / p95_safe, 0.0, 1.0))
    health_norm = 1.0 - min(max(float(health_score), 0.0), 100.0) / 100.0

    raw = (
        weights["churn_probability"] * churn_norm +
        weights["revenue_at_risk"]   * rev_norm +
        weights["health_score"]      * health_norm
    ) * 100

    return round(float(np.clip(raw, 0.0, 100.0)), 2)


# ─────────────────────────────────────────────
# WEEK 4 — Expected Recovery
# ─────────────────────────────────────────────

def calculate_expected_recovery(
    revenue_at_risk: float,
    rule_triggered: str,
    success_rates: Dict[str, float] = None,
) -> float:
    """
    Expected revenue recovery from an intervention.

        expected_recovery = revenue_at_risk × success_rate
    """
    if success_rates is None:
        success_rates = {
            "VP_ESCALATION":  0.50,
            "EXECUTIVE_QBR":  0.30,
            "TAM_ASSIGNMENT": 0.40,
            "UPSELL":         0.20,
            "WINBACK":        0.25,
            "MONITOR":        0.05,
        }
    rate = success_rates.get(rule_triggered.upper(), 0.05)
    return round(float(revenue_at_risk) * rate, 2)
