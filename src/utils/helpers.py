"""
Shared Helper Utilities
SaaS Revenue Intelligence System
Week 1 + Week 4
"""

import numpy as np
import pandas as pd
from typing import Dict


# ─────────────────────────────────────────────
# MATH UTILITIES
# ─────────────────────────────────────────────

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or (isinstance(denominator, float) and np.isnan(denominator)):
        return default
    return numerator / denominator


def normalize_series(series: pd.Series) -> pd.Series:
    if series.std() == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def format_currency(value: float, symbol: str = "$") -> str:
    return f"{symbol}{value:,.2f}"


# ─────────────────────────────────────────────
# ACCOUNT HEALTH SCORE
# ─────────────────────────────────────────────

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

    return round((
        0.30 * engagement_norm +
        0.25 * support_health +
        0.25 * churn_health +
        0.10 * tenure_norm +
        0.10 * auto_renew_norm
    ) * 100, 2)


# ─────────────────────────────────────────────
# WEEK 4 — RISK TIER ASSIGNMENT
# ─────────────────────────────────────────────

def assign_risk_tier(
    risk_score: float,
    revenue_at_risk: float,
    p90_revenue: float,
    thresholds: Dict[str, float] = None,
) -> str:
    """
    Assign risk tier based on score + revenue override.

    Top 10% revenue accounts are always CRITICAL regardless of score.
    Thresholds are read from config — not hardcoded here.
    """
    if thresholds is None:
        thresholds = {"critical": 75, "high": 60, "medium": 35}

    if risk_score >= thresholds["critical"] or revenue_at_risk >= p90_revenue:
        return "CRITICAL"
    elif risk_score >= thresholds["high"]:
        return "HIGH"
    elif risk_score >= thresholds["medium"]:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────────
# WEEK 4 — RISK TYPE CLASSIFIER
# ─────────────────────────────────────────────

def assign_risk_type(
    clv: float,
    churn_probability: float,
    revenue_at_risk: float,
    engagement_state: int,
    support_pressure_signal: int,
    clv_p75: float,
    rev_risk_p75: float,
) -> str:
    """
    Deterministic risk type classification.

    Precedence (first match wins):
        1. Strategic Risk  — high CLV + high churn
        2. Revenue Risk    — high revenue at risk
        3. Support Risk    — support pressure signal
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

    non_strategic_count = sum([revenue, support, usage])
    if non_strategic_count >= 2:
        return "Composite Risk"

    return "Healthy"


# ─────────────────────────────────────────────
# WEEK 4 — VELOCITY FLAG
# ─────────────────────────────────────────────

def assign_velocity_flag(
    velocity: float,
    thresholds: Dict[str, float] = None,
) -> str:
    """
    Classify risk velocity direction.

    Thresholds are always read from config.
    Defaults match model_config.yaml values.
    """
    if thresholds is None:
        thresholds = {"accelerating": 15, "improving": -15}

    if velocity >= thresholds["accelerating"]:
        return "ACCELERATING"
    elif velocity <= thresholds["improving"]:
        return "IMPROVING"
    else:
        return "STABLE"


# ─────────────────────────────────────────────
# WEEK 4 — PRIORITY SCORE
# ─────────────────────────────────────────────

def calculate_priority_score(risk_score: float, revenue_at_risk: float) -> float:
    """
    priority_score = risk_score × revenue_at_risk

    Ensures ranking considers both probability and financial impact.
    """
    return round(float(risk_score) * float(revenue_at_risk), 2)


if __name__ == "__main__":
    # Self-tests
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(10, 0) == 0.0
    assert all(normalize_series(pd.Series([5, 5, 5])) == 0.5)
    assert format_currency(1234.5) == "$1,234.50"

    tier = assign_risk_tier(80, 5000, 100000)
    assert tier == "CRITICAL"

    tier2 = assign_risk_tier(30, 200000, 100000)
    assert tier2 == "CRITICAL"   # revenue override

    flag = assign_velocity_flag(20)
    assert flag == "ACCELERATING"

    flag2 = assign_velocity_flag(-20)
    assert flag2 == "IMPROVING"

    flag3 = assign_velocity_flag(0)
    assert flag3 == "STABLE"

    ps = calculate_priority_score(90, 200000)
    assert ps == 18000000.0

    print("✅ helpers.py Week 4 self-tests passed.")
