"""
Formatters & Style Utilities
SaaS Revenue Intelligence Dashboard - Week 6
"""

import pandas as pd
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────
# COLOR MAPS (consistent across all pages)
# ─────────────────────────────────────────────

TIER_COLORS = {
    "CRITICAL": "#FF4B4B",
    "HIGH":     "#FF8C00",
    "MEDIUM":   "#FFD700",
    "LOW":      "#00CC44",
}

URGENCY_COLORS = {
    "IMMEDIATE": "#FF4B4B",
    "HIGH":      "#FF8C00",
    "MEDIUM":    "#FFD700",
    "LOW":       "#00CC44",
}

RULE_COLORS = {
    "VP_ESCALATION": "#FF4B4B",
    "WINBACK":       "#FF8C00",
    "MONITOR":       "#4B9EFF",
}


# ─────────────────────────────────────────────
# NUMBER FORMATTERS
# ─────────────────────────────────────────────

def format_currency(value: float) -> str:
    """$131,550"""
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def format_large_currency(value: float) -> str:
    """$11.3M / $950K / $500"""
    try:
        v = float(value)
        if v >= 1_000_000:
            return f"${v / 1_000_000:.1f}M"
        if v >= 1_000:
            return f"${v / 1_000:.0f}K"
        return f"${v:.0f}"
    except Exception:
        return "$0"


def format_percent(value: float) -> str:
    """98.9%"""
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"


def format_score(value: float) -> str:
    """82.7"""
    try:
        return f"{float(value):.1f}"
    except Exception:
        return "0.0"


# ─────────────────────────────────────────────
# BADGES
# ─────────────────────────────────────────────

def tier_badge(tier: str) -> str:
    icons = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "🟢",
    }
    return f"{icons.get(tier, '⚪')} {tier}"


def urgency_badge(urgency: str) -> str:
    icons = {
        "IMMEDIATE": "⚡",
        "HIGH":      "🔥",
        "MEDIUM":    "⚠️",
        "LOW":       "💤",
    }
    return f"{icons.get(urgency, '')} {urgency}"


# ─────────────────────────────────────────────
# TIMESTAMP
# ─────────────────────────────────────────────

def format_timestamp(iso_str: str) -> str:
    """'2026-03-07T07:49:10Z' → '07:49 AM UTC'"""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%I:%M %p UTC")
    except Exception:
        return iso_str


# ─────────────────────────────────────────────
# TREND INDICATOR
# ─────────────────────────────────────────────

def format_trend(
    current: float,
    previous: float,
    invert: bool = False
) -> str:
    """
    Returns markdown trend string.
    invert=True  → decrease is good (e.g. revenue_at_risk)
    invert=False → increase is good (e.g. total_mrr)
    """
    if previous == 0 or previous is None:
        return ""
    try:
        pct  = (current - previous) / abs(previous) * 100
        good = (pct < 0) if invert else (pct > 0)
        color = "green" if good else "red"
        arrow = "↑" if pct > 0 else "↓"
        return f":{color}[{arrow} {abs(pct):.1f}% vs last run]"
    except Exception:
        return ""


# ─────────────────────────────────────────────
# TABLE STYLING (no matplotlib required)
# ─────────────────────────────────────────────

def style_accounts_table(df: pd.DataFrame):
    """
    Apply professional formatting to account DataFrames
    without requiring matplotlib (no background_gradient).
    """
    format_map = {
        "total_mrr":          "${:,.0f}",
        "revenue_at_risk":    "${:,.0f}",
        "expected_recovery":  "${:,.0f}",
        "clv":                "${:,.0f}",
        "risk_score":         "{:.1f}",
        "health_score":       "{:.1f}",
        "churn_probability":  "{:.1%}",
        "priority_score":     "{:,.0f}",
    }

    active_format = {k: v for k, v in format_map.items() if k in df.columns}

    return df.style.format(active_format)


def prepare_display_df(
    accounts: list,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Convert list of account dicts to a display DataFrame.
    Optionally select + rename columns.
    """
    if not accounts:
        return pd.DataFrame()

    df = pd.DataFrame(accounts)

    default_cols = [
        "account_name", "total_mrr", "risk_score", "risk_tier",
        "churn_probability", "revenue_at_risk",
        "recommended_action", "action_owner",
        "urgency", "expected_recovery", "rule_triggered",
    ]

    cols = columns or [c for c in default_cols if c in df.columns]
    return df[[c for c in cols if c in df.columns]]
