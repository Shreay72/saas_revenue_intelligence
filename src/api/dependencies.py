"""
Shared Dependencies — Singleton Pipeline Loader + Cache
SaaS Revenue Intelligence API - Week 5

Design:
    - All heavy objects loaded ONCE at startup
    - get_cached_summary() uses lru_cache for portfolio summary
    - invalidate_cache() clears lru_cache for manual refresh
"""

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("api.dependencies")

# ─────────────────────────────────────────────
# SINGLETONS (loaded once at startup)
# ─────────────────────────────────────────────

_intelligence_df: Optional[pd.DataFrame] = None
_cache_loaded_at: Optional[str] = None

INTELLIGENCE_PATH = "data/processed/account_intelligence.csv"
CONFIG_PATH       = "config/model_config.yaml"
VERSION           = "1.0.0"


def load_intelligence(path: str = INTELLIGENCE_PATH) -> pd.DataFrame:
    """Load intelligence CSV once at startup."""
    global _intelligence_df, _cache_loaded_at

    if _intelligence_df is None:
        logger.info(f"Loading intelligence table: {path}")
        _intelligence_df = pd.read_csv(path)
        _cache_loaded_at = datetime.utcnow().isoformat() + "Z"
        logger.info(
            f"✅ Intelligence loaded: "
            f"{len(_intelligence_df)} accounts × "
            f"{len(_intelligence_df.columns)} columns"
        )

    return _intelligence_df


def get_intelligence() -> pd.DataFrame:
    """Dependency: returns the loaded intelligence DataFrame."""
    return load_intelligence()


def get_accounts_count() -> int:
    """Returns number of loaded accounts."""
    df = get_intelligence()
    return len(df)


def is_pipeline_ready() -> bool:
    """Check if intelligence data is loaded and valid."""
    try:
        df = get_intelligence()
        required = {"account_id", "risk_score", "risk_tier",
                    "recommended_action", "expected_recovery"}
        return required.issubset(set(df.columns))
    except Exception:
        return False


def get_cache_loaded_at() -> str:
    return _cache_loaded_at or "Never"


# ─────────────────────────────────────────────
# PORTFOLIO SUMMARY CACHE
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_cached_summary() -> dict:
    """
    Compute and cache portfolio summary.
    Cache persists until invalidate_cache() is called
    or the server restarts.
    """
    logger.info("Computing portfolio summary (caching result)...")
    df = get_intelligence()

    tier_counts = df["risk_tier"].value_counts().to_dict() \
        if "risk_tier" in df.columns else {}
    type_counts = df["risk_type"].value_counts().to_dict() \
        if "risk_type" in df.columns else {}
    action_counts = df["rule_triggered"].value_counts().to_dict() \
        if "rule_triggered" in df.columns else {}
    velocity_counts = df["velocity_flag"].value_counts().to_dict() \
        if "velocity_flag" in df.columns else {}

    summary = {
        "total_accounts":        int(len(df)),
        "tier_distribution":     {k: int(v) for k, v in tier_counts.items()},
        "type_distribution":     {k: int(v) for k, v in type_counts.items()},
        "action_distribution":   {k: int(v) for k, v in action_counts.items()},
        "velocity_distribution": {k: int(v) for k, v in velocity_counts.items()},
        "total_mrr":             round(float(df["total_mrr"].sum()), 2)
            if "total_mrr" in df.columns else 0.0,
        "total_revenue_at_risk": round(float(df["revenue_at_risk"].sum()), 2)
            if "revenue_at_risk" in df.columns else 0.0,
        "total_recoverable":     round(float(df["expected_recovery"].sum()), 2)
            if "expected_recovery" in df.columns else 0.0,
        "pct_recoverable":       0.0,
        "generated_at":          datetime.utcnow().isoformat() + "Z",
    }

    if summary["total_revenue_at_risk"] > 0:
        summary["pct_recoverable"] = round(
            summary["total_recoverable"] / summary["total_revenue_at_risk"] * 100,
            1
        )

    logger.info("✅ Portfolio summary cached.")
    return summary


def invalidate_cache() -> dict:
    """
    Clear the portfolio summary cache.
    Called by POST /api/v1/cache/refresh.
    Use when new intelligence file is generated.
    """
    get_cached_summary.cache_clear()
    refreshed_at = datetime.utcnow().isoformat() + "Z"
    logger.info(f"✅ Portfolio summary cache cleared at {refreshed_at}")
    return {
        "cache_cleared": True,
        "refreshed_at":  refreshed_at,
        "message":       "Cache cleared. Next /portfolio/summary call will recompute.",
    }


def get_cache_status() -> dict:
    """Return current cache state."""
    cache_info = get_cached_summary.cache_info()
    is_active  = cache_info.currsize > 0
    return {
        "cache_active":    is_active,
        "last_refreshed":  get_cache_loaded_at(),
        "accounts_cached": get_accounts_count() if is_active else 0,
    }
