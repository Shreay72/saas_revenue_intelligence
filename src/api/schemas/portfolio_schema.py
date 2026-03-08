"""
Portfolio Pydantic Schemas
SaaS Revenue Intelligence API - Week 5
"""

from pydantic import BaseModel
from typing import Dict, List, Optional
from src.api.schemas.account_schema import AccountIntelligenceResponse


class PortfolioSummaryResponse(BaseModel):
    """Full portfolio intelligence summary"""
    total_accounts:           int
    tier_distribution:        Dict[str, int]
    type_distribution:        Dict[str, int]
    action_distribution:      Dict[str, int]
    velocity_distribution:    Dict[str, int]
    total_mrr:                float
    total_revenue_at_risk:    float
    total_recoverable:        float
    pct_recoverable:          float
    generated_at:             str       # ISO timestamp


class CriticalAccountsResponse(BaseModel):
    """Critical accounts list"""
    total_critical:   int
    accounts:         list[AccountIntelligenceResponse]


class CacheStatusResponse(BaseModel):
    """Cache state information"""
    cache_active:     bool
    last_refreshed:   str    # ISO timestamp or "Never"
    accounts_cached:  int


class CacheRefreshResponse(BaseModel):
    """Cache invalidation confirmation"""
    cache_cleared:  bool
    refreshed_at:   str      # ISO timestamp
    message:        str


class HealthResponse(BaseModel):
    """Deep health check response"""
    status:           str     # "healthy" / "degraded" / "unhealthy"
    accounts_loaded:  int
    pipeline_ready:   bool
    version:          str
    checked_at:       str     # ISO timestamp


class StatusResponse(BaseModel):
    """API pipeline status"""
    api_version:        str
    accounts_loaded:    int
    intelligence_cols:  int
    model_info:         Dict[str, str]
    uptime_info:        str
