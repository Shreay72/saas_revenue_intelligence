"""
Portfolio & Cache Endpoints
SaaS Revenue Intelligence API - Week 5

GET  /api/v1/portfolio/summary   → cached portfolio summary
GET  /api/v1/portfolio/critical  → CRITICAL accounts
POST /api/v1/cache/refresh       → invalidate summary cache
GET  /api/v1/cache/status        → cache state
"""

import logging
from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.dependencies import (
    get_intelligence,
    get_cached_summary,
    invalidate_cache,
    get_cache_status,
)
from src.api.schemas.portfolio_schema import (
    PortfolioSummaryResponse,
    CriticalAccountsResponse,
    CacheStatusResponse,
    CacheRefreshResponse,
)
from src.api.schemas.account_schema import AccountIntelligenceResponse
from src.api.routers.accounts import _row_to_intelligence

router  = APIRouter(tags=["Portfolio"])
limiter = Limiter(key_func=get_remote_address)
logger  = logging.getLogger("api.portfolio")


@router.get(
    "/api/v1/portfolio/summary",
    response_model=PortfolioSummaryResponse
)
@limiter.limit("100/minute")
async def portfolio_summary(request: Request):
    """
    Cached portfolio intelligence summary.

    Result is cached until POST /api/v1/cache/refresh is called
    or the server restarts.
    """
    summary = get_cached_summary()
    return PortfolioSummaryResponse(**summary)


@router.get(
    "/api/v1/portfolio/critical",
    response_model=CriticalAccountsResponse
)
@limiter.limit("100/minute")
async def critical_accounts(request: Request):
    """
    All CRITICAL tier accounts sorted by priority_score descending.
    """
    df       = get_intelligence()
    critical = df[df["risk_tier"] == "CRITICAL"].sort_values(
        "priority_score", ascending=False
    )
    accounts = [
        _row_to_intelligence(row)
        for _, row in critical.iterrows()
    ]
    return CriticalAccountsResponse(
        total_critical=len(accounts),
        accounts=accounts,
    )


@router.post(
    "/api/v1/cache/refresh",
    response_model=CacheRefreshResponse
)
@limiter.limit("100/minute")
async def refresh_cache(request: Request):
    """
    Manually invalidate the portfolio summary cache.

    Use when:
    - A new account_intelligence.csv has been generated
    - The dashboard needs fresh data without server restart
    """
    result = invalidate_cache()
    return CacheRefreshResponse(**result)


@router.get(
    "/api/v1/cache/status",
    response_model=CacheStatusResponse
)
@limiter.limit("100/minute")
async def cache_status(request: Request):
    """
    Current cache state — whether summary is cached and when it was loaded.
    """
    status = get_cache_status()
    return CacheStatusResponse(**status)
