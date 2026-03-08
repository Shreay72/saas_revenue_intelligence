"""
Health & Status Endpoints
SaaS Revenue Intelligence API - Week 5

GET /health         → deep dependency health check
GET /api/v1/status  → pipeline info
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.dependencies import (
    get_intelligence,
    get_accounts_count,
    is_pipeline_ready,
    VERSION,
)
from src.api.schemas.portfolio_schema import HealthResponse, StatusResponse

router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger  = logging.getLogger("api.health")


@router.get("/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit("100/minute")
async def health_check(request: Request):
    """
    Deep health check.

    Verifies:
        - intelligence CSV loaded
        - required columns present
        - account count > 0
    """
    pipeline_ready  = is_pipeline_ready()
    accounts_loaded = get_accounts_count() if pipeline_ready else 0
    status          = "healthy" if pipeline_ready and accounts_loaded > 0 \
                      else "degraded"

    logger.debug(f"Health check: {status} | accounts: {accounts_loaded}")

    return HealthResponse(
        status=status,
        accounts_loaded=accounts_loaded,
        pipeline_ready=pipeline_ready,
        version=VERSION,
        checked_at=datetime.utcnow().isoformat() + "Z",
    )


@router.get("/api/v1/status", response_model=StatusResponse, tags=["Health"])
@limiter.limit("100/minute")
async def pipeline_status(request: Request):
    """
    Pipeline status — model info and loaded data summary.
    """
    df = get_intelligence()

    return StatusResponse(
        api_version=VERSION,
        accounts_loaded=len(df),
        intelligence_cols=len(df.columns),
        model_info={
            "churn_model":   "Logistic Regression (Week 2)",
            "revenue_model": "GradientBoosting R²=0.9186 (Week 3)",
            "clv_model":     "Deterministic Formula (Week 3)",
            "risk_engine":   "Composite Score (Week 4)",
        },
        uptime_info=f"Serving {len(df)} accounts across "
                    f"{len(df.columns)} intelligence dimensions",
    )
