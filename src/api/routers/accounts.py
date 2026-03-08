"""
Account Endpoints
SaaS Revenue Intelligence API - Week 5

GET  /api/v1/accounts               paginated + filtered list
GET  /api/v1/accounts/top           top N by priority_score
GET  /api/v1/accounts/search        fuzzy name search
GET  /api/v1/accounts/{id}          full intelligence
GET  /api/v1/accounts/{id}/risk     risk only
GET  /api/v1/accounts/{id}/recommendation  recommendation only
POST /api/v1/accounts/score         real-time scoring
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.dependencies import get_intelligence, get_cached_summary
from src.api.schemas.account_schema import (
    AccountIntelligenceResponse,
    AccountListResponse,
    AccountRiskResponse,
    AccountRecommendationResponse,
    ScoreRequest,
    ScoreResponse,
    SearchResponse,
)

router  = APIRouter(prefix="/api/v1/accounts", tags=["Accounts"])
limiter = Limiter(key_func=get_remote_address)
logger  = logging.getLogger("api.accounts")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _row_to_intelligence(row: pd.Series) -> AccountIntelligenceResponse:
    """Convert a DataFrame row to AccountIntelligenceResponse."""
    return AccountIntelligenceResponse(
        account_id=str(row.get("account_id", "")),
        account_name=str(row.get("account_name", "")),
        total_mrr=float(row.get("total_mrr", 0)),
        risk_score=float(row.get("risk_score", 0)),
        risk_tier=str(row.get("risk_tier", "LOW")),
        risk_type=str(row.get("risk_type", "Healthy")),
        risk_velocity=float(row.get("risk_velocity", 0)),
        velocity_flag=str(row.get("velocity_flag", "STABLE")),
        priority_score=float(row.get("priority_score", 0)),
        churn_probability=float(row.get("churn_probability", 0)),
        revenue_at_risk=float(row.get("revenue_at_risk", 0)),
        health_score=float(row.get("health_score", 0)),
        clv=float(row.get("clv", 0)),
        recommended_action=str(row.get("recommended_action", "")),
        action_owner=str(row.get("action_owner", "")),
        urgency=str(row.get("urgency", "LOW")),
        confidence_level=str(row.get("confidence_level", "LOW")),
        reason=str(row.get("reason", "")),
        expected_recovery=float(row.get("expected_recovery", 0)),
        rule_triggered=str(row.get("rule_triggered", "MONITOR")),
    )


def _get_account_row(account_id: str) -> pd.Series:
    """Lookup one account or raise 404."""
    df  = get_intelligence()
    row = df[df["account_id"] == account_id]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Account '{account_id}' not found",
        )
    return row.iloc[0]


# ─────────────────────────────────────────────
# LIST ACCOUNTS
# ─────────────────────────────────────────────

@router.get("", response_model=AccountListResponse)
@limiter.limit("100/minute")
async def list_accounts(
    request:       Request,
    risk_tier:     Optional[str]   = Query(None),
    risk_type:     Optional[str]   = Query(None),
    urgency:       Optional[str]   = Query(None),
    rule_triggered:Optional[str]   = Query(None),
    min_mrr:       Optional[float] = Query(None, ge=0),
    max_mrr:       Optional[float] = Query(None, ge=0),
    sort_by:       str             = Query("priority_score"),
    sort_order:    str             = Query("desc"),
    page:          int             = Query(1, ge=1),
    page_size:     int             = Query(20, ge=1, le=100),
):
    """
    List all accounts with optional filters and pagination.

    - **risk_tier**: CRITICAL / HIGH / MEDIUM / LOW
    - **risk_type**: Strategic Risk / Revenue Risk / etc.
    - **urgency**: IMMEDIATE / HIGH / MEDIUM / LOW
    - **page_size**: max 100
    """
    df = get_intelligence().copy()

    # Apply filters
    if risk_tier:
        df = df[df["risk_tier"] == risk_tier.upper()]
    if risk_type and "risk_type" in df.columns:
        df = df[df["risk_type"].str.lower() == risk_type.lower()]
    if urgency and "urgency" in df.columns:
        df = df[df["urgency"] == urgency.upper()]
    if rule_triggered and "rule_triggered" in df.columns:
        df = df[df["rule_triggered"] == rule_triggered.upper()]
    if min_mrr is not None and "total_mrr" in df.columns:
        df = df[df["total_mrr"] >= min_mrr]
    if max_mrr is not None and "total_mrr" in df.columns:
        df = df[df["total_mrr"] <= max_mrr]

    # Sort
    if sort_by in df.columns:
        df = df.sort_values(
            sort_by,
            ascending=(sort_order.lower() == "asc")
        )

    total      = len(df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start       = (page - 1) * page_size
    end         = start + page_size
    page_df     = df.iloc[start:end]

    accounts = [_row_to_intelligence(row) for _, row in page_df.iterrows()]

    return AccountListResponse(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        accounts=accounts,
    )


# ─────────────────────────────────────────────
# TOP ACCOUNTS
# ─────────────────────────────────────────────

@router.get("/top", response_model=list[AccountIntelligenceResponse])
@limiter.limit("100/minute")
async def top_accounts(
    request: Request,
    limit:   int = Query(20, ge=1, le=50),
):
    """
    Top N accounts ranked by priority_score (risk × revenue).

    - **limit**: max 50
    """
    df   = get_intelligence()
    top  = df.nlargest(limit, "priority_score")
    return [_row_to_intelligence(row) for _, row in top.iterrows()]


# ─────────────────────────────────────────────
# SEARCH ACCOUNTS
# ─────────────────────────────────────────────

@router.get("/search", response_model=SearchResponse)
@limiter.limit("100/minute")
async def search_accounts(
    request: Request,
    q:       str = Query(..., min_length=2, max_length=100),
    limit:   int = Query(10, ge=1, le=50),
):
    """
    Fuzzy account name search.

    - **q**: partial name (min 2 characters)
    - **limit**: max 50 results
    """
    df       = get_intelligence()
    mask     = df["account_name"].str.lower().str.contains(
        q.lower(), na=False
    )
    results  = df[mask].head(limit)
    accounts = [_row_to_intelligence(row) for _, row in results.iterrows()]

    return SearchResponse(
        query=q,
        total=len(accounts),
        accounts=accounts,
    )


# ─────────────────────────────────────────────
# SINGLE ACCOUNT — FULL INTELLIGENCE
# ─────────────────────────────────────────────

@router.get("/{account_id}", response_model=AccountIntelligenceResponse)
@limiter.limit("100/minute")
async def get_account(request: Request, account_id: str):
    """
    Full intelligence for a single account.
    """
    row = _get_account_row(account_id)
    return _row_to_intelligence(row)


# ─────────────────────────────────────────────
# SINGLE ACCOUNT — RISK ONLY
# ─────────────────────────────────────────────

@router.get("/{account_id}/risk", response_model=AccountRiskResponse)
@limiter.limit("100/minute")
async def get_account_risk(request: Request, account_id: str):
    """
    Risk score, tier, type and velocity for a single account.
    """
    row = _get_account_row(account_id)
    return AccountRiskResponse(
        account_id=str(row.get("account_id", "")),
        account_name=str(row.get("account_name", "")),
        total_mrr=float(row.get("total_mrr", 0)),
        risk_score=float(row.get("risk_score", 0)),
        risk_tier=str(row.get("risk_tier", "LOW")),
        risk_type=str(row.get("risk_type", "Healthy")),
        risk_velocity=float(row.get("risk_velocity", 0)),
        velocity_flag=str(row.get("velocity_flag", "STABLE")),
        priority_score=float(row.get("priority_score", 0)),
    )


# ─────────────────────────────────────────────
# SINGLE ACCOUNT — RECOMMENDATION ONLY
# ─────────────────────────────────────────────

@router.get(
    "/{account_id}/recommendation",
    response_model=AccountRecommendationResponse
)
@limiter.limit("100/minute")
async def get_account_recommendation(request: Request, account_id: str):
    """
    Recommendation, action owner, urgency, reason
    and expected recovery for a single account.
    """
    row = _get_account_row(account_id)
    return AccountRecommendationResponse(
        account_id=str(row.get("account_id", "")),
        account_name=str(row.get("account_name", "")),
        recommended_action=str(row.get("recommended_action", "")),
        action_owner=str(row.get("action_owner", "")),
        urgency=str(row.get("urgency", "LOW")),
        confidence_level=str(row.get("confidence_level", "LOW")),
        reason=str(row.get("reason", "")),
        expected_recovery=float(row.get("expected_recovery", 0)),
        rule_triggered=str(row.get("rule_triggered", "MONITOR")),
    )


# ─────────────────────────────────────────────
# REAL-TIME SCORING
# ─────────────────────────────────────────────

@router.post("/score", response_model=ScoreResponse)
@limiter.limit("100/minute")
async def score_account(request: Request, body: ScoreRequest):
    """
    Score a new account in real-time through the full pipeline.

    Returns risk score, tier, recommendation, and recovery estimate.
    Adds **scored_at** ISO timestamp for audit trail.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from src.risk.risk_engine import RiskEngine
    from src.risk.scoring_logic import ScoringLogic
    from src.recommendation.business_rules import BusinessRules
    from src.recommendation.recommendation_engine import RecommendationEngine
    from src.utils.metrics import calculate_expected_recovery

    # Build portfolio p95 from loaded intelligence for normalization
    df         = get_intelligence()
    p95_revenue = max(float(df["revenue_at_risk"].quantile(0.95)), 1.0) \
        if "revenue_at_risk" in df.columns else 1.0
    p90_revenue = max(float(df["revenue_at_risk"].quantile(0.90)), 1.0) \
        if "revenue_at_risk" in df.columns else 1.0
    p75_clv     = float(df["clv"].quantile(0.75)) \
        if "clv" in df.columns else 0.0
    p75_rev     = float(df["revenue_at_risk"].quantile(0.75)) \
        if "revenue_at_risk" in df.columns else 0.0
    median_tickets = float(df["ticket_count"].median()) \
        if "ticket_count" in df.columns else 0.0
    median_usage   = float(df["unique_features_used"].median()) \
        if "unique_features_used" in df.columns else 0.0

    # Score
    engine = RiskEngine()
    risk_score = engine.score_account(
        churn_probability=body.churn_probability,
        revenue_at_risk=body.revenue_at_risk,
        health_score=body.health_score,
        p95_revenue=p95_revenue,
    )

    # Tier + type
    from src.utils.helpers import assign_risk_tier, assign_risk_type
    risk_tier = assign_risk_tier(
        risk_score=risk_score,
        revenue_at_risk=body.revenue_at_risk,
        p90_revenue=p90_revenue,
    )
    risk_type = assign_risk_type(
        clv=body.clv,
        churn_probability=body.churn_probability,
        revenue_at_risk=body.revenue_at_risk,
        engagement_state=body.engagement_state,
        support_pressure_signal=body.support_pressure_signal,
        clv_p75=p75_clv,
        rev_risk_p75=p75_rev,
    )

    # Business rules
    import pandas as pd
    rules      = BusinessRules()
    rule_result = rules.evaluate(
        pd.Series(body.model_dump()),
        median_tickets=median_tickets,
        median_usage=median_usage,
    )

    # Reason + recovery
    rec_engine = RecommendationEngine()
    row_series = pd.Series({**body.model_dump(),
                             "risk_score": risk_score})
    reason     = rec_engine._build_reason(row_series, rule_result["rule_triggered"])
    recovery   = calculate_expected_recovery(
        revenue_at_risk=body.revenue_at_risk,
        rule_triggered=rule_result["rule_triggered"],
    )

    return ScoreResponse(
        account_id=body.account_id,
        risk_score=risk_score,
        risk_tier=risk_tier,
        risk_type=risk_type,
        recommended_action=rule_result["recommended_action"],
        action_owner=rule_result["action_owner"],
        urgency=rule_result["urgency"],
        confidence_level=rule_result["confidence_level"],
        reason=reason,
        expected_recovery=recovery,
        rule_triggered=rule_result["rule_triggered"],
        scored_at=datetime.utcnow().isoformat() + "Z",
    )
