"""
Account Pydantic Schemas
SaaS Revenue Intelligence API - Week 5
"""

from pydantic import BaseModel, Field
from typing import Optional


class AccountRiskResponse(BaseModel):
    """Risk-only response for GET /accounts/{id}/risk"""
    account_id:       str
    account_name:     str
    total_mrr:        float
    risk_score:       float
    risk_tier:        str
    risk_type:        str
    risk_velocity:    float
    velocity_flag:    str
    priority_score:   float


class AccountRecommendationResponse(BaseModel):
    """Recommendation-only response for GET /accounts/{id}/recommendation"""
    account_id:          str
    account_name:        str
    recommended_action:  str
    action_owner:        str
    urgency:             str
    confidence_level:    str
    reason:              str
    expected_recovery:   float
    rule_triggered:      str


class AccountIntelligenceResponse(BaseModel):
    """Full intelligence response for GET /accounts/{id}"""
    # Identity
    account_id:           str
    account_name:         str
    total_mrr:            float

    # Risk Intelligence
    risk_score:           float
    risk_tier:            str
    risk_type:            str
    risk_velocity:        float
    velocity_flag:        str
    priority_score:       float

    # ML Inputs
    churn_probability:    float
    revenue_at_risk:      float
    health_score:         float
    clv:                  float

    # Recommendation
    recommended_action:   str
    action_owner:         str
    urgency:              str
    confidence_level:     str
    reason:               str
    expected_recovery:    float
    rule_triggered:       str


class AccountListResponse(BaseModel):
    """Paginated account list response"""
    total:        int
    page:         int
    page_size:    int
    total_pages:  int
    accounts:     list[AccountIntelligenceResponse]


class ScoreRequest(BaseModel):
    """Request body for POST /accounts/score"""
    account_id:               str
    total_mrr:                float   = Field(ge=0.0)
    churn_probability:        float   = Field(ge=0.0, le=1.0)
    revenue_at_risk:          float   = Field(ge=0.0)
    health_score:             float   = Field(ge=0.0, le=100.0)
    clv:                      float   = Field(ge=0.0)
    engagement_state:         int     = Field(ge=-1, le=1)
    support_pressure_signal:  int     = Field(ge=0, le=1)
    revenue_change_signal:    float   = Field(ge=-1.0, le=1.0)
    tenure_months:            float   = Field(ge=0.0)
    ticket_count:             int     = Field(ge=0)
    unique_features_used:     int     = Field(ge=0)
    total_usage:              int     = Field(ge=0)
    churn_flag:               int     = Field(ge=0, le=1)
    auto_renew_ratio:         float   = Field(ge=0.0, le=1.0)


class ScoreResponse(BaseModel):
    """Real-time scoring response for POST /accounts/score"""
    account_id:           str
    risk_score:           float
    risk_tier:            str
    risk_type:            str
    recommended_action:   str
    action_owner:         str
    urgency:              str
    confidence_level:     str
    reason:               str
    expected_recovery:    float
    rule_triggered:       str
    scored_at:            str          # ISO timestamp — for audit trail


class SearchResponse(BaseModel):
    """Search results response"""
    query:    str
    total:    int
    accounts: list[AccountIntelligenceResponse]
