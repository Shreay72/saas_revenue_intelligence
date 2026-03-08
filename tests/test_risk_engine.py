"""
Week 4 Tests — Risk Engine & Recommendation System
Run:
    pytest tests/test_risk_engine.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_account():
    return {
        "account_id":               "A-001",
        "account_name":             "TestCo",
        "total_mrr":                50000,
        "churn_probability":        0.85,
        "revenue_at_risk":          42500,
        "health_score":             18.0,
        "clv":                      600000,
        "engagement_state":         -1,
        "support_pressure_signal":  1,
        "revenue_change_signal":    -0.3,
        "tenure_months":            24,
        "ticket_count":             15,
        "unique_features_used":     3,
        "total_usage":              100,
        "churn_flag":               1,
        "auto_renew_ratio":         0.2,
        "engagement_score":         15.0,
        "support_risk_score":       80.0,
    }


@pytest.fixture
def healthy_account():
    return {
        "account_id":               "A-002",
        "account_name":             "HealthyCo",
        "total_mrr":                8000,
        "churn_probability":        0.05,
        "revenue_at_risk":          400,
        "health_score":             82.0,
        "clv":                      80000,
        "engagement_state":         1,
        "support_pressure_signal":  0,
        "revenue_change_signal":    0.5,
        "tenure_months":            18,
        "ticket_count":             2,
        "unique_features_used":     12,
        "total_usage":              500,
        "churn_flag":               0,
        "auto_renew_ratio":         1.0,
        "engagement_score":         85.0,
        "support_risk_score":       10.0,
    }


@pytest.fixture
def sample_df(sample_account, healthy_account):
    rows = []
    for i in range(5):
        row = sample_account.copy()
        row["account_id"] = f"A-{i:03d}"
        row["churn_probability"] = 0.75 + i * 0.04
        row["revenue_at_risk"]   = 30000 + i * 5000
        row["health_score"]      = 20 - i * 2
        row["clv"]               = 500000 + i * 100000
        rows.append(row)
    rows.append(healthy_account)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# TEST METRICS
# ─────────────────────────────────────────────────────────────

class TestMetrics:

    def test_composite_risk_high(self):
        from src.utils.metrics import calculate_composite_risk_score
        score = calculate_composite_risk_score(
            churn_probability=0.95,
            revenue_at_risk=90000,
            health_score=10.0,
            p95_revenue=100000,
        )
        assert score >= 75

    def test_composite_risk_low(self):
        from src.utils.metrics import calculate_composite_risk_score
        score = calculate_composite_risk_score(
            churn_probability=0.05,
            revenue_at_risk=500,
            health_score=90.0,
            p95_revenue=100000,
        )
        assert score < 35

    def test_composite_risk_always_in_range(self):
        from src.utils.metrics import calculate_composite_risk_score
        for cp in [0.0, 0.5, 1.0]:
            for rar in [0, 50000, 200000]:
                for hs in [0, 50, 100]:
                    score = calculate_composite_risk_score(
                        cp, rar, hs, p95_revenue=100000
                    )
                    assert 0 <= score <= 100, f"Score {score} out of range"

    def test_p95_zero_safe(self):
        from src.utils.metrics import calculate_composite_risk_score
        # p95 of 0 should not cause division error
        score = calculate_composite_risk_score(
            churn_probability=0.5,
            revenue_at_risk=0,
            health_score=50,
            p95_revenue=0,    # edge case
        )
        assert 0 <= score <= 100

    def test_expected_recovery_vp(self):
        from src.utils.metrics import calculate_expected_recovery
        recovery = calculate_expected_recovery(100000, "VP_ESCALATION")
        assert recovery == 50000.0

    def test_expected_recovery_monitor(self):
        from src.utils.metrics import calculate_expected_recovery
        recovery = calculate_expected_recovery(100000, "MONITOR")
        assert recovery == 5000.0

    def test_expected_recovery_unknown_rule(self):
        from src.utils.metrics import calculate_expected_recovery
        recovery = calculate_expected_recovery(100000, "UNKNOWN_RULE")
        assert recovery == 5000.0   # defaults to MONITOR rate


# ─────────────────────────────────────────────────────────────
# TEST HELPERS
# ─────────────────────────────────────────────────────────────

class TestHelpers:

    def test_assign_risk_tier_critical_by_score(self):
        from src.utils.helpers import assign_risk_tier
        assert assign_risk_tier(80, 1000, 100000) == "CRITICAL"

    def test_assign_risk_tier_critical_by_revenue(self):
        from src.utils.helpers import assign_risk_tier
        # Score is only 50 but revenue override kicks in
        assert assign_risk_tier(50, 200000, 100000) == "CRITICAL"

    def test_assign_risk_tier_high(self):
        from src.utils.helpers import assign_risk_tier
        assert assign_risk_tier(65, 1000, 100000) == "HIGH"

    def test_assign_risk_tier_medium(self):
        from src.utils.helpers import assign_risk_tier
        assert assign_risk_tier(40, 1000, 100000) == "MEDIUM"

    def test_assign_risk_tier_low(self):
        from src.utils.helpers import assign_risk_tier
        assert assign_risk_tier(20, 1000, 100000) == "LOW"

    def test_assign_risk_type_strategic(self):
        from src.utils.helpers import assign_risk_type
        t = assign_risk_type(
            clv=600000, churn_probability=0.85,
            revenue_at_risk=50000, engagement_state=-1,
            support_pressure_signal=1,
            clv_p75=500000, rev_risk_p75=30000
        )
        assert t == "Strategic Risk"

    def test_assign_risk_type_revenue(self):
        from src.utils.helpers import assign_risk_type
        t = assign_risk_type(
            clv=100000, churn_probability=0.4,
            revenue_at_risk=80000, engagement_state=0,
            support_pressure_signal=0,
            clv_p75=500000, rev_risk_p75=30000
        )
        assert t == "Revenue Risk"

    def test_assign_risk_type_healthy(self):
        from src.utils.helpers import assign_risk_type
        t = assign_risk_type(
            clv=50000, churn_probability=0.1,
            revenue_at_risk=500, engagement_state=1,
            support_pressure_signal=0,
            clv_p75=500000, rev_risk_p75=30000
        )
        assert t == "Healthy"

    def test_velocity_flag_accelerating(self):
        from src.utils.helpers import assign_velocity_flag
        assert assign_velocity_flag(20) == "ACCELERATING"

    def test_velocity_flag_improving(self):
        from src.utils.helpers import assign_velocity_flag
        assert assign_velocity_flag(-20) == "IMPROVING"

    def test_velocity_flag_stable(self):
        from src.utils.helpers import assign_velocity_flag
        assert assign_velocity_flag(0) == "STABLE"

    def test_priority_score(self):
        from src.utils.helpers import calculate_priority_score
        assert calculate_priority_score(90, 200000) == 18000000.0

    def test_priority_score_zero(self):
        from src.utils.helpers import calculate_priority_score
        assert calculate_priority_score(0, 200000) == 0.0


# ─────────────────────────────────────────────────────────────
# TEST RISK ENGINE
# ─────────────────────────────────────────────────────────────

class TestRiskEngine:

    def test_initialization(self):
        from src.risk.risk_engine import RiskEngine
        engine = RiskEngine()
        assert engine.weights is not None
        assert "churn_probability" in engine.weights

    def test_score_account_high(self, sample_account):
        from src.risk.risk_engine import RiskEngine
        engine = RiskEngine()
        score  = engine.score_account(
            churn_probability=0.95,
            revenue_at_risk=90000,
            health_score=10.0,
            p95_revenue=100000,
        )
        assert score >= 75

    def test_score_account_low(self):
        from src.risk.risk_engine import RiskEngine
        engine = RiskEngine()
        score  = engine.score_account(
            churn_probability=0.05,
            revenue_at_risk=200,
            health_score=90.0,
            p95_revenue=100000,
        )
        assert score < 35

    def test_portfolio_scoring_shape(self, sample_df):
        from src.risk.risk_engine import RiskEngine
        engine = RiskEngine()
        result = engine.score_portfolio(sample_df)
        assert len(result) == len(sample_df)
        assert "risk_score" in result.columns

    def test_portfolio_scores_in_range(self, sample_df):
        from src.risk.risk_engine import RiskEngine
        engine = RiskEngine()
        result = engine.score_portfolio(sample_df)
        assert result["risk_score"].between(0, 100).all()

    def test_portfolio_has_required_columns(self, sample_df):
        from src.risk.risk_engine import RiskEngine
        engine  = RiskEngine()
        result  = engine.score_portfolio(sample_df)
        required = {
            "risk_score", "risk_tier", "risk_type",
            "risk_velocity", "velocity_flag", "priority_score"
        }
        assert required.issubset(set(result.columns))

    def test_velocity_default_zero(self, sample_df):
        from src.risk.risk_engine import RiskEngine
        engine = RiskEngine()
        result = engine.score_portfolio(sample_df)
        assert (result["risk_velocity"] == 0.0).all()


# ─────────────────────────────────────────────────────────────
# TEST BUSINESS RULES
# ─────────────────────────────────────────────────────────────

class TestBusinessRules:

    def test_vp_escalation_fires(self, sample_account):
        from src.recommendation.business_rules import BusinessRules
        rules  = BusinessRules()
        result = rules.evaluate(
            pd.Series(sample_account),
            median_tickets=5.0,
            median_usage=5.0
        )
        assert result["rule_triggered"] == "VP_ESCALATION"

    def test_upsell_fires(self, healthy_account):
        from src.recommendation.business_rules import BusinessRules
        rules  = BusinessRules()
        result = rules.evaluate(
            pd.Series(healthy_account),
            median_tickets=5.0,
            median_usage=5.0
        )
        assert result["rule_triggered"] == "UPSELL"

    def test_monitor_fires_when_no_rule(self):
        from src.recommendation.business_rules import BusinessRules
        rules = BusinessRules()
        neutral = pd.Series({
            "churn_probability":       0.30,
            "clv":                     50000,
            "engagement_state":        0,
            "tenure_months":           6,
            "support_pressure_signal": 0,
            "ticket_count":            3,
            "revenue_change_signal":   0.0,
            "churn_flag":              0,
            "unique_features_used":    8,
            "total_usage":             200,
        })
        result = rules.evaluate(neutral, median_tickets=5.0, median_usage=5.0)
        assert result["rule_triggered"] == "MONITOR"

    def test_monitor_has_sub_action(self):
        from src.recommendation.business_rules import BusinessRules
        rules = BusinessRules()
        neutral = pd.Series({
            "churn_probability":       0.30,
            "clv":                     50000,
            "engagement_state":        0,
            "tenure_months":           6,
            "support_pressure_signal": 0,
            "ticket_count":            3,
            "revenue_change_signal":   0.0,
            "churn_flag":              0,
            "unique_features_used":    2,   # below median → usage nudge
            "total_usage":             200,
        })
        result = rules.evaluate(neutral, median_tickets=5.0, median_usage=5.0)
        assert result["recommended_action"] != "Monitor"
        assert len(result["recommended_action"]) > 5

    def test_result_has_required_fields(self, sample_account):
        from src.recommendation.business_rules import BusinessRules
        rules  = BusinessRules()
        result = rules.evaluate(
            pd.Series(sample_account),
            median_tickets=5.0,
            median_usage=5.0
        )
        required = {
            "rule_triggered", "priority", "recommended_action",
            "action_owner", "urgency", "confidence_level"
        }
        assert required.issubset(set(result.keys()))

    def test_vp_escalation_owner(self, sample_account):
        from src.recommendation.business_rules import BusinessRules
        rules  = BusinessRules()
        result = rules.evaluate(
            pd.Series(sample_account),
            median_tickets=5.0,
            median_usage=5.0
        )
        assert result["action_owner"] == "VP_CUSTOMER_SUCCESS"

    def test_upsell_owner(self, healthy_account):
        from src.recommendation.business_rules import BusinessRules
        rules  = BusinessRules()
        result = rules.evaluate(
            pd.Series(healthy_account),
            median_tickets=5.0,
            median_usage=5.0
        )
        assert result["action_owner"] == "ACCOUNT_MANAGER"


# ─────────────────────────────────────────────────────────────
# TEST RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────

class TestRecommendationEngine:

    def test_intelligence_schema(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        result = engine.generate_intelligence(sample_df)
        required = {
            "risk_score", "risk_tier", "risk_type",
            "recommended_action", "action_owner",
            "urgency", "confidence_level",
            "reason", "expected_recovery", "rule_triggered",
        }
        assert required.issubset(set(result.columns))

    def test_all_accounts_have_recommendation(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        result = engine.generate_intelligence(sample_df)
        assert result["recommended_action"].notna().all()
        assert (result["recommended_action"] != "").all()

    def test_reason_not_empty(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        result = engine.generate_intelligence(sample_df)
        assert result["reason"].notna().all()
        assert (result["reason"].str.len() > 10).all()

    def test_expected_recovery_positive(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        result = engine.generate_intelligence(sample_df)
        assert (result["expected_recovery"] >= 0).all()

    def test_urgency_valid_values(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine  = RecommendationEngine()
        result  = engine.generate_intelligence(sample_df)
        valid   = {"IMMEDIATE", "HIGH", "MEDIUM", "LOW"}
        assert set(result["urgency"].unique()).issubset(valid)

    def test_confidence_valid_values(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine  = RecommendationEngine()
        result  = engine.generate_intelligence(sample_df)
        valid   = {"HIGH", "MEDIUM", "LOW"}
        assert set(result["confidence_level"].unique()).issubset(valid)

    def test_sorted_by_priority_score(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        result = engine.generate_intelligence(sample_df)
        scores = result["priority_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────
# TEST PORTFOLIO SUMMARY
# ─────────────────────────────────────────────────────────────

class TestPortfolioSummary:

    def test_summary_keys(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine = RecommendationEngine()
        intel  = engine.generate_intelligence(sample_df)
        summary = engine.generate_portfolio_summary(intel)
        required = {
            "total_accounts", "tier_distribution",
            "total_mrr", "total_revenue_at_risk",
            "total_recoverable", "pct_recoverable",
            "top_20_accounts", "action_distribution",
        }
        assert required.issubset(set(summary.keys()))

    def test_total_accounts_correct(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine  = RecommendationEngine()
        intel   = engine.generate_intelligence(sample_df)
        summary = engine.generate_portfolio_summary(intel)
        assert summary["total_accounts"] == len(sample_df)

    def test_recoverable_less_than_at_risk(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine  = RecommendationEngine()
        intel   = engine.generate_intelligence(sample_df)
        summary = engine.generate_portfolio_summary(intel)
        assert summary["total_recoverable"] <= summary["total_revenue_at_risk"]

    def test_top20_not_empty(self, sample_df):
        from src.recommendation.recommendation_engine import RecommendationEngine
        engine  = RecommendationEngine()
        intel   = engine.generate_intelligence(sample_df)
        summary = engine.generate_portfolio_summary(intel)
        assert len(summary["top_20_accounts"]) > 0
