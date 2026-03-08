"""
Week 5 API Tests — 40+ endpoint tests
Run:
    pytest tests/test_api.py -v
"""

import sys
import pytest
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


# ─────────────────────────────────────────────
# TEST HEALTH ENDPOINTS
# ─────────────────────────────────────────────

class TestHealthEndpoints:

    def test_health_status_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_required_keys(self):
        r = client.get("/health")
        data = r.json()
        for key in ["status", "accounts_loaded", "pipeline_ready",
                    "version", "checked_at"]:
            assert key in data

    def test_health_accounts_loaded_positive(self):
        r = client.get("/health")
        assert r.json()["accounts_loaded"] > 0

    def test_health_pipeline_ready_true(self):
        r = client.get("/health")
        assert r.json()["pipeline_ready"] is True

    def test_status_endpoint_200(self):
        r = client.get("/api/v1/status")
        assert r.status_code == 200

    def test_status_has_model_info(self):
        r = client.get("/api/v1/status")
        assert "model_info" in r.json()


# ─────────────────────────────────────────────
# TEST ACCOUNT LIST
# ─────────────────────────────────────────────

class TestAccountList:

    def test_list_accounts_200(self):
        r = client.get("/api/v1/accounts")
        assert r.status_code == 200

    def test_list_accounts_has_pagination(self):
        r    = client.get("/api/v1/accounts")
        data = r.json()
        for key in ["total", "page", "page_size", "total_pages", "accounts"]:
            assert key in data

    def test_list_accounts_default_page_size_20(self):
        r = client.get("/api/v1/accounts")
        assert len(r.json()["accounts"]) <= 20

    def test_list_accounts_filter_by_tier(self):
        r    = client.get("/api/v1/accounts?risk_tier=CRITICAL")
        data = r.json()
        assert data["total"] > 0
        for acc in data["accounts"]:
            assert acc["risk_tier"] == "CRITICAL"

    def test_list_accounts_filter_by_urgency(self):
        r    = client.get("/api/v1/accounts?urgency=IMMEDIATE")
        data = r.json()
        for acc in data["accounts"]:
            assert acc["urgency"] == "IMMEDIATE"

    def test_list_accounts_page_size_cap_422(self):
        r = client.get("/api/v1/accounts?page_size=200")
        assert r.status_code == 422

    def test_list_accounts_pagination_page2(self):
        r1 = client.get("/api/v1/accounts?page=1&page_size=10")
        r2 = client.get("/api/v1/accounts?page=2&page_size=10")
        ids1 = [a["account_id"] for a in r1.json()["accounts"]]
        ids2 = [a["account_id"] for a in r2.json()["accounts"]]
        assert set(ids1).isdisjoint(set(ids2))


# ─────────────────────────────────────────────
# TEST TOP ACCOUNTS
# ─────────────────────────────────────────────

class TestTopAccounts:

    def test_top_accounts_200(self):
        r = client.get("/api/v1/accounts/top")
        assert r.status_code == 200

    def test_top_accounts_default_limit_20(self):
        r = client.get("/api/v1/accounts/top")
        assert len(r.json()) <= 20

    def test_top_accounts_sorted_by_priority(self):
        r      = client.get("/api/v1/accounts/top?limit=10")
        scores = [a["priority_score"] for a in r.json()]
        assert scores == sorted(scores, reverse=True)

    def test_top_accounts_limit_cap_422(self):
        r = client.get("/api/v1/accounts/top?limit=100")
        assert r.status_code == 422


# ─────────────────────────────────────────────
# TEST SEARCH
# ─────────────────────────────────────────────

class TestAccountSearch:

    def test_search_returns_results(self):
        r    = client.get("/api/v1/accounts/search?q=Company")
        data = r.json()
        assert r.status_code == 200
        assert data["total"] > 0

    def test_search_query_too_short_422(self):
        r = client.get("/api/v1/accounts/search?q=A")
        assert r.status_code == 422

    def test_search_no_results(self):
        r    = client.get("/api/v1/accounts/search?q=ZZZNOMATCH999")
        data = r.json()
        assert data["total"] == 0

    def test_search_limit_cap_422(self):
        r = client.get("/api/v1/accounts/search?q=Company&limit=100")
        assert r.status_code == 422


# ─────────────────────────────────────────────
# TEST SINGLE ACCOUNT
# ─────────────────────────────────────────────

class TestAccountGet:

    def _get_first_id(self):
        r = client.get("/api/v1/accounts?page_size=1")
        return r.json()["accounts"][0]["account_id"]

    def test_get_account_200(self):
        account_id = self._get_first_id()
        r = client.get(f"/api/v1/accounts/{account_id}")
        assert r.status_code == 200

    def test_get_account_not_found_404(self):
        r = client.get("/api/v1/accounts/INVALID-99999")
        assert r.status_code == 404

    def test_get_account_has_all_fields(self):
        account_id = self._get_first_id()
        data       = client.get(f"/api/v1/accounts/{account_id}").json()
        required   = [
            "account_id", "account_name", "total_mrr",
            "risk_score", "risk_tier", "risk_type",
            "churn_probability", "revenue_at_risk", "health_score", "clv",
            "recommended_action", "action_owner", "urgency",
            "confidence_level", "reason", "expected_recovery",
            "rule_triggered",
        ]
        for field in required:
            assert field in data

    def test_get_account_risk_endpoint(self):
        account_id = self._get_first_id()
        r          = client.get(f"/api/v1/accounts/{account_id}/risk")
        assert r.status_code == 200
        data = r.json()
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 100

    def test_get_account_risk_tier_valid(self):
        account_id = self._get_first_id()
        data       = client.get(f"/api/v1/accounts/{account_id}/risk").json()
        assert data["risk_tier"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

    def test_get_account_recommendation_endpoint(self):
        account_id = self._get_first_id()
        r          = client.get(
            f"/api/v1/accounts/{account_id}/recommendation"
        )
        assert r.status_code == 200
        data = r.json()
        assert "recommended_action" in data
        assert "action_owner" in data
        assert data["expected_recovery"] >= 0


# ─────────────────────────────────────────────
# TEST PORTFOLIO
# ─────────────────────────────────────────────

class TestPortfolio:

    def test_portfolio_summary_200(self):
        r = client.get("/api/v1/portfolio/summary")
        assert r.status_code == 200

    def test_portfolio_summary_has_required_keys(self):
        data = client.get("/api/v1/portfolio/summary").json()
        for key in [
            "total_accounts", "tier_distribution",
            "total_mrr", "total_revenue_at_risk",
            "total_recoverable", "pct_recoverable", "generated_at"
        ]:
            assert key in data

    def test_portfolio_summary_accounts_count(self):
        data = client.get("/api/v1/portfolio/summary").json()
        assert data["total_accounts"] == 500

    def test_portfolio_summary_cached(self):
        r1 = client.get("/api/v1/portfolio/summary").json()
        r2 = client.get("/api/v1/portfolio/summary").json()
        assert r1["total_mrr"] == r2["total_mrr"]

    def test_portfolio_critical_200(self):
        r = client.get("/api/v1/portfolio/critical")
        assert r.status_code == 200

    def test_portfolio_critical_only_critical(self):
        data = client.get("/api/v1/portfolio/critical").json()
        for acc in data["accounts"]:
            assert acc["risk_tier"] == "CRITICAL"


# ─────────────────────────────────────────────
# TEST CACHE
# ─────────────────────────────────────────────

class TestCacheEndpoints:

    def test_cache_status_200(self):
        r = client.get("/api/v1/cache/status")
        assert r.status_code == 200

    def test_cache_status_has_keys(self):
        data = client.get("/api/v1/cache/status").json()
        for key in ["cache_active", "last_refreshed", "accounts_cached"]:
            assert key in data

    def test_cache_refresh_200(self):
        r    = client.post("/api/v1/cache/refresh")
        data = r.json()
        assert r.status_code == 200
        assert data["cache_cleared"] is True
        assert "refreshed_at" in data


# ─────────────────────────────────────────────
# TEST REAL-TIME SCORING
# ─────────────────────────────────────────────

class TestRealTimeScoring:

    VALID_PAYLOAD = {
        "account_id":               "TEST-001",
        "total_mrr":                25000,
        "churn_probability":        0.75,
        "revenue_at_risk":          18750,
        "health_score":             35.0,
        "clv":                      210000,
        "engagement_state":         -1,
        "support_pressure_signal":  0,
        "revenue_change_signal":    0.1,
        "tenure_months":            14,
        "ticket_count":             8,
        "unique_features_used":     5,
        "total_usage":              150,
        "churn_flag":               0,
        "auto_renew_ratio":         0.8,
    }

    def test_score_valid_account_200(self):
        r = client.post("/api/v1/accounts/score", json=self.VALID_PAYLOAD)
        assert r.status_code == 200

    def test_score_response_has_scored_at(self):
        r    = client.post("/api/v1/accounts/score", json=self.VALID_PAYLOAD)
        data = r.json()
        assert "scored_at" in data
        assert len(data["scored_at"]) > 10

    def test_score_risk_score_in_range(self):
        r    = client.post("/api/v1/accounts/score", json=self.VALID_PAYLOAD)
        data = r.json()
        assert 0 <= data["risk_score"] <= 100

    def test_score_missing_field_422(self):
        bad = self.VALID_PAYLOAD.copy()
        del bad["churn_probability"]
        r = client.post("/api/v1/accounts/score", json=bad)
        assert r.status_code == 422

    def test_score_invalid_churn_prob_422(self):
        bad = {**self.VALID_PAYLOAD, "churn_probability": 1.5}
        r   = client.post("/api/v1/accounts/score", json=bad)
        assert r.status_code == 422
