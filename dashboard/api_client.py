"""
API Client — Single Source of Truth
SaaS Revenue Intelligence Dashboard - Week 6

All HTTP calls go through this class.
Features:
    - timeout=5 on every request
    - 3 retries on timeout/connection error
    - No retry on 4xx errors
    - Safe empty return on final failure
    - @st.cache_data(ttl=60) on heavy endpoints
"""

import time
import streamlit as st
import requests
from typing import Optional


BASE_URL = "http://localhost:8000"
TIMEOUT  = 5      # seconds
RETRIES  = 3
RETRY_SLEEP = 1   # seconds between retries


class APIClient:

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url

    # ─────────────────────────────────────────────
    # CORE HTTP METHODS (with retry + timeout)
    # ─────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict = None) -> dict | list:
        url        = f"{self.base_url}{endpoint}"
        last_error = None

        for attempt in range(RETRIES):
            try:
                r = requests.get(url, params=params, timeout=TIMEOUT)
                r.raise_for_status()
                return r.json()

            except requests.exceptions.Timeout:
                last_error = "timeout"
                time.sleep(RETRY_SLEEP)

            except requests.exceptions.ConnectionError:
                last_error = "connection_error"
                time.sleep(RETRY_SLEEP)

            except requests.exceptions.HTTPError as e:
                # 4xx — real error, don't retry
                code = e.response.status_code if e.response else 0
                return {"error": f"http_{code}", "detail": str(e)}

        return {"error": last_error}

    def _post(self, endpoint: str, payload: dict) -> dict:
        url        = f"{self.base_url}{endpoint}"
        last_error = None

        for attempt in range(RETRIES):
            try:
                r = requests.post(url, json=payload, timeout=TIMEOUT)
                r.raise_for_status()
                return r.json()

            except requests.exceptions.Timeout:
                last_error = "timeout"
                time.sleep(RETRY_SLEEP)

            except requests.exceptions.ConnectionError:
                last_error = "connection_error"
                time.sleep(RETRY_SLEEP)

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response else 0
                return {"error": f"http_{code}", "detail": str(e)}

        return {"error": last_error}

    # ─────────────────────────────────────────────
    # HEALTH
    # ─────────────────────────────────────────────

    def get_health(self) -> dict:
        return self._get("/health")

    def is_healthy(self) -> bool:
        result = self.get_health()
        return (
            "error" not in result
            and result.get("pipeline_ready", False)
            and result.get("accounts_loaded", 0) > 0
        )

    # ─────────────────────────────────────────────
    # PORTFOLIO (cached 60s)
    # ─────────────────────────────────────────────

    @st.cache_data(ttl=60)
    def get_portfolio_summary(_self) -> dict:
        return _self._get("/api/v1/portfolio/summary")

    @st.cache_data(ttl=60)
    def get_critical_accounts(_self) -> list:
        result = _self._get("/api/v1/portfolio/critical")
        if isinstance(result, dict):
            return result.get("accounts", [])
        return []

    # ─────────────────────────────────────────────
    # ACCOUNTS (cached 60s)
    # ─────────────────────────────────────────────

    @st.cache_data(ttl=60)
    def get_top_accounts(_self, limit: int = 20) -> list:
        result = _self._get("/api/v1/accounts/top", params={"limit": limit})
        if isinstance(result, list):
            return result
        return []

    @st.cache_data(ttl=60)
    def get_accounts(
        _self,
        risk_tier:      Optional[str]   = None,
        risk_type:      Optional[str]   = None,
        urgency:        Optional[str]   = None,
        rule_triggered: Optional[str]   = None,
        min_mrr:        Optional[float] = None,
        max_mrr:        Optional[float] = None,
        sort_by:        str             = "priority_score",
        sort_order:     str             = "desc",
        page:           int             = 1,
        page_size:      int             = 20,
    ) -> dict:
        params = {
            "sort_by":    sort_by,
            "sort_order": sort_order,
            "page":       page,
            "page_size":  page_size,
        }
        if risk_tier and risk_tier != "ALL":
            params["risk_tier"] = risk_tier
        if risk_type and risk_type != "ALL":
            params["risk_type"] = risk_type
        if urgency and urgency != "ALL":
            params["urgency"] = urgency
        if rule_triggered and rule_triggered != "ALL":
            params["rule_triggered"] = rule_triggered
        if min_mrr is not None:
            params["min_mrr"] = min_mrr
        if max_mrr is not None:
            params["max_mrr"] = max_mrr

        result = _self._get("/api/v1/accounts", params=params)
        if isinstance(result, dict) and "accounts" in result:
            return result
        return {"accounts": [], "total": 0, "total_pages": 1,
                "page": 1, "page_size": page_size}

    def get_account(self, account_id: str) -> dict:
        result = self._get(f"/api/v1/accounts/{account_id}")
        return result if "error" not in result else {}

    @st.cache_data(ttl=60)
    def search_accounts(_self, q: str, limit: int = 20) -> list:
        result = _self._get(
            "/api/v1/accounts/search",
            params={"q": q, "limit": limit}
        )
        if isinstance(result, dict):
            return result.get("accounts", [])
        return []

    # ─────────────────────────────────────────────
    # REAL-TIME SCORING (no cache)
    # ─────────────────────────────────────────────

    def score_account(self, payload: dict) -> dict:
        return self._post("/api/v1/accounts/score", payload)

    # ─────────────────────────────────────────────
    # CACHE MANAGEMENT
    # ─────────────────────────────────────────────

    def refresh_cache(self) -> dict:
        return self._post("/api/v1/cache/refresh", {})

    def clear_local_cache(self):
        """Clear Streamlit's local cache for all cached methods."""
        self.get_portfolio_summary.clear()
        self.get_critical_accounts.clear()
        self.get_top_accounts.clear()
        self.get_accounts.clear()
        self.search_accounts.clear()


# ── Singleton ─────────────────────────────────────
client = APIClient()
