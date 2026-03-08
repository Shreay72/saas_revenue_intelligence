"""
SaaS Revenue Intelligence Dashboard
Week 6 — Main Entry Point

Run:
    streamlit run dashboard/main.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.api_client import client
from dashboard.utils.formatters import format_timestamp, tier_badge, TIER_COLORS

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SaaS Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SESSION STATE — initialize once
# ─────────────────────────────────────────────

DEFAULTS = {
    "risk_tier":    "ALL",
    "risk_type":    "ALL",
    "urgency":      "ALL",
    "rule":         "ALL",
    "min_mrr":      0,
    "max_mrr":      200000,
    "page":         1,
    "page_size":    20,
    "search_query": "",
    "selected_page": "🏠 Portfolio Overview",
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Revenue Intelligence")
    st.markdown("---")

    # ── Page Navigation ────────────────────────
    pages = [
        "🏠 Portfolio Overview",
        "🚨 Critical Accounts",
        "🔍 Account Explorer",
        "📊 Risk Analytics",
        "⚡ Real-Time Scorer",
    ]
    selected_page = st.radio(
        "Navigate",
        pages,
        index=pages.index(st.session_state.selected_page),
        key="selected_page",
    )

    st.markdown("---")

    # ── API Health ─────────────────────────────
    health = client.get_health()
    if "error" not in health and health.get("pipeline_ready"):
        st.success(f"🟢 API Online — {health.get('accounts_loaded', 0)} accounts")
    else:
        st.error("🔴 API Offline")

    # ── Last Updated ───────────────────────────
    try:
        summary = client.get_portfolio_summary()
        ts      = summary.get("generated_at", "")
        if ts:
            st.caption(f"🕐 Last updated: {format_timestamp(ts)}")
    except Exception:
        pass

    st.markdown("---")

    # ── Auto Refresh ───────────────────────────
    auto_refresh = st.checkbox("🔄 Auto refresh (30s)", value=False)
    if st.button("🔄 Refresh Now"):
        client.clear_local_cache()
        st.rerun()

    st.markdown("---")

    # ── Risk Tier Legend ───────────────────────
    st.markdown("**RISK TIER LEGEND**")
    st.markdown("🔴 **CRITICAL**")
    st.markdown("🟠 **HIGH**")
    st.markdown("🟡 **MEDIUM**")
    st.markdown("🟢 **LOW**")

    st.markdown("---")
    st.caption("SaaS Revenue Intelligence v1.0.0")

# ─────────────────────────────────────────────
# AUTO REFRESH LOGIC
# ─────────────────────────────────────────────

if auto_refresh:
    time.sleep(30)
    client.clear_local_cache()
    st.rerun()

# ─────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────

if selected_page == "🏠 Portfolio Overview":
    from dashboard.pages.portfolio_overview import render
    render()

elif selected_page == "🚨 Critical Accounts":
    from dashboard.pages.critical_accounts import render
    render()

elif selected_page == "🔍 Account Explorer":
    from dashboard.pages.account_explorer import render
    render()

elif selected_page == "📊 Risk Analytics":
    from dashboard.pages.risk_analytics import render
    render()

elif selected_page == "⚡ Real-Time Scorer":
    from dashboard.pages.realtime_scorer import render
    render()
