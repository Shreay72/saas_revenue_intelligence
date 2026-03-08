"""
Account Explorer Page
SaaS Revenue Intelligence Dashboard - Week 6
"""

import pandas as pd
import streamlit as st

from dashboard.api_client import client
from dashboard.utils.formatters import (
    format_currency, format_percent, format_score,
    tier_badge, urgency_badge,
    style_accounts_table,
)


def render():
    st.title("🔍 Account Explorer")
    st.markdown("Search, filter, and drill into any account.")

    # ── API Health Guard ───────────────────────
    if not client.is_healthy():
        st.error("⚠️ API is not reachable. Run: `python scripts/run_api.py`")
        st.stop()

    # ─────────────────────────────────────────────
    # SEARCH BAR
    # ─────────────────────────────────────────────

    search_q = st.text_input(
        "🔍 Search by account name",
        value=st.session_state.search_query,
        placeholder="Type company name...",
        key="search_query",
    )

    # ─────────────────────────────────────────────
    # FILTER PANEL
    # ─────────────────────────────────────────────

    with st.expander("🔧 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            risk_tier = st.selectbox(
                "Risk Tier",
                ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                index=["ALL","CRITICAL","HIGH","MEDIUM","LOW"]
                      .index(st.session_state.risk_tier),
                key="risk_tier",
            )
            rule = st.selectbox(
                "Rule Triggered",
                ["ALL", "VP_ESCALATION", "WINBACK", "MONITOR"],
                index=["ALL","VP_ESCALATION","WINBACK","MONITOR"]
                      .index(st.session_state.rule),
                key="rule",
            )

        with col2:
            urgency = st.selectbox(
                "Urgency",
                ["ALL", "IMMEDIATE", "HIGH", "MEDIUM", "LOW"],
                index=["ALL","IMMEDIATE","HIGH","MEDIUM","LOW"]
                      .index(st.session_state.urgency),
                key="urgency",
            )
            page_size = st.selectbox(
                "Page Size",
                [10, 20, 50, 100],
                index=[10, 20, 50, 100].index(st.session_state.page_size),
                key="page_size",
            )

        with col3:
            min_mrr = st.number_input(
                "Min MRR ($)",
                min_value=0,
                max_value=500000,
                value=st.session_state.min_mrr,
                step=1000,
                key="min_mrr",
            )
            max_mrr = st.number_input(
                "Max MRR ($)",
                min_value=0,
                max_value=500000,
                value=st.session_state.max_mrr,
                step=1000,
                key="max_mrr",
            )

    # Reset page to 1 if filters changed
    if st.button("🔍 Apply Filters"):
        st.session_state.page = 1
        st.rerun()

    # ─────────────────────────────────────────────
    # FETCH DATA
    # ─────────────────────────────────────────────

    with st.spinner("Loading accounts..."):
        if search_q and len(search_q) >= 2:
            # Search mode
            accounts = client.search_accounts(q=search_q, limit=50)
            total        = len(accounts)
            total_pages  = 1
            current_page = 1
        else:
            # Filter mode
            result = client.get_accounts(
                risk_tier=st.session_state.risk_tier,
                urgency=st.session_state.urgency,
                rule_triggered=st.session_state.rule,
                min_mrr=float(st.session_state.min_mrr)
                        if st.session_state.min_mrr > 0 else None,
                max_mrr=float(st.session_state.max_mrr)
                        if st.session_state.max_mrr < 500000 else None,
                page=st.session_state.page,
                page_size=st.session_state.page_size,
            )
            accounts     = result.get("accounts", [])
            total        = result.get("total", 0)
            total_pages  = result.get("total_pages", 1)
            current_page = result.get("page", 1)

    # ─────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────

    if not accounts:
        st.info("No accounts match these filters.")
        return

    st.caption(f"**{total} accounts found** | Page {current_page} of {total_pages}")

    # ─────────────────────────────────────────────
    # TABLE + DOWNLOAD
    # ─────────────────────────────────────────────

    display_cols = [
        "account_name", "total_mrr", "risk_score", "risk_tier",
        "churn_probability", "revenue_at_risk",
        "recommended_action", "urgency",
        "expected_recovery", "rule_triggered",
    ]
    df      = pd.DataFrame(accounts)
    disp_df = df[[c for c in display_cols if c in df.columns]]

    col_tbl, col_dl = st.columns([5, 1])
    with col_tbl:
        st.dataframe(
            style_accounts_table(disp_df),
            use_container_width=True,
            height=380,
        )
    with col_dl:
        st.download_button(
            label="📥 CSV",
            data=disp_df.to_csv(index=False),
            file_name="explorer_results.csv",
            mime="text/csv",
        )

    # ─────────────────────────────────────────────
    # PAGINATION
    # ─────────────────────────────────────────────

    if not search_q and total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 3, 1])

        with col_prev:
            if st.button("← Previous") and st.session_state.page > 1:
                st.session_state.page -= 1
                st.rerun()

        with col_info:
            st.markdown(
                f"<div style='text-align:center'>Page "
                f"<b>{current_page}</b> of <b>{total_pages}</b></div>",
                unsafe_allow_html=True,
            )

        with col_next:
            if st.button("Next →") and st.session_state.page < total_pages:
                st.session_state.page += 1
                st.rerun()

    st.markdown("---")

    # ─────────────────────────────────────────────
    # ACCOUNT DRILL-DOWN
    # ─────────────────────────────────────────────

    st.subheader("🔎 Account Detail")
    account_names = [a.get("account_name", "") for a in accounts]
    selected_name = st.selectbox(
        "Select account to inspect",
        ["— select —"] + account_names,
    )

    if selected_name != "— select —":
        row = next(
            (a for a in accounts if a.get("account_name") == selected_name),
            None
        )
        if row:
            _render_account_card(row)


def _render_account_card(acc: dict):
    """Render a detailed account intelligence card."""
    tier  = acc.get("risk_tier", "LOW")
    color = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}.get(tier,"⚪")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {acc.get('account_name', 'Unknown')}  {color} {tier}")
        st.markdown(f"**MRR:** {format_currency(acc.get('total_mrr', 0))}")
        st.markdown(f"**CLV:** {format_currency(acc.get('clv', 0))}")
        st.markdown(f"**Risk Score:** {format_score(acc.get('risk_score', 0))}")
        st.markdown(f"**Health Score:** {format_score(acc.get('health_score', 0))}")
        st.markdown(f"**Churn Probability:** {format_percent(acc.get('churn_probability', 0))}")
        st.markdown(f"**Revenue at Risk:** {format_currency(acc.get('revenue_at_risk', 0))}")

    with col2:
        st.markdown("#### 🎯 Recommended Action")
        st.markdown(f"**Action:** {acc.get('recommended_action', '')}")
        st.markdown(f"**Owner:** {acc.get('action_owner', '')}")
        st.markdown(f"**Urgency:** {urgency_badge(acc.get('urgency', 'LOW'))}")
        st.markdown(f"**Confidence:** {acc.get('confidence_level', '')}")
        st.markdown(f"**Recovery:** {format_currency(acc.get('expected_recovery', 0))}")
        st.markdown(f"**Rule:** `{acc.get('rule_triggered', '')}`")
        st.info(f"📝 {acc.get('reason', '')}")
