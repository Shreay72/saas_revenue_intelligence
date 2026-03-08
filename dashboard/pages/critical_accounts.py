"""
Critical Accounts Page
SaaS Revenue Intelligence Dashboard - Week 6
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.api_client import client
from dashboard.utils.formatters import (
    TIER_COLORS, RULE_COLORS,
    format_large_currency,
    style_accounts_table,
    tier_badge, urgency_badge,
)


def render():
    st.title("🚨 Critical Accounts")
    st.markdown("All CRITICAL tier accounts — sorted by priority score.")

    # ── API Health Guard ───────────────────────
    if not client.is_healthy():
        st.error("⚠️ API is not reachable. Run: `python scripts/run_api.py`")
        st.stop()

    # ── Fetch ONCE ────────────────────────────
    with st.spinner("Loading critical accounts..."):
        accounts = client.get_critical_accounts()

    if not accounts:
        st.info("No CRITICAL accounts found.")
        return

    df = pd.DataFrame(accounts)

    # ─────────────────────────────────────────────
    # SUMMARY BANNER
    # ─────────────────────────────────────────────

    total_at_risk   = df["revenue_at_risk"].sum() \
        if "revenue_at_risk" in df.columns else 0
    total_recoverable = df["expected_recovery"].sum() \
        if "expected_recovery" in df.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("🚨 Critical Accounts", len(df))
    col2.metric("⚠️ Total at Risk",     format_large_currency(total_at_risk))
    col3.metric("💚 Total Recoverable", format_large_currency(total_recoverable))

    st.markdown("---")

    # ─────────────────────────────────────────────
    # FILTERS
    # ─────────────────────────────────────────────

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        rules = ["ALL"] + sorted(df["rule_triggered"].unique().tolist()) \
            if "rule_triggered" in df.columns else ["ALL"]
        selected_rule = st.selectbox("Filter by Action", rules)

    with col_f2:
        urgencies = ["ALL"] + sorted(df["urgency"].unique().tolist()) \
            if "urgency" in df.columns else ["ALL"]
        selected_urgency = st.selectbox("Filter by Urgency", urgencies)

    # Apply filters
    filtered = df.copy()
    if selected_rule != "ALL" and "rule_triggered" in filtered.columns:
        filtered = filtered[filtered["rule_triggered"] == selected_rule]
    if selected_urgency != "ALL" and "urgency" in filtered.columns:
        filtered = filtered[filtered["urgency"] == selected_urgency]

    st.caption(f"Showing {len(filtered)} of {len(df)} critical accounts")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # TABLE + DOWNLOAD
    # ─────────────────────────────────────────────

    display_cols = [
        "account_name", "total_mrr", "risk_score",
        "churn_probability", "revenue_at_risk",
        "recommended_action", "action_owner",
        "urgency", "expected_recovery", "rule_triggered",
    ]
    disp_df = filtered[[c for c in display_cols if c in filtered.columns]]

    col_tbl, col_dl = st.columns([5, 1])

    with col_tbl:
        st.dataframe(
            style_accounts_table(disp_df),
            use_container_width=True,
            height=450,
        )

    with col_dl:
        st.download_button(
            label="📥 CSV",
            data=disp_df.to_csv(index=False),
            file_name="critical_accounts.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ─────────────────────────────────────────────
    # MINI CHART — Action Breakdown
    # ─────────────────────────────────────────────

    if "rule_triggered" in df.columns:
        st.subheader("🎯 Action Breakdown within Critical Accounts")
        rule_counts = (
            df["rule_triggered"].value_counts()
            .reset_index()
        )
        rule_counts.columns = ["Action", "Count"]

        fig = px.bar(
            rule_counts,
            x="Action",
            y="Count",
            color="Action",
            color_discrete_map=RULE_COLORS,
            text="Count",
        )
        fig.update_layout(
            showlegend=False,
            height=280,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
