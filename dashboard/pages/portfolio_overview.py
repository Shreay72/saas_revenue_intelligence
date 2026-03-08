"""
Portfolio Overview Page
SaaS Revenue Intelligence Dashboard - Week 6
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import client
from dashboard.utils.formatters import (
    TIER_COLORS, RULE_COLORS,
    format_large_currency, format_currency,
    format_percent, format_trend,
    prepare_display_df, style_accounts_table,
)


def render():
    st.title("🏠 Portfolio Overview")
    st.markdown("Real-time SaaS revenue risk intelligence across all accounts.")

    # ── API Health Guard ───────────────────────
    if not client.is_healthy():
        st.error("⚠️ API is not reachable. Run: `python scripts/run_api.py`")
        st.stop()

    # ── Fetch ONCE — reuse everywhere ─────────
    with st.spinner("Loading portfolio data..."):
        summary  = client.get_portfolio_summary()
        top_accs = client.get_top_accounts(limit=20)

    if "error" in summary:
        st.error("Failed to load portfolio data. Check the API.")
        st.stop()

    # ─────────────────────────────────────────────
    # KPI CARDS
    # ─────────────────────────────────────────────

    total_mrr    = summary.get("total_mrr", 0)
    at_risk      = summary.get("total_revenue_at_risk", 0)
    recoverable  = summary.get("total_recoverable", 0)
    pct_recov    = summary.get("pct_recoverable", 0)
    total_accts  = summary.get("total_accounts", 0)
    tier_dist    = summary.get("tier_distribution", {})
    critical_cnt = tier_dist.get("CRITICAL", 0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Total MRR",
            value=format_large_currency(total_mrr),
            help="Total Monthly Recurring Revenue across all accounts",
        )

    with col2:
        st.metric(
            label="⚠️ Revenue at Risk",
            value=format_large_currency(at_risk),
            help="Expected monthly revenue loss (MRR × churn_probability)",
        )

    with col3:
        st.metric(
            label="💚 Recoverable",
            value=format_large_currency(recoverable),
            delta=f"{pct_recov:.1f}% of at-risk",
            delta_color="normal",
            help="Expected recovery if interventions succeed",
        )

    with col4:
        st.metric(
            label="🚨 Critical Accounts",
            value=f"{critical_cnt}",
            delta=f"of {total_accts} total",
            delta_color="off",
            help="Accounts in CRITICAL risk tier",
        )

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CHARTS ROW 1
    # ─────────────────────────────────────────────

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Risk Tier Distribution")
        labels = list(tier_dist.keys())
        values = list(tier_dist.values())
        colors = [TIER_COLORS.get(t, "#CCCCCC") for t in labels]

        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=colors),
            textinfo="label+percent+value",
        )])
        fig_donut.update_layout(
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.subheader("🎯 Action Distribution")
        action_dist = summary.get("action_distribution", {})
        if action_dist:
            action_df = pd.DataFrame({
                "Action":  list(action_dist.keys()),
                "Count":   list(action_dist.values()),
            }).sort_values("Count", ascending=True)

            colors_bar = [RULE_COLORS.get(a, "#4B9EFF") for a in action_df["Action"]]

            fig_bar = px.bar(
                action_df,
                x="Count",
                y="Action",
                orientation="h",
                color="Action",
                color_discrete_map=RULE_COLORS,
                title="Accounts per Recommended Action",
            )
            fig_bar.update_layout(
                showlegend=False,
                margin=dict(t=30, b=20, l=20, r=20),
                height=320,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CHART ROW 2 — MRR by Tier
    # ─────────────────────────────────────────────

    st.subheader("💰 Revenue Distribution by Risk Tier")

    if top_accs:
        top_df = pd.DataFrame(top_accs)
        if "risk_tier" in top_df.columns and "total_mrr" in top_df.columns:
            tier_mrr = top_df.groupby("risk_tier").agg(
                total_mrr=("total_mrr", "sum"),
                revenue_at_risk=("revenue_at_risk", "sum"),
            ).reset_index()

            fig_stacked = go.Figure()
            fig_stacked.add_trace(go.Bar(
                x=tier_mrr["risk_tier"],
                y=tier_mrr["total_mrr"],
                name="Total MRR",
                marker_color=[TIER_COLORS.get(t, "#CCC")
                               for t in tier_mrr["risk_tier"]],
                opacity=0.8,
            ))
            fig_stacked.add_trace(go.Bar(
                x=tier_mrr["risk_tier"],
                y=tier_mrr["revenue_at_risk"],
                name="Revenue at Risk",
                marker_color="#FF4B4B",
                opacity=0.5,
            ))
            fig_stacked.update_layout(
                barmode="overlay",
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis_title="Amount ($)",
            )
            st.plotly_chart(fig_stacked, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # TOP 20 ACCOUNTS TABLE
    # ─────────────────────────────────────────────

    st.subheader("🏆 Top 20 Accounts by Priority Score")

    if top_accs:
        display_cols = [
            "account_name", "total_mrr", "risk_score", "risk_tier",
            "churn_probability", "revenue_at_risk",
            "recommended_action", "urgency", "expected_recovery",
        ]
        top_df   = pd.DataFrame(top_accs)
        disp_df  = top_df[[c for c in display_cols if c in top_df.columns]]

        col_table, col_dl = st.columns([5, 1])
        with col_table:
            st.dataframe(
                style_accounts_table(disp_df),
                use_container_width=True,
                height=400,
            )
        with col_dl:
            st.download_button(
                label="📥 CSV",
                data=disp_df.to_csv(index=False),
                file_name="top_accounts.csv",
                mime="text/csv",
            )
    else:
        st.info("No account data available.")
