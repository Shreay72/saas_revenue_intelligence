"""
Risk Analytics Page — 5 Charts + Heatmap
SaaS Revenue Intelligence Dashboard - Week 6
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import client
from dashboard.utils.formatters import (
    TIER_COLORS, URGENCY_COLORS,
)


def render():
    st.title("📊 Risk Analytics")
    st.markdown("Portfolio-wide risk intelligence visualizations.")

    # ── API Health Guard ───────────────────────
    if not client.is_healthy():
        st.error("⚠️ API is not reachable. Run: `python scripts/run_api.py`")
        st.stop()

    # ── Fetch ONCE ────────────────────────────
    with st.spinner("Loading analytics data..."):
        result = client.get_accounts(page_size=100, page=1)
        all_pages = [result.get("accounts", [])]
        total_pages = result.get("total_pages", 1)

        for p in range(2, min(total_pages + 1, 6)):
            page_result = client.get_accounts(page_size=100, page=p)
            all_pages.append(page_result.get("accounts", []))

    all_accounts = [acc for page in all_pages for acc in page]

    if not all_accounts:
        st.info("No data available.")
        return

    df = pd.DataFrame(all_accounts)

    # ── Download button ────────────────────────
    st.download_button(
        label="📥 Download Analytics Data (CSV)",
        data=df.to_csv(index=False),
        file_name="analytics_data.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CHART 1 — Risk Score Distribution
    # ─────────────────────────────────────────────

    st.subheader("1️⃣ Risk Score Distribution")

    if "risk_score" in df.columns:
        fig1 = px.histogram(
            df,
            x="risk_score",
            nbins=30,
            color="risk_tier",
            color_discrete_map=TIER_COLORS,
            title="Distribution of Risk Scores across Portfolio",
            labels={"risk_score": "Risk Score (0–100)", "count": "Accounts"},
            category_orders={"risk_tier": ["CRITICAL","HIGH","MEDIUM","LOW"]},
        )
        fig1.update_layout(height=350, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CHART 2 — Churn Probability vs Revenue at Risk
    # ─────────────────────────────────────────────

    st.subheader("2️⃣ Churn Probability vs Revenue at Risk")

    if all(c in df.columns for c in
           ["churn_probability", "revenue_at_risk", "risk_tier", "clv"]):
        fig2 = px.scatter(
            df,
            x="churn_probability",
            y="revenue_at_risk",
            color="risk_tier",
            size="clv",
            size_max=25,
            color_discrete_map=TIER_COLORS,
            hover_data=["account_name", "total_mrr",
                        "recommended_action", "action_owner"],
            title="Churn Probability vs Revenue at Risk (size = CLV)",
            labels={
                "churn_probability": "Churn Probability",
                "revenue_at_risk":   "Revenue at Risk ($)",
            },
            category_orders={"risk_tier": ["CRITICAL","HIGH","MEDIUM","LOW"]},
        )
        fig2.update_layout(height=420, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CHART 3 — CLV vs Health Score
    # ─────────────────────────────────────────────

    st.subheader("3️⃣ CLV vs Health Score")

    if all(c in df.columns for c in ["health_score", "clv", "urgency"]):
        fig3 = px.scatter(
            df,
            x="health_score",
            y="clv",
            color="urgency",
            color_discrete_map=URGENCY_COLORS,
            hover_data=["account_name", "rule_triggered", "risk_tier"],
            title="CLV vs Health Score (color = urgency)",
            labels={
                "health_score": "Health Score (0–100)",
                "clv":          "Customer Lifetime Value ($)",
            },
            category_orders={"urgency": ["IMMEDIATE","HIGH","MEDIUM","LOW"]},
        )
        fig3.update_layout(height=380, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CHART 4 — Expected Recovery by Action Type
    # ─────────────────────────────────────────────

    st.subheader("4️⃣ Expected Recovery by Action Type")

    if all(c in df.columns for c in ["rule_triggered", "expected_recovery"]):
        recovery_df = (
            df.groupby("rule_triggered")["expected_recovery"]
            .sum()
            .reset_index()
            .sort_values("expected_recovery", ascending=False)
        )
        recovery_df.columns = ["Action", "Total Recovery ($)"]

        fig4 = px.bar(
            recovery_df,
            x="Action",
            y="Total Recovery ($)",
            color="Action",
            text="Total Recovery ($)",
            title="Total Expected Recovery per Action Type",
        )
        fig4.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
        )
        fig4.update_layout(
            showlegend=False,
            height=350,
            margin=dict(t=40,b=20,l=20,r=80),
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CHART 5 — Revenue at Risk vs MRR
    # ─────────────────────────────────────────────

    st.subheader("5️⃣ Revenue at Risk vs MRR by Tier")

    if all(c in df.columns for c in
           ["total_mrr", "revenue_at_risk", "risk_tier"]):
        fig5 = px.scatter(
            df,
            x="total_mrr",
            y="revenue_at_risk",
            color="risk_tier",
            color_discrete_map=TIER_COLORS,
            hover_data=["account_name", "churn_probability", "risk_score"],
            title="Revenue at Risk vs MRR (by Risk Tier)",
            labels={
                "total_mrr":       "Total MRR ($)",
                "revenue_at_risk": "Revenue at Risk ($)",
            },
            category_orders={"risk_tier": ["CRITICAL","HIGH","MEDIUM","LOW"]},
        )
        fig5.update_layout(height=380, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CHART 6 — Risk Heatmap (Risk Type × Risk Tier)
    # ─────────────────────────────────────────────

    st.subheader("6️⃣ Risk Concentration Heatmap")

    if all(c in df.columns for c in ["risk_type", "risk_tier"]):
        heatmap_data = (
            df.groupby(["risk_type", "risk_tier"])
            .size()
            .reset_index(name="count")
            .pivot(index="risk_type", columns="risk_tier", values="count")
            .fillna(0)
        )

        tier_order = [t for t in ["CRITICAL","HIGH","MEDIUM","LOW"]
                      if t in heatmap_data.columns]
        heatmap_data = heatmap_data[tier_order]

        fig6 = px.imshow(
            heatmap_data,
            color_continuous_scale="RdYlGn_r",
            title="Risk Concentration: Risk Type × Risk Tier",
            labels={"color": "Account Count"},
            text_auto=True,
            aspect="auto",
        )
        fig6.update_layout(height=350, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig6, use_container_width=True)
