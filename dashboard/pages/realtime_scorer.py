"""
Real-Time Scorer Page
SaaS Revenue Intelligence Dashboard - Week 6
"""

import streamlit as st

from dashboard.api_client import client
from dashboard.utils.formatters import (
    format_currency, format_score, format_percent,
    tier_badge, urgency_badge, TIER_COLORS,
)


def render():
    st.title("⚡ Real-Time Account Scorer")
    st.markdown(
        "Score any account through the full risk + recommendation pipeline instantly."
    )

    # ── API Health Guard ───────────────────────
    if not client.is_healthy():
        st.error("⚠️ API is not reachable. Run: `python scripts/run_api.py`")
        st.stop()

    # ─────────────────────────────────────────────
    # INPUT FORM
    # ─────────────────────────────────────────────

    with st.form("scorer_form"):
        st.subheader("📋 Account Details")

        col1, col2 = st.columns(2)

        with col1:
            account_id      = st.text_input("Account ID", value="TEST-001")
            total_mrr       = st.number_input("MRR ($)", min_value=0.0,
                                               value=25000.0, step=500.0)
            churn_prob      = st.slider("Churn Probability",
                                         0.0, 1.0, 0.75, 0.01)
            health_score    = st.slider("Health Score",
                                         0.0, 100.0, 35.0, 1.0)
            clv             = st.number_input("CLV ($)", min_value=0.0,
                                               value=210000.0, step=1000.0)
            revenue_at_risk = st.number_input("Revenue at Risk ($)",
                                               min_value=0.0,
                                               value=18750.0, step=500.0)
            engagement_state = st.selectbox(
                "Engagement State",
                [-1, 0, 1],
                index=1,
                format_func=lambda x: {-1:"↓ Declining",0:"→ Neutral",1:"↑ Improving"}[x],
            )
            support_pressure = st.selectbox(
                "Support Pressure",
                [0, 1],
                format_func=lambda x: {0:"Normal",1:"High Pressure"}[x],
            )

        with col2:
            revenue_signal  = st.slider("Revenue Change Signal",
                                         -1.0, 1.0, 0.1, 0.1)
            tenure_months   = st.number_input("Tenure (months)",
                                               min_value=0.0,
                                               value=14.0, step=1.0)
            ticket_count    = st.number_input("Support Tickets",
                                               min_value=0,
                                               value=8, step=1)
            features_used   = st.number_input("Unique Features Used",
                                               min_value=0,
                                               value=5, step=1)
            total_usage     = st.number_input("Total Usage Events",
                                               min_value=0,
                                               value=150, step=10)
            churn_flag      = st.selectbox(
                "Churn Flag",
                [0, 1],
                format_func=lambda x: {0:"Active",1:"Churned"}[x],
            )
            auto_renew      = st.slider("Auto Renew Ratio",
                                         0.0, 1.0, 0.8, 0.05)

        submitted = st.form_submit_button(
            "🎯 Score This Account",
            use_container_width=True,
            type="primary",
        )

    # ─────────────────────────────────────────────
    # SCORING + RESULT
    # ─────────────────────────────────────────────

    if submitted:
        payload = {
            "account_id":               account_id,
            "total_mrr":                total_mrr,
            "churn_probability":        churn_prob,
            "revenue_at_risk":          revenue_at_risk,
            "health_score":             health_score,
            "clv":                      clv,
            "engagement_state":         int(engagement_state),
            "support_pressure_signal":  int(support_pressure),
            "revenue_change_signal":    revenue_signal,
            "tenure_months":            tenure_months,
            "ticket_count":             int(ticket_count),
            "unique_features_used":     int(features_used),
            "total_usage":              int(total_usage),
            "churn_flag":               int(churn_flag),
            "auto_renew_ratio":         auto_renew,
        }

        with st.spinner("Scoring account through full pipeline..."):
            result = client.score_account(payload)

        if "error" in result:
            st.error(f"❌ Scoring failed: {result.get('error', 'Unknown error')}")
            return

        # ── Result Card ──────────────────────────
        tier  = result.get("risk_tier", "LOW")
        color = {"CRITICAL":"#FF4B4B","HIGH":"#FF8C00",
                 "MEDIUM":"#FFD700","LOW":"#00CC44"}.get(tier, "#CCC")

        st.markdown("---")
        st.subheader("📊 Scoring Result")

        st.markdown(
            f"""
            <div style="
                border: 2px solid {color};
                border-radius: 10px;
                padding: 20px;
                background-color: #1E1E1E;
            ">
            <h3 style="color:{color}">
                {tier_badge(tier)} — Risk Score: {format_score(result.get('risk_score', 0))}
            </h3>
            <p><b>Risk Type:</b> {result.get('risk_type', '')}</p>
            <hr style="border-color:{color}">
            <p>🎯 <b>Action:</b> {result.get('recommended_action', '')}</p>
            <p>👤 <b>Owner:</b> {result.get('action_owner', '')}</p>
            <p>{urgency_badge(result.get('urgency', 'LOW'))} <b>Urgency:</b> {result.get('urgency', '')}</p>
            <p>🔒 <b>Confidence:</b> {result.get('confidence_level', '')}</p>
            <p>💰 <b>Expected Recovery:</b> {format_currency(result.get('expected_recovery', 0))}</p>
            <p>📋 <b>Rule:</b> <code>{result.get('rule_triggered', '')}</code></p>
            <p>📝 <b>Reason:</b> {result.get('reason', '')}</p>
            <p>🕐 <b>Scored At:</b> {result.get('scored_at', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
