"""
Week 4 Risk Engine & Recommendations — End-to-End Script
Run:
    python scripts/test_week4_risk.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

from src.pipelines.churn_pipeline import ChurnPredictionPipeline
from src.pipelines.revenue_pipeline import load_revenue_pipeline
from src.recommendation.recommendation_engine import RecommendationEngine

print("=" * 80)
print("🚀 WEEK 4: RISK ENGINE & RECOMMENDATION INTELLIGENCE")
print("=" * 80)

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/5] Loading account-level features...")

df = pd.read_csv("data/processed/account_level_features.csv")
print(f"  Loaded: {df.shape[0]} accounts × {df.shape[1]} features")

# ─────────────────────────────────────────────
# STEP 2 — ATTACH CHURN PROBABILITIES
# ─────────────────────────────────────────────
print("\n[2/5] Scoring churn probabilities (Week 2 model)...")

if "churn_probability" not in df.columns:
    churn_pipeline = ChurnPredictionPipeline(model_dir="models/churn")
    df["churn_probability"] = churn_pipeline.predict_proba(df)

print(f"  churn_probability — mean: {df['churn_probability'].mean():.3f} | "
      f"min: {df['churn_probability'].min():.3f} | "
      f"max: {df['churn_probability'].max():.3f}")

# ─────────────────────────────────────────────
# STEP 3 — ATTACH REVENUE INTELLIGENCE
# ─────────────────────────────────────────────
print("\n[3/5] Running revenue pipeline (Week 3)...")

rev_pipeline = load_revenue_pipeline()
intel = rev_pipeline.generate_account_intelligence(df)

df["predicted_clv"]   = intel["predicted_clv"].values
df["revenue_at_risk"] = intel["revenue_at_risk"].values
df["health_score"]    = intel["health_score"].values
df["clv"]             = intel["predicted_clv"].values

print(f"  Revenue at Risk total: ${df['revenue_at_risk'].sum():,.2f}")
print(f"  Avg health score:      {df['health_score'].mean():.1f}")
print(f"  Avg CLV:               ${df['clv'].mean():,.2f}")

# ─────────────────────────────────────────────
# STEP 4 — GENERATE RISK + RECOMMENDATIONS
# ─────────────────────────────────────────────
print("\n[4/5] Running Risk Engine + Recommendation Engine...")

engine = RecommendationEngine(config_path="config/model_config.yaml")
intelligence = engine.generate_intelligence(df)

print(f"\n  Risk score — mean: {intelligence['risk_score'].mean():.1f} | "
      f"min: {intelligence['risk_score'].min():.1f} | "
      f"max: {intelligence['risk_score'].max():.1f}")

# ─────────────────────────────────────────────
# STEP 5 — PORTFOLIO SUMMARY
# ─────────────────────────────────────────────
print("\n[5/5] Generating Portfolio Summary...")

summary = engine.generate_portfolio_summary(intelligence)

print(f"""
{"=" * 80}
📊 PORTFOLIO RISK INTELLIGENCE REPORT
{"=" * 80}

👥 ACCOUNT RISK DISTRIBUTION ({summary['total_accounts']} accounts)
   CRITICAL:  {summary['tier_distribution'].get('CRITICAL', 0):>4} accounts
   HIGH:      {summary['tier_distribution'].get('HIGH',     0):>4} accounts
   MEDIUM:    {summary['tier_distribution'].get('MEDIUM',   0):>4} accounts
   LOW:       {summary['tier_distribution'].get('LOW',      0):>4} accounts

🏷️  RISK TYPE DISTRIBUTION
   Strategic Risk:  {summary['type_distribution'].get('Strategic Risk', 0):>4}
   Revenue Risk:    {summary['type_distribution'].get('Revenue Risk',   0):>4}
   Support Risk:    {summary['type_distribution'].get('Support Risk',   0):>4}
   Usage Risk:      {summary['type_distribution'].get('Usage Risk',     0):>4}
   Composite Risk:  {summary['type_distribution'].get('Composite Risk', 0):>4}
   Healthy:         {summary['type_distribution'].get('Healthy',        0):>4}

💰 REVENUE INTELLIGENCE
   Total MRR:              ${summary['total_mrr']:>15,.2f}
   Total Revenue at Risk:  ${summary['total_revenue_at_risk']:>15,.2f}
   Total Recoverable:      ${summary['total_recoverable']:>15,.2f}
   % Recoverable:          {summary['pct_recoverable']:>14.1f}%

🎯 ACTION DISTRIBUTION
   VP Escalations:         {summary['action_distribution'].get('VP_ESCALATION',  0):>4}
   Executive QBRs:         {summary['action_distribution'].get('EXECUTIVE_QBR',  0):>4}
   TAM Assignments:        {summary['action_distribution'].get('TAM_ASSIGNMENT', 0):>4}
   Upsell Opportunities:   {summary['action_distribution'].get('UPSELL',         0):>4}
   Win-back Campaigns:     {summary['action_distribution'].get('WINBACK',        0):>4}
   Monitor:                {summary['action_distribution'].get('MONITOR',        0):>4}

⚡ RISK VELOCITY
   Accelerating:           {summary['velocity_distribution'].get('ACCELERATING', 0):>4}
   Stable:                 {summary['velocity_distribution'].get('STABLE',       0):>4}
   Improving:              {summary['velocity_distribution'].get('IMPROVING',    0):>4}
{"=" * 80}
""")

# ─────────────────────────────────────────────
# TOP 10 ACCOUNTS TO SAVE
# ─────────────────────────────────────────────
print("🔥 TOP 10 ACCOUNTS TO SAVE (by priority_score)")
print("-" * 80)

top10 = summary["top_20_accounts"].head(10)[
    [c for c in [
        "account_name", "risk_tier", "risk_type",
        "revenue_at_risk", "expected_recovery",
        "recommended_action", "urgency",
    ] if c in summary["top_20_accounts"].columns]
]
print(top10.to_string(index=False))

# ─────────────────────────────────────────────
# SAVE INTELLIGENCE TABLE
# ─────────────────────────────────────────────
output_path = "data/processed/account_intelligence.csv"
intelligence.to_csv(output_path, index=False)
print(f"\n✅ Intelligence table saved → {output_path}")

print("""
================================================================================
✅ WEEK 4 COMPLETE!
================================================================================

📦 Artifacts Created:
   • data/processed/account_intelligence.csv

🔬 Features Delivered:
   ✅ Composite risk score (config-driven weights, p95 normalization)
   ✅ Risk tiers (CRITICAL/HIGH/MEDIUM/LOW + revenue override)
   ✅ Risk type classification (6 types, deterministic precedence)
   ✅ Risk velocity (schema ready for weekly runs)
   ✅ Priority score (risk_score × revenue_at_risk)
   ✅ Business rules (5 rules + Monitor sub-actions)
   ✅ Action owner assignment
   ✅ Confidence levels (config-driven)
   ✅ Human-readable reason per recommendation
   ✅ Expected recovery (ROI estimate per account)
   ✅ Portfolio summary with top 20 accounts

💡 Next Step:
   → pytest tests/test_risk_engine.py -v
================================================================================
""")
