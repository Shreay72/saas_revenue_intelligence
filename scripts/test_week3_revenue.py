"""
Week 3 Revenue Intelligence Training Script
Run:
    python scripts/test_week3_revenue.py
"""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.train_revenue import RevenueModelTrainer
from src.pipelines.revenue_pipeline import load_revenue_pipeline
from src.pipelines.churn_pipeline import ChurnPredictionPipeline
from src.utils.metrics import calculate_portfolio_metrics

print("=" * 80)
print("🚀 WEEK 3: REVENUE INTELLIGENCE TRAINING")
print("=" * 80)

# ─────────────────────────────────────────────
# STEP 1 — TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[1/3] Training Revenue Models...")

trainer = RevenueModelTrainer(output_dir="models/revenue")
mrr_metrics, clv_metrics = trainer.run(
    data_path="data/processed/account_level_features.csv"
)

print("\n✅ All revenue models trained successfully!")

# ─────────────────────────────────────────────
# STEP 2 — LOAD TEST DATA + CHURN PROBS
# ─────────────────────────────────────────────
print("\n[2/3] Loading test data for validation...")

df = pd.read_csv("data/processed/account_level_features.csv")

if "churn_probability" not in df.columns:
    churn_pipeline = ChurnPredictionPipeline(model_dir="models/churn")
    df["churn_probability"] = churn_pipeline.predict_proba(df)

# ─────────────────────────────────────────────
# STEP 3 — REVENUE AT RISK REPORT
# ─────────────────────────────────────────────
print("\n[3/3] Generating Revenue At Risk Report...")

pipeline = load_revenue_pipeline()
risk_report = trainer.calculate_revenue_at_risk(df)
total_at_risk = risk_report["revenue_at_risk"].sum()

print(f"\nTotal Monthly Revenue At Risk: ${total_at_risk:,.2f}")
print("  (Formula: MRR × churn_probability — expected value)\n")

print("📊 TOP 10 ACCOUNTS BY REVENUE AT RISK:")
top10 = risk_report.head(10)[
    ["account_id", "account_name", "total_mrr", "churn_probability", "revenue_at_risk"]
]
print(top10.to_string(index=False))

print("\n" + "=" * 80)
print("📈 PORTFOLIO REVENUE REPORT")
print("=" * 80)

print("\n" + "=" * 80)
print("📊 REVENUE INTELLIGENCE REPORT")
print("=" * 80)

metrics = calculate_portfolio_metrics(df)

print(
    f"""
👥 Account Overview:
   Total Accounts:   {metrics['total_accounts']}
   Active Accounts:  {metrics['active_accounts']}
   Churned Accounts: {metrics['churned_accounts']}
   Churn Rate:       {metrics['churn_rate'] * 100:.1f}%

💰 Revenue Overview:
   Total MRR:        ${metrics['total_mrr']:,.2f}
   Total ARR:        ${metrics['total_arr']:,.2f}
   Avg MRR/Account:  ${metrics['avg_mrr']:,.2f}
"""
)

if "total_revenue_at_risk" in metrics:
    print(f"⚠️  Monthly Revenue at Risk:  ${metrics['total_revenue_at_risk']:,.2f}")
print("=" * 80)

print(
    f"""
📉 MRR Model Performance:
   Best Model:       {mrr_metrics['model_name']}
   R² Score:         {mrr_metrics['r2_mean']:.4f} ± {mrr_metrics['r2_std']:.4f}
   MAE:              ${mrr_metrics['mae_mean']:,.2f} ± ${mrr_metrics['mae_std']:,.2f}
   Stability Check:  {mrr_metrics['stability_flag']}
   Cross-seed R²:    {mrr_metrics['stability_r2_mean']:.4f} ± {mrr_metrics['stability_r2_std']:.4f}
"""
)

print(
    f"""
💎 CLV Model:
   Method:           {clv_metrics['method']}
   Formula:          CLV = (MRR × gross_margin) / (monthly_churn + monthly_discount)
   Mean CLV:         ${clv_metrics['clv_mean']:,.2f}
   Median CLV:       ${clv_metrics['clv_median']:,.2f}
   Max CLV:          ${clv_metrics['clv_max']:,.2f}
   Overfitting Risk: None ✅ (deterministic formula)
"""
)

print("=" * 80)
print("✅ WEEK 3 TRAINING COMPLETE!")
print("=" * 80)

print(
    """
📦 Artifacts Created:
   • models/revenue/revenue_model_v1.pkl
   • models/revenue/clv_model_v1.pkl
   • models/revenue/clv_metadata.json
   • models/revenue/revenue_metadata.json

🔬 Changes Applied:
   ✅ CLV — Deterministic formula (no ML overfitting)
   ✅ No total_arr leakage in MRR model
   ✅ Churn probabilities from churn model (no hard-coded mapping)
   ✅ State/signal features for engagement/support/revenue
   ✅ Expected-loss revenue at risk (MRR × churn_probability)
   ✅ Stability check + confidence intervals

💡 Next Step:
   → pytest tests/test_revenue_model.py -v
"""
)
print("=" * 80)
