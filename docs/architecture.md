# 🏗️ SaaS Revenue Intelligence — Architecture

## System Overview

```
Raw Data (5 CSVs)
       ↓
Week 1 — Feature Engineering  →  account_level_features.csv (500 × 36)
       ↓
Week 2 — Churn Model           →  churn_model_v1.pkl
       ↓
Week 3 — Revenue + CLV Models  →  revenue_model_v1.pkl, clv_model_v1.pkl
       ↓
Week 4 — Intelligence Engine   →  account_intelligence.csv (500 × 20)
       ↓
Week 5 — API + Dashboard + Monitoring + Reports
```

**Portfolio: 500 accounts | $11.3M MRR | $8M revenue at risk | $2.2M recoverable**

---

## Data Pipeline (Week 1)

| Raw File | Key Columns | Joined On |
|---|---|---|
| accounts.csv | account_id, industry, seats, plan_tier | account_id |
| subscriptions.csv | total_mrr, tenure_months, auto_renew_ratio | account_id |
| feature_usage.csv | engagement_score, unique_features_used, error_rate | account_id |
| support_tickets.csv | ticket_count, escalation_ratio, avg_resolution_time | account_id |
| churn_events.csv | churn_flag (target) | account_id |

Output: `data/processed/account_level_features.csv`

---

## ML Pipeline (Weeks 2–4)

### Week 2 — Churn Model
- Models: LogisticRegression, RandomForest, XGBoost
- Winner: LogisticRegression (Brier=0.0005)
- Artifacts: `models/churn/churn_model_v1.pkl`

### Week 3 — Revenue Model
- MRR: GradientBoosting (R²=0.9182, MAE=$2,744)
- CLV = (MRR × gross_margin) / (monthly_churn + monthly_discount)
- Artifacts: `models/revenue/revenue_model_v1.pkl`, `clv_model_v1.pkl`

### Week 4 — Intelligence Engine
- Risk Score = 40%×churn_prob + 35%×rev_at_risk_norm + 25%×(1−health/100)
- Tiers: CRITICAL(73) | HIGH(184) | MEDIUM(95) | LOW(148)
- Output: `data/processed/account_intelligence.csv`

---

## Production Architecture (Week 5)

```
┌──────────────────┐     ┌───────────────────┐
│  FastAPI :8000   │────▶│  Streamlit :8501  │
│  /health         │     │  Portfolio view   │
│  /accounts       │     │  Risk charts      │
│  /portfolio      │     │  Top 20 accounts  │
│  /score          │     │  Action drilldown │
└────────┬─────────┘     └───────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  account_intelligence.csv (500 × 20)    │
│  Cached in memory (TTL = 300s)          │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  Monitoring                              │
│  DataDriftDetector → drift_report.json  │
│  ModelMonitor      → model_report.json  │
│  AlertManager      → alert_log.json     │
│  Schedule: daily via cron / Docker      │
└──────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Config | PyYAML |
| Serialization | joblib |
| Testing | pytest (182 tests, 100% pass) |
| Containers | Docker + docker-compose |
