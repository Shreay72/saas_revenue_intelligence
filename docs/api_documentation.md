# 🔌 API Documentation

**Base URL**: `http://localhost:8000`  
**Swagger UI**: http://localhost:8000/docs  
**ReDoc**: http://localhost:8000/redoc

---

## Health Endpoints

### GET `/health`
```json
{
  "status": "healthy",
  "accounts_loaded": 500,
  "pipeline_ready": true,
  "model_version": "1.0.0",
  "cache_status": "fresh"
}
```

### GET `/status`
```json
{
  "api_version": "1.0.0",
  "churn_model": "LogisticRegression",
  "revenue_model": "GradientBoosting",
  "accounts_count": 500
}
```

---

## Portfolio

### GET `/portfolio`
Returns portfolio summary with tier distribution and top 20 accounts.
```json
{
  "total_accounts": 500,
  "tier_distribution": {"CRITICAL": 73, "HIGH": 184, "MEDIUM": 95, "LOW": 148},
  "total_revenue_at_risk": 7968352.23,
  "total_recoverable": 2155634.15,
  "top_20_accounts": [...]
}
```

### GET `/portfolio/critical`
Returns only CRITICAL tier accounts.

---

## Account Intelligence

### GET `/accounts?page=1&limit=20&tier=HIGH&urgency=IMMEDIATE`

| Parameter | Type | Default | Options |
|---|---|---|---|
| page | int | 1 | any |
| limit | int | 20 | 1–100 |
| tier | str | all | CRITICAL, HIGH, MEDIUM, LOW |
| urgency | str | all | IMMEDIATE, HIGH, MEDIUM, LOW |

### GET `/accounts/{account_id}`
```json
{
  "account_id": "ACC_001",
  "risk_tier": "CRITICAL",
  "risk_score": 91.2,
  "churn_probability": 0.94,
  "revenue_at_risk": 125000.00,
  "recommended_action": "VP_ESCALATION",
  "urgency": "IMMEDIATE",
  "expected_recovery": 62500.00,
  "reason": "churn_probability=0.94, CLV=$1,250,000 — strategic account at critical risk"
}
```

### GET `/accounts/{account_id}/risk`
### GET `/accounts/{account_id}/recommendation`

---

## Realtime Scoring

### POST `/score`
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "total_mrr": 15000,
    "churn_flag": 0,
    "engagement_score": 25,
    "ticket_count": 45,
    "tenure_months": 12,
    "auto_renew_ratio": 0.3
  }'
```
```json
{
  "risk_score": 87.3,
  "risk_tier": "CRITICAL",
  "churn_probability": 0.91,
  "health_score": 32.1,
  "revenue_at_risk": 13650.00,
  "recommended_action": "VP_ESCALATION",
  "scored_at": "2026-03-07T19:40:00"
}
```

---

## Cache

### GET `/cache/status`
### POST `/cache/refresh`
