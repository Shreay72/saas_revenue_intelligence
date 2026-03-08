# 📊 Feature Importance & Engineering

## Account-Level Features (36 total)

### Base Features
```
account_id, account_name, industry, plan_tier, seats
```

### Subscription Features
```
total_mrr           — primary revenue signal
tenure_months       — customer longevity
auto_renew_ratio    — renewal intent
revenue_per_seat    — account efficiency
```

### Engagement Features
```
engagement_score          — composite usage health (0–100)
unique_features_used      — product adoption breadth
error_count / error_rate  — friction indicator
```

### Support Features
```
ticket_count              — support burden
avg_resolution_time       — support quality
avg_first_response_time   — responsiveness
avg_satisfaction_score    — customer sentiment
escalation_ratio          — severity indicator
total_refund_amount       — financial dissatisfaction
```

### Signal Features (Derived)
```
engagement_state          — -1 declining | 0 stable | +1 growing
support_pressure_signal   — 0 normal | 1 high pressure
revenue_change_signal     — directional MRR trend
```

### Target
```
churn_flag    — 1 = churned | 0 = active
```

---

## Churn Model — XGBoost Feature Importance

| Feature | Importance | Note |
|---|---|---|
| churn_probability | 1.0000 | ⚠️ DATA LEAKAGE — remove before prod |
| error_count | 0.0000 | |
| unique_features_used | 0.0000 | |
| ticket_count | 0.0000 | |

**Action required**: Remove `churn_probability` from training features.
After removal, expected top features: `engagement_score`, `ticket_count`,
`auto_renew_ratio`, `tenure_months`, `error_rate`.

---

## Revenue Model — GradientBoosting

```
R² = 0.9182 ± 0.0413   (stable across seeds 42, 7, 99)
MAE = $2,744 ± $342

Cross-seed R²: 0.9189 ± 0.0142  → STABLE ✅
```

Implicit top predictors: `total_mrr`, `tenure_months`,
`engagement_score`, `auto_renew_ratio`, `support_pressure_signal`

---

## Risk Engine Weights (Configurable)

```yaml
# config/model_config.yaml
risk_engine:
  weights:
    churn_probability: 0.40   # primary signal
    revenue_at_risk:   0.35   # financial impact
    health_score:      0.25   # product health
```

Adjust these weights based on your business priorities.
Higher `revenue_at_risk` weight → more financially-driven prioritization.

---

## Production Improvements

1. Remove `churn_probability` leakage from churn training data
2. Add SHAP explanations to `/explain/{account_id}` API endpoint
3. Track feature importance weekly via drift detector
4. A/B test risk engine weights quarterly
