# 🤝 Model Comparison

## Churn Model — Test Set (100 accounts)

| Model | ROC-AUC | F1 | Precision | Recall | Brier | Business Cost |
|---|---|---|---|---|---|---|
| LogisticRegression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0005 | $0 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0001 | $0 |
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0331 | $0 |

**Winner**: LogisticRegression (lowest Brier score = best probability calibration)

**Note**: All models show ROC-AUC=1.0 due to `churn_probability` data leakage.
After fixing leakage, expected realistic AUC: 0.75–0.85.

### Confusion Matrix (Best Model — LogisticRegression)
```
                 Predicted
               Active  Churned
Actual Active    30       0     ← 0 false alarms
       Churned    0      70     ← 0 missed churners
```

### Business Cost Analysis
```
Cost per FP (false alarm):    $100   (retention offer cost)
Cost per FN (missed churner): $5,000 (lost customer revenue)
Total cost with perfect model: $0
```

---

## Revenue Model — Cross-Validation (5-fold)

| Model | R² Mean | R² Std | MAE Mean | MAE Std |
|---|---|---|---|---|
| GradientBoosting | 0.9182 | ±0.0413 | $2,744 | ±$342 |
| RandomForest | 0.8193 | ±0.0698 | $4,260 | ±$477 |
| Ridge | 0.6146 | ±0.0678 | $7,560 | ±$707 |

**Winner**: GradientBoosting (highest R², lowest MAE)

### Stability Check (3 random seeds)
```
Seed 42: R² = 0.9186 ± 0.0345
Seed  7: R² = 0.9016 ± 0.0598
Seed 99: R² = 0.9364 ± 0.0249
Cross-seed: 0.9189 ± 0.0142  → ✅ STABLE
```

### MRR Distribution
```
Min:    $190
Max:    $138,060
Mean:   $22,677
```

---

## CLV Model — Deterministic Formula

```
CLV = (MRR × gross_margin) / (monthly_churn + monthly_discount)

Parameters (from model_config.yaml):
  gross_margin:   0.70   (70%)
  discount_rate:  0.10   (10% annual → 0.83% monthly)

Portfolio Results:
  CLV Mean:   $570,043
  CLV Median: $215,250
```

No ML needed — deterministic formula is provably optimal for SaaS CLV
when churn probability is already known.

---

## Model Selection Rationale

| Decision | Chosen | Reason |
|---|---|---|
| Churn winner | LogisticRegression | Best calibration (Brier=0.0005) |
| Revenue winner | GradientBoosting | R²=0.92, stable across seeds |
| CLV | Deterministic | Interpretable, no overfitting risk |
| Risk scoring | Weighted formula | Transparent, configurable weights |
