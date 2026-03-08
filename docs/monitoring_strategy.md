# 📡 Monitoring Strategy

## Overview

```
Daily monitoring cycle:

1. DataDriftDetector → check 7 features for mean shift
2. ModelMonitor     → check AUC, calibration, portfolio health
3. AlertManager     → dispatch alerts (console + file + email)
4. retrain_model.py → auto-retrain if triggered
5. generate_report  → produce HTML + JSON report
```

---

## Data Drift Detection

### Method: Normalised Mean Shift
```
shift = |mean_current - mean_baseline| / (|mean_baseline| + ε)
```

### Monitored Features
```
total_mrr              — revenue changes
churn_probability      — model output shift
health_score           — account health trend
tenure_months          — cohort aging
ticket_count           — support load change
engagement_score       — product usage shift
revenue_at_risk        — portfolio risk level
```

### Thresholds (config/monitoring_config.yaml)
```
WARNING:  shift > 5%   → alert + flag for review
CRITICAL: shift > 10%  → alert + trigger retrain evaluation
```

---

## Model Health Monitoring

### 4 Checks Run Daily

| Check | Method | Threshold |
|---|---|---|
| Churn Rate Sanity | mean(churn_flag) | 5%–95% |
| Probability Calibration | \|mean_prob − actual_rate\| | WARNING>10%, CRITICAL>20% |
| AUC Estimate | Mann-Whitney U statistic | min_auc=0.75 |
| Portfolio Drift | metric deltas vs baseline | configurable |

### Retrain Triggers
```
RETRAIN_REQUIRED     ← 1+ CRITICAL check
RETRAIN_RECOMMENDED  ← 2+ WARNING checks
OK                   ← all checks pass
```

---

## Alert Channels

```yaml
# config/monitoring_config.yaml
alerts:
  channels:
    console: true    # always on
    file:    true    # monitoring/alert_log.json
    email:   false   # set smtp credentials to enable
```

### Alert Severity Levels
```
INFO     — routine, no action needed
WARNING  — review recommended
CRITICAL — immediate action required
```

---

## Baseline Management

```bash
# First run auto-creates baseline from current data
python scripts/generate_report.py --run-monitoring

# Manually reset baseline after intentional data changes
python -c "
from src.monitoring.data_drift_detector import DataDriftDetector
import pandas as pd
df = pd.read_csv('data/processed/account_intelligence.csv')
DataDriftDetector().create_baseline(df)
"
```

---

## Monitoring Schedule

```bash
# Run manually
python scripts/monitor_model.py

# Run as part of report generation
python scripts/generate_report.py --run-monitoring

# Docker cron (daily at 2am)
# See docker-compose.yml cron service
```

---

## Monitoring Outputs

```
monitoring/
├── baseline_snapshot.csv     ← drift baseline
├── drift_report.json         ← per-feature drift results
├── model_health_report.json  ← 4 health checks + recommendation
└── alert_log.json            ← full alert history

reports/
└── portfolio_report_TIMESTAMP.html   ← shareable HTML report
└── portfolio_report_TIMESTAMP.json   ← structured JSON report
```
