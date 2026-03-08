<p align="center">
  <h1 align="center">🚀 SaaS Revenue Intelligence</h1>
  <p align="center">
    <strong>AI-Powered Revenue Optimization & Churn Prevention Platform</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-api-reference">API</a> •
    <a href="#-documentation">Docs</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-Models-006600" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-182%20Passed-brightgreen" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Business Problem & ROI](#-business-problem--roi)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Model Training](#-model-training)
- [ML Models & Performance](#-ml-models--performance)
- [Risk Engine & Intelligence](#-risk-engine--intelligence)
- [API Reference](#-api-reference)
- [Dashboard](#-dashboard)
- [Monitoring & MLOps](#-monitoring--mlops)
- [Jupyter Notebooks](#-jupyter-notebooks)
- [Testing](#-testing)
- [Makefile Commands](#-makefile-commands)
- [Docker Deployment](#-docker-deployment)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**SaaS Revenue Intelligence** is an end-to-end machine learning platform that transforms how SaaS businesses manage subscription revenue. It combines **predictive analytics**, **composite risk scoring**, and **automated action planning** to move Customer Success teams from reactive firefighting to systematic, data-driven account prioritization.

### What It Does

| Capability | Description |
|---|---|
| **Churn Prediction** | ML-powered customer churn forecasting using LogisticRegression, XGBoost, and RandomForest |
| **Revenue Forecasting** | MRR prediction with GradientBoosting (R² = 0.92, MAE = $2,744) |
| **Customer Lifetime Value** | Deterministic CLV formula (Mean CLV = $570,043) |
| **Risk Scoring** | Composite risk engine with configurable weights (0-100 scale) |
| **Action Planning** | Automated intervention mapping with urgency and expected recovery |
| **Real-time API** | FastAPI REST endpoints for on-demand account scoring |
| **Interactive Dashboard** | Streamlit-based portfolio visualization with drill-down analysis |
| **MLOps Monitoring** | Automated data drift detection, model health tracking, retrain triggers |

### Portfolio Snapshot

```
Total Accounts:    500
Total MRR:         $11,338,747
Annual ARR:        $136,064,964
Revenue at Risk:   $7,968,352  (70% of MRR)
Recoverable:       $2,155,634  (27% via intervention)
```

---

## 💡 Business Problem & ROI

### The Problem

SaaS companies lose **15-25% of revenue annually** to preventable churn. Without systematic scoring, Customer Success Managers (CSMs) work reactively:

- ❌ No visibility into which accounts are at risk
- ❌ No financial impact quantification per account
- ❌ No prioritized action playbook
- ❌ Upsell opportunities buried in noise

**Result**: ~$1.7M/year lost to avoidable churn with no systematic defense.

### The Solution

| From | To |
|---|---|
| Reactive CSM firefighting | Systematic, data-driven prioritization |
| Gut-feel risk assessment | ML-powered scoring (0-100) with confidence |
| Manual account reviews | Automated daily scoring + action recommendations |
| No financial context | Per-account revenue-at-risk + recovery estimates |

### ROI

| Metric | Value |
|---|---|
| Revenue at Risk | $7,968,352 |
| Expected Recovery | $2,155,634 |
| Implementation Cost | ~2 developer weeks |
| Payback Period | < 1 month |
| Net ARR Protected | $2,155,634/year |
| **ROI** | **~10x in Year 1** |

### Risk Distribution

| Tier | Accounts | % of Total | MRR at Risk | Intervention | Recovery Rate |
|---|---|---|---|---|---|
| 🔴 CRITICAL | 73 | 15% | ~$2.8M | VP Escalation | 50% |
| 🟠 HIGH | 184 | 37% | ~$3.2M | TAM Assignment | 40% |
| 🟡 MEDIUM | 95 | 19% | ~$1.2M | Executive QBR | 30% |
| 🟢 LOW | 148 | 30% | ~$0.8M | Upsell Focus | 20% |

---

## ✨ Features

### Core ML Pipeline
- **Multi-model training** with Optuna hyperparameter optimization (TPE-based)
- **Probability calibration** (Sigmoid/Isotonic) for reliable risk assessment
- **Cost-sensitive evaluation** — optimizes classification thresholds based on business costs ($100/FP vs $5,000/FN)
- **SHAP explainability** — global feature importance and individual prediction explanations
- **31 engineered features** from 5 raw data sources, covering financial, engagement, support, and behavioral signals

### Intelligence Engine
- **Composite Risk Score** (0-100) combining churn probability (40%), revenue-at-risk (35%), and health score (25%)
- **Risk Tier Classification** — CRITICAL / HIGH / MEDIUM / LOW with configurable thresholds
- **Automated Action Planning** — maps risk profiles to VP Escalation, TAM Assignment, Executive QBR, Upsell, or Winback
- **Priority Scoring** — combines churn probability with account value (MRR) for resource allocation
- **Expected Recovery Calculation** — revenue-based ROI estimates per intervention

### Production Infrastructure
- **FastAPI REST API** with Swagger/ReDoc documentation, rate limiting, and caching
- **Streamlit Dashboard** with real-time portfolio view, risk charts, and account drill-down
- **Docker + docker-compose** for containerized deployment
- **Monitoring pipeline** with data drift detection, model health checks, and automated alerts
- **Auto-retrain triggers** when model performance degrades

---

## 🏗 Architecture

### System Overview

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

### Production Architecture

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

### Data Pipeline

| Raw File | Key Columns | Join Key |
|---|---|---|
| accounts.csv | account_id, industry, seats, plan_tier | account_id |
| subscriptions.csv | total_mrr, tenure_months, auto_renew_ratio | account_id |
| feature_usage.csv | engagement_score, unique_features_used, error_rate | account_id |
| support_tickets.csv | ticket_count, escalation_ratio, avg_resolution_time | account_id |
| churn_events.csv | churn_flag (target variable) | account_id |

**Output**: `data/processed/account_level_features.csv` (500 rows × 36 columns)

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11 | Core runtime |
| ML | scikit-learn, XGBoost | Model training & evaluation |
| Optimization | Optuna | Hyperparameter tuning (TPE) |
| API | FastAPI + Uvicorn | REST API server |
| Dashboard | Streamlit + Plotly | Interactive visualization |
| Config | PyYAML | Configuration management |
| Serialization | joblib | Model artifact persistence |
| Rate Limiting | slowapi | API request throttling |
| HTTP Client | httpx | Async API testing |
| Testing | pytest + pytest-cov | 182 tests, 100% pass rate |
| Containers | Docker + docker-compose | Containerized deployment |

---

## 📁 Project Structure

```
saas_revenue_intelligence/
│
├── config/                         # Configuration files
│   ├── config.yaml                 # App settings, paths, API config
│   ├── model_config.yaml           # Model hyperparams, risk weights, thresholds
│   ├── logging_config.yaml         # Logging levels and handlers
│   └── monitoring_config.yaml      # Drift thresholds, alert channels
│
├── data/
│   ├── raw/                        # Original CSV data files (5 sources)
│   └── processed/                  # Engineered features & intelligence output
│       ├── account_level_features.csv   # 500 × 36 feature matrix
│       └── account_intelligence.csv     # 500 × 20 scored accounts
│
├── src/                            # Core business logic
│   ├── data/                       # Data pipeline modules
│   │   ├── data_loader.py          # CSV loading with validation & memory tracking
│   │   ├── data_cleaning.py        # Schema standardization, null handling
│   │   ├── data_validator.py       # Constraint enforcement & integrity checks
│   │   └── feature_engineering.py  # 31 account-level features from 5 sources
│   │
│   ├── models/                     # ML model training
│   │   ├── train_churn.py          # Churn model (LR, RF, XGB) with Optuna
│   │   ├── train_revenue.py        # Revenue model (GBR, RF, Ridge) with Optuna
│   │   ├── evaluate.py             # ROC-AUC, Brier, business cost evaluation
│   │   ├── model_explainability.py # SHAP global + local explanations
│   │   ├── clv_model.py            # Customer Lifetime Value calculator
│   │   ├── model_registry.py       # Model artifact tracking & loading
│   │   └── model_versioning.py     # Semantic version management
│   │
│   ├── pipelines/                  # End-to-end orchestration
│   │   ├── preprocessing_pipeline.py   # ColumnTransformer (scaling + encoding)
│   │   ├── churn_pipeline.py       # Preprocessing → Churn → Risk → Actions
│   │   └── revenue_pipeline.py     # Revenue forecasting pipeline
│   │
│   ├── inference/                  # Prediction serving
│   │   ├── schema.py               # Pydantic request/response models
│   │   ├── predict.py              # Real-time single-account predictions
│   │   └── batch_predict.py        # Large-scale offline predictions
│   │
│   ├── monitoring/                 # MLOps monitoring
│   │   ├── data_drift_detector.py  # Normalized mean shift detection
│   │   ├── model_monitor.py        # AUC, calibration, portfolio health
│   │   └── alert_manager.py        # Console, file, email alert dispatch
│   │
│   ├── risk/                       # Business logic
│   │   └── risk_engine.py          # Composite risk scoring (0-100)
│   │
│   ├── recommendation/             # Action planning
│   │   └── business_rules.py       # Intervention mapping & playbooks
│   │
│   ├── api/                        # FastAPI application
│   │   ├── main.py                 # API server entry point
│   │   └── routes.py               # Endpoint definitions
│   │
│   └── utils/                      # Shared utilities
│       ├── logger.py               # Colored console + file logging
│       ├── config_loader.py        # YAML + env variable parser
│       ├── helpers.py              # IO utilities, JSON, timers
│       └── metrics.py              # MRR, ARR, CLV, CAC, NRR, risk formulas
│
├── models/                         # Trained model artifacts
│   ├── churn/                      # churn_model_v1.pkl + metadata
│   └── revenue/                    # revenue_model_v1.pkl, clv_model_v1.pkl
│
├── dashboard/                      # Streamlit UI application
│   ├── main.py                     # Dashboard entry point
│   ├── api_client.py               # API communication layer
│   ├── pages/                      # Multi-page views
│   │   ├── overview.py             # Portfolio overview + KPIs
│   │   ├── risk_analysis.py        # Risk distribution charts
│   │   ├── account_detail.py       # Individual account drill-down
│   │   └── monitoring.py           # Model health dashboard
│   └── utils/                      # Dashboard utilities
│
├── notebooks/                      # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb   # EDA on raw datasets
│   ├── 02_feature_engineering.ipynb# Feature creation walkthrough
│   ├── 04_revenue_modeling.ipynb   # Revenue model training
│   ├── 05_model_evaluation.ipynb   # Comprehensive model eval
│   ├── 06_shap_interpretation.ipynb# SHAP feature explanations
│   └── 07_model_drift_analysis.ipynb # Drift detection analysis
│
├── scripts/                        # Automation & integration tests
│   ├── train_all.py                # Full pipeline orchestrator
│   ├── run_api.py                  # Start FastAPI server
│   ├── run_streamlit.py            # Start Streamlit dashboard
│   ├── generate_report.py          # HTML + JSON report generation
│   ├── monitor_model.py            # Run monitoring checks
│   ├── retrain_model.py            # Conditional model retraining
│   ├── test_week1_pipeline.py      # Phase 1 integration test (data)
│   ├── test_week2_models.py        # Phase 2 integration test (churn)
│   ├── test_week3_revenue.py       # Phase 3 integration test (revenue)
│   ├── test_week4_risk.py          # Phase 4 integration test (risk)
│   └── test_inference.py           # Phase 5 integration test (inference)
│
├── tests/                          # pytest test suites
│   ├── test_churn_model.py         # Churn model & evaluator tests
│   ├── test_revenue_model.py       # Revenue model tests
│   ├── test_risk_engine.py         # Risk engine tests
│   ├── test_api.py                 # API endpoint tests
│   └── ...
│
├── docs/                           # Documentation
│   ├── architecture.md             # System architecture overview
│   ├── api_documentation.md        # API endpoint reference
│   ├── business_problem.md         # Business problem & ROI analysis
│   ├── model_comparison.md         # Model performance comparison
│   ├── feature_importance.md       # Feature engineering details
│   ├── monitoring_strategy.md      # Monitoring pipeline docs
│   ├── deployment_guide.md         # Setup & deployment instructions
│   ├── troubleshooting.md          # Common errors & fixes
│   ├── generate_pdf.py             # PDF documentation generator
│   └── SaaS_Revenue_Intelligence_Documentation.pdf  # 25-page PDF doc
│
├── monitoring/                     # Monitoring outputs
│   ├── baseline_snapshot.csv       # Drift detection baseline
│   ├── drift_report.json           # Per-feature drift results
│   ├── model_health_report.json    # Health check results
│   └── alert_log.json              # Alert history
│
├── reports/                        # Generated reports
│   └── portfolio_report_*.html     # Shareable HTML reports
│
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Multi-service orchestration
├── Makefile                        # CLI command shortcuts
├── requirements.txt                # Python dependencies
├── setup.py                        # Package configuration
└── .gitignore                      # Git exclusion rules
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Shreay72/saas_revenue_intelligence.git
cd saas_revenue_intelligence

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# 4. Train all models (fast mode, ~10 seconds)
python scripts/train_all.py --skip-week1 --no-tune

# 5. Start FastAPI server (Terminal 1)
python scripts/run_api.py
# API available at: http://localhost:8000/docs

# 6. Start Streamlit dashboard (Terminal 2)
python scripts/run_streamlit.py
# Dashboard available at: http://localhost:8501
```

### Verify Everything Works

```bash
# Run all 182 tests
pytest tests/ -v

# Verify all model artifacts exist
python scripts/train_all.py --verify-only

# Check API health
curl http://localhost:8000/health
```

---

## 🧠 Model Training

### Training Modes

```bash
# Fast training (no hyperparameter tuning, ~10 seconds)
python scripts/train_all.py --skip-week1 --no-tune

# Full training with Optuna optimization
python scripts/train_all.py --skip-week1

# Complete pipeline from raw data
python scripts/train_all.py --no-tune

# Verify all artifacts exist
python scripts/train_all.py --verify-only
```

### Training Pipeline Stages

| Stage | Script | Output | Time |
|---|---|---|---|
| Week 1 | Feature Engineering | `account_level_features.csv` (500 × 36) | ~5s |
| Week 2 | Churn Model Training | `churn_model_v1.pkl` | ~3s |
| Week 3 | Revenue + CLV Models | `revenue_model_v1.pkl`, `clv_model_v1.pkl` | ~2s |
| Week 4 | Intelligence Engine | `account_intelligence.csv` (500 × 20) | ~1s |

### ChurnModelTrainer Architecture

```python
class ChurnModelTrainer:
    """Elite churn model trainer with Optuna optimization."""

    def __init__(self, tune_hyperparameters=True,
                 n_trials=50, cost_fp=100.0, cost_fn=5000.0):
        # cost_fp: $100 per false alarm (retention offer cost)
        # cost_fn: $5,000 per missed churner (lost revenue)

    def train_logistic_regression(self, cv_folds=5):
        """Optuna-tuned LR with isotonic calibration"""

    def train_random_forest(self, cv_folds=5):
        """RandomForest with Optuna tuning"""

    def train_xgboost(self, cv_folds=5):
        """XGBoost with early stopping + Optuna"""

    def compare_models(self):
        """Compare all models on ROC-AUC, F1, Brier, Business Cost"""

    def save_best_model(self, output_dir="models/churn"):
        """Persist best model + metadata + feature importance"""
```

---

## 📊 ML Models & Performance

### Churn Model (Classification)

| Model | ROC-AUC | F1 | Precision | Recall | Brier Score | Business Cost |
|---|---|---|---|---|---|---|
| **LogisticRegression** ✅ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0005 | $0 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0001 | $0 |
| RandomForest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0331 | $0 |

**Winner**: LogisticRegression (best probability calibration, Brier = 0.0005)

> ⚠️ **Note**: Perfect scores are due to `churn_probability` data leakage in training features. After removing this feature, expected realistic AUC: 0.75-0.85.

### Revenue Model (Regression)

| Model | R² Mean | R² Std | MAE Mean | MAE Std |
|---|---|---|---|---|
| **GradientBoosting** ✅ | 0.9182 | ±0.0413 | $2,744 | ±$342 |
| RandomForest | 0.8193 | ±0.0698 | $4,260 | ±$477 |
| Ridge | 0.6146 | ±0.0678 | $7,560 | ±$707 |

**Winner**: GradientBoosting (highest R², lowest MAE, stable across seeds)

**Stability Check** (3 random seeds):
```
Seed 42: R² = 0.9186 ± 0.0345
Seed  7: R² = 0.9016 ± 0.0598
Seed 99: R² = 0.9364 ± 0.0249
Cross-seed: 0.9189 ± 0.0142  → ✅ STABLE
```

### CLV Model (Deterministic)

```
CLV = (MRR × gross_margin) / (monthly_churn + monthly_discount)

Parameters:
  gross_margin:   0.70 (70%)
  discount_rate:  0.10 (10% annual → 0.83% monthly)

Portfolio Results:
  CLV Mean:   $570,043
  CLV Median: $215,250
```

### Model Selection Rationale

| Decision | Chosen | Reason |
|---|---|---|
| Churn winner | LogisticRegression | Best calibration (Brier = 0.0005) |
| Revenue winner | GradientBoosting | R² = 0.92, stable across seeds |
| CLV | Deterministic formula | Interpretable, no overfitting risk |
| Risk scoring | Weighted formula | Transparent, configurable weights |

---

## 🎯 Risk Engine & Intelligence

### Composite Risk Score (0-100)

```
Risk Score = (w1 × churn_probability
           + w2 × clip(revenue_at_risk / p95, 0, 1)
           + w3 × (1 - health_score / 100)) × 100
```

### Configuration

```yaml
# config/model_config.yaml
risk_engine:
  weights:
    churn_probability: 0.40   # Primary signal
    revenue_at_risk:   0.35   # Financial impact
    health_score:      0.25   # Product health

  tier_thresholds:
    critical: 75              # Score >= 75 → CRITICAL
    high:     60              # Score >= 60 → HIGH
    medium:   35              # Score >= 35 → MEDIUM
                              # Score <  35 → LOW
```

### Health Score Formula

```python
health_score = (
    0.30 × engagement_norm +    # Product engagement (0-100)
    0.25 × support_health +     # 1 - support_risk_score
    0.25 × churn_health +       # 1 - churn_probability
    0.10 × tenure_norm +        # min(tenure_months / 36, 1)
    0.10 × auto_renew_norm      # Auto-renewal ratio
) × 100
```

### Intervention Mapping

| Risk Tier | Action | Owner | Success Rate | Confidence |
|---|---|---|---|---|
| CRITICAL | VP Escalation | VP of CS | 50% | HIGH |
| HIGH | TAM Assignment | Technical AM | 40% | HIGH |
| MEDIUM | Executive QBR | Account Executive | 30% | HIGH |
| LOW (upgrade potential) | Upsell | Sales | 20% | MEDIUM |
| LOW (recently churned) | Winback | Marketing | 25% | MEDIUM |
| LOW (healthy) | Monitor | CSM | 5% | LOW |

---

## 🔌 API Reference

**Base URL**: `http://localhost:8000`
| **Swagger UI**: http://localhost:8000/docs | **ReDoc**: http://localhost:8000/redoc |

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API health check with model status |
| `/status` | GET | API version, model info, account count |
| `/portfolio` | GET | Portfolio summary with tier distribution |
| `/portfolio/critical` | GET | Critical tier accounts only |
| `/accounts` | GET | Paginated account list with filters |
| `/accounts/{id}` | GET | Detailed account intelligence |
| `/accounts/{id}/risk` | GET | Risk breakdown for account |
| `/accounts/{id}/recommendation` | GET | Action recommendation |
| `/score` | POST | Real-time scoring of new data |
| `/cache/status` | GET | Cache freshness check |
| `/cache/refresh` | POST | Force data reload |

### Example: Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "accounts_loaded": 500,
  "pipeline_ready": true,
  "model_version": "1.0.0",
  "cache_status": "fresh"
}
```

### Example: Account Intelligence

```bash
curl http://localhost:8000/accounts/ACC_001
```

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

### Example: Real-time Scoring

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "total_mrr": 15000,
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

### Query Parameters for `/accounts`

| Parameter | Type | Default | Options |
|---|---|---|---|
| page | int | 1 | any |
| limit | int | 20 | 1-100 |
| tier | string | all | CRITICAL, HIGH, MEDIUM, LOW |
| urgency | string | all | IMMEDIATE, HIGH, MEDIUM, LOW |

---

## 📈 Dashboard

The Streamlit dashboard provides real-time portfolio visibility:

- **Portfolio Overview** — KPI cards (MRR, ARR, accounts, risk), tier distribution
- **Risk Analysis** — Risk distribution charts, revenue-at-risk by industry
- **Account Detail** — Individual account drill-down with risk breakdown & recommendations
- **Monitoring** — Model health status, drift detection results, alert history

```bash
# Start dashboard
python scripts/run_streamlit.py
# Access at: http://localhost:8501
```

---

## 📡 Monitoring & MLOps

### Daily Monitoring Cycle

```
1. DataDriftDetector → check 7 features for mean shift
2. ModelMonitor      → check AUC, calibration, portfolio health
3. AlertManager      → dispatch alerts (console + file + email)
4. retrain_model.py  → auto-retrain if triggered
5. generate_report   → produce HTML + JSON report
```

### Data Drift Detection

**Method**: Normalized Mean Shift

```
shift = |mean_current - mean_baseline| / (|mean_baseline| + epsilon)
```

**Monitored Features**: `total_mrr`, `churn_probability`, `health_score`, `tenure_months`, `ticket_count`, `engagement_score`, `revenue_at_risk`

**Thresholds**:
- ⚠️ WARNING: shift > 5% → flag for review
- 🔴 CRITICAL: shift > 10% → trigger retrain evaluation

### Model Health Checks

| Check | Method | Threshold |
|---|---|---|
| Churn Rate Sanity | mean(churn_flag) | 5% - 95% |
| Probability Calibration | \|mean_prob - actual_rate\| | WARNING > 10%, CRITICAL > 20% |
| AUC Estimate | Mann-Whitney U statistic | min_auc = 0.75 |
| Portfolio Drift | Metric deltas vs baseline | Configurable |

### Retrain Triggers

```
RETRAIN_REQUIRED     ← 1+ CRITICAL check
RETRAIN_RECOMMENDED  ← 2+ WARNING checks
OK                   ← All checks pass
```

### Alert Channels

```yaml
alerts:
  channels:
    console: true    # Always on
    file:    true    # monitoring/alert_log.json
    email:   false   # Set SMTP credentials to enable
```

### Monitoring Commands

```bash
# Run monitoring checks
python scripts/monitor_model.py

# Generate report with monitoring
python scripts/generate_report.py --run-monitoring

# Check if retrain needed
python scripts/retrain_model.py --check-first --no-tune
```

### Monitoring Outputs

```
monitoring/
├── baseline_snapshot.csv       # Drift baseline
├── drift_report.json           # Per-feature drift results
├── model_health_report.json    # 4 health checks + recommendation
└── alert_log.json              # Full alert history

reports/
├── portfolio_report_*.html     # Shareable HTML report
└── portfolio_report_*.json     # Structured JSON report
```

---

## 📓 Jupyter Notebooks

| Notebook | Purpose | Key Outputs |
|---|---|---|
| `01_data_exploration.ipynb` | EDA on raw datasets | Distribution plots, correlations, data quality assessment |
| `02_feature_engineering.ipynb` | Feature creation walkthrough | 31 features from 5 sources, transformation logic |
| `04_revenue_modeling.ipynb` | Revenue model training | GradientBoosting R²=0.92, model comparison |
| `05_model_evaluation.ipynb` | Comprehensive model eval | ROC curves, confusion matrices, business cost analysis |
| `06_shap_interpretation.ipynb` | SHAP explanations | Global feature importance, individual prediction reasoning |
| `07_model_drift_analysis.ipynb` | Drift detection analysis | Feature stability reports, drift visualization |

---

## 🧪 Testing

### Test Strategy

| Phase | Script | Type | Purpose |
|---|---|---|---|
| Phase 1 | `scripts/test_week1_pipeline.py` | Integration | E2E data flow validation |
| Phase 2 | `scripts/test_week2_models.py` | Integration | Training orchestration |
| Phase 3 | `scripts/test_week3_revenue.py` | Integration | Revenue model training |
| Phase 4 | `scripts/test_week4_risk.py` | Integration | Risk engine & intelligence |
| Phase 5 | `scripts/test_inference.py` | Integration | Pipeline prediction logic |
| Unit | `tests/test_churn_model.py` | Unit | Model evaluator & pipeline |
| Unit | `tests/test_api.py` | Unit | REST endpoint validation |

### Running Tests

```bash
# Run all 182 tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Quick test (stop on first failure)
pytest tests/ -x -q

# Run specific integration test
python scripts/test_week2_models.py
```

---

## 🔧 Makefile Commands

| Command | Description |
|---|---|
| `make install` | Install all Python dependencies |
| `make train` | Train models (fast, no tuning) |
| `make train-full` | Train with Optuna hyperparameter tuning |
| `make api` | Start FastAPI server (port 8000) |
| `make dashboard` | Start Streamlit dashboard (port 8501) |
| `make test` | Run all 182 tests |
| `make test-cov` | Run tests with coverage report |
| `make monitor` | Run monitoring health checks |
| `make report` | Generate HTML + JSON portfolio report |
| `make retrain` | Retrain if monitoring recommends it |
| `make docker` | Build and start Docker containers |
| `make docker-down` | Stop Docker containers |
| `make verify` | Verify all model artifacts exist |
| `make clean` | Remove logs, reports, monitoring outputs |
| `make clean-all` | Full clean including models & intelligence |

---

## 🐳 Docker Deployment

### Quick Start

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Stop services
docker-compose down
```

### Production Deployment

```bash
# Build production image
docker-compose -f docker-compose.prod.yml build

# Push to registry
docker tag saas-intelligence:latest your-registry/saas-intelligence:latest
docker push your-registry/saas-intelligence:latest

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

### Scaling Guide

| Scale | Accounts | Infrastructure |
|---|---|---|
| Current | 500 | 2 CPU, 4GB RAM (single machine) |
| Phase 1 | 5,000 | + Redis cache + PostgreSQL |
| Phase 2 | 50,000 | + Kubernetes + S3 + read replicas |
| Phase 3 | 500,000 | + Apache Spark + distributed training |

---

## ⚙️ Configuration

### Application Config (`config/config.yaml`)

```yaml
app:
  name: "SaaS Revenue Intelligence"
  version: "1.0.0"
  environment: "development"

api:
  host: "0.0.0.0"
  port: 8000
  cache_ttl_seconds: 300   # 5-minute cache

dashboard:
  port: 8501
  api_url: "http://localhost:8000"

retrain:
  trigger_auc_drop: 0.05   # Retrain if AUC drops 5%+
  trigger_drift_score: 0.10 # Retrain if drift > 10%
```

### Model Config (`config/model_config.yaml`)

```yaml
churn_model:
  test_size: 0.2
  random_state: 42
  models: [logistic_regression, random_forest, xgboost]
  cv_folds: 5
  scoring: roc_auc

revenue_model:
  test_size: 0.2
  cv_folds: 5
  clv:
    gross_margin: 0.70
    discount_rate: 0.10

risk_engine:
  weights:
    churn_probability: 0.40
    revenue_at_risk: 0.35
    health_score: 0.25
  tier_thresholds:
    critical: 75
    high: 60
    medium: 35
```

### Environment Variables

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
CACHE_TTL=300

# Monitoring alerts
SMTP_USERNAME=alerts@yourcompany.com
SMTP_PASSWORD=your_password
ALERT_EMAIL=team@yourcompany.com
```

---

## 🔍 Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `FileNotFoundError: churn_model_v1.pkl` | Models not trained | `python scripts/train_all.py --no-tune` |
| `KeyError: 'revenue_at_risk'` | Pipeline order issue | Run revenue pipeline before intelligence |
| `ROC-AUC = 1.0000` | Data leakage | Remove `churn_probability` from features |
| `API not reachable` in dashboard | API not started | Start API first: `python scripts/run_api.py` |
| `No module named 'X'` | Missing dependencies | `pip install -r requirements.txt` |
| `Baseline not found` | First monitoring run | Expected — auto-creates on first run |
| `Port already in use` | Stale process | `netstat -ano \| findstr :8000` |
| `use_container_width` warning | Streamlit version | Replace with `width='stretch'` |
| CLV values > $500K | Low churn + high MRR | Expected by formula — not a bug |

### Debug Commands

```bash
# Verify all model artifacts exist
python scripts/train_all.py --verify-only

# Check API health
curl http://localhost:8000/health

# Check account count
python -c "import pandas as pd; print(pd.read_csv('data/processed/account_intelligence.csv').shape)"

# Check risk distribution
python -c "import pandas as pd; print(pd.read_csv('data/processed/account_intelligence.csv')['risk_tier'].value_counts())"

# Run full pipeline
python scripts/train_all.py --skip-week1 --no-tune
```

---

## 🔮 Future Enhancements

### Predictive Analytics
- Time-series churn prediction (30/60/90-day forecasts)
- Cohort analysis by industry, plan tier, and engagement
- Upsell propensity modeling
- Revenue trajectory forecasting
- Seasonal pattern detection

### AI-Driven Insights
- NLP-powered account risk narratives
- Automated playbook generation per risk profile
- Anomaly detection for unusual account behavior
- Churn root cause analysis (top 3 drivers per account)
- Sentiment analysis on support ticket text

### Platform Improvements
- Real-time streaming with Apache Kafka
- A/B testing framework for interventions
- Multi-tenant support
- Mobile-responsive dashboard
- Webhook integrations (Slack, PagerDuty, Salesforce, HubSpot)
- Data warehouse integration (Snowflake, BigQuery, Redshift)
- Feature store with versioning and lineage tracking

### Immediate Action Items
1. Remove `churn_probability` leakage from training features
2. Add SHAP explanations to `/explain/{account_id}` API endpoint
3. Track feature importance weekly via drift detector
4. A/B test risk engine weights quarterly
5. Implement OAuth2 authentication for API endpoints

---

## 📖 Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|---|---|
| [architecture.md](docs/architecture.md) | System architecture & data flow |
| [api_documentation.md](docs/api_documentation.md) | Complete API reference |
| [business_problem.md](docs/business_problem.md) | Business problem & ROI analysis |
| [model_comparison.md](docs/model_comparison.md) | ML model performance comparison |
| [feature_importance.md](docs/feature_importance.md) | Feature engineering & importance |
| [monitoring_strategy.md](docs/monitoring_strategy.md) | Monitoring pipeline documentation |
| [deployment_guide.md](docs/deployment_guide.md) | Setup & deployment instructions |
| [troubleshooting.md](docs/troubleshooting.md) | Common errors & fixes |

📄 **Full PDF Documentation**: A professional 25-page PDF is available at [`docs/SaaS_Revenue_Intelligence_Documentation.pdf`](docs/SaaS_Revenue_Intelligence_Documentation.pdf)

Regenerate the PDF anytime:
```bash
python docs/generate_pdf.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Standards
- Python 3.11+ with type hints on all public functions
- Google-style docstrings on all classes and public methods
- Configuration-driven — no magic numbers in code
- Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`)

---

## 📄 License

This project is for educational and demonstration purposes.

---

<p align="center">
  <strong>Built with ❤️ for SaaS Revenue Optimization</strong>
  <br>
  <sub>500 Accounts • $11.3M MRR • $136M ARR • 3 ML Models • 182 Tests • 100% Pass Rate</sub>
</p>
