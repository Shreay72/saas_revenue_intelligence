# 🚀 Deployment Guide

## Development Setup

```bash
# 1. Clone & create venv
git clone <repo>
cd saas_revenue_intelligence
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Train all models (9.6s)
python scripts/train_all.py --skip-week1 --no-tune

# 4. Start services
# Terminal 1
python scripts/run_api.py

# Terminal 2
python scripts/run_streamlit.py

# 5. Access
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## Docker Setup (Recommended)

```bash
# Build and start everything
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Stop
docker-compose down
```

---

## Production Deployment

```bash
# 1. Build production image
docker-compose -f docker-compose.prod.yml build

# 2. Push to registry
docker tag saas-intelligence:latest your-registry/saas-intelligence:latest
docker push your-registry/saas-intelligence:latest

# 3. Deploy
docker-compose -f docker-compose.prod.yml up -d
```

---

## Monitoring & Maintenance

```bash
# Run monitoring + generate report
python scripts/generate_report.py --run-monitoring

# Check if retrain needed
python scripts/retrain_model.py --check-first --no-tune

# Run monitoring only
python scripts/monitor_model.py

# Run full tests
pytest tests/ -v

# Verify all artifacts exist
python scripts/train_all.py --verify-only
```

---

## Scaling Guide

| Accounts | Infrastructure |
|---|---|
| 500 | Current setup (2 CPU, 4GB RAM) |
| 5,000 | + Redis cache + PostgreSQL |
| 50,000 | + Kubernetes + S3 + read replicas |
| 500,000 | + Spark + distributed training |

---

## Environment Variables

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
