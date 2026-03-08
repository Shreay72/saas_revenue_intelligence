# =============================================================================
# SaaS Revenue Intelligence — Makefile
# Usage: make <target>
# =============================================================================

.PHONY: help install train api dashboard test monitor report retrain docker clean

# Default target
help:
	@echo ""
	@echo "  SaaS Revenue Intelligence — Available Commands"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make install     Install all dependencies"
	@echo "  make train       Train all models (fast, no tuning)"
	@echo "  make train-full  Train with Optuna hyperparameter tuning"
	@echo "  make api         Start FastAPI server (port 8000)"
	@echo "  make dashboard   Start Streamlit dashboard (port 8501)"
	@echo "  make test        Run all 182 tests"
	@echo "  make monitor     Run monitoring checks"
	@echo "  make report      Generate HTML + JSON portfolio report"
	@echo "  make retrain     Retrain models (if monitoring recommends)"
	@echo "  make docker      Build and start Docker containers"
	@echo "  make docker-down Stop Docker containers"
	@echo "  make verify      Verify all model artifacts exist"
	@echo "  make clean       Remove logs, monitoring outputs, reports"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed."

# ── Train ─────────────────────────────────────────────────────────────────────
train:
	python scripts/train_all.py --skip-week1 --no-tune

train-full:
	python scripts/train_all.py --skip-week1

train-from-scratch:
	python scripts/train_all.py --no-tune

verify:
	python scripts/train_all.py --verify-only

# ── Services ──────────────────────────────────────────────────────────────────
api:
	python scripts/run_api.py

dashboard:
	python scripts/run_streamlit.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -x -q

test-cov:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ── Monitoring & Reports ──────────────────────────────────────────────────────
monitor:
	python scripts/monitor_model.py

report:
	python scripts/generate_report.py --run-monitoring

report-json:
	python scripts/generate_report.py --format json

report-html:
	python scripts/generate_report.py --format html

retrain:
	python scripts/retrain_model.py --check-first --no-tune

retrain-force:
	python scripts/retrain_model.py --no-tune

# ── Docker ────────────────────────────────────────────────────────────────────
docker:
	docker-compose up --build

docker-bg:
	docker-compose up --build -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-monitoring:
	docker-compose --profile monitoring up cron

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/*.log
	rm -rf monitoring/drift_report.json monitoring/model_health_report.json
	rm -rf reports/*.html reports/*.json
	@echo "✅ Cleaned logs, monitoring outputs, and reports."

clean-all: clean
	rm -rf models/churn/* models/revenue/*
	rm -f data/processed/account_intelligence.csv
	@echo "✅ Full clean — models and intelligence removed."
