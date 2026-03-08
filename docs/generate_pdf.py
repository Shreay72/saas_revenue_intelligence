"""
SaaS Revenue Intelligence -- Professional PDF Documentation Generator
Generates a comprehensive, visually polished PDF document using fpdf2.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from fpdf import FPDF

# --- Resolve project root ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_PATH = DOCS_DIR / "SaaS_Revenue_Intelligence_Documentation.pdf"

# --- Diagram image paths ----------------------------------------------------
IMG_ARCHITECTURE = str(DOCS_DIR / "architecture_diagram.png")
IMG_PIPELINE     = str(DOCS_DIR / "pipeline_diagram.png")
IMG_ER           = str(DOCS_DIR / "er_diagram.png")
IMG_DASHBOARD    = str(DOCS_DIR / "dashboard_mock.png")

# --- Color Palette ----------------------------------------------------------
NAVY       = (15, 23, 42)
DARK_BLUE  = (30, 58, 138)
TEAL       = (13, 148, 136)
ORANGE     = (234, 88, 12)
LIGHT_GRAY = (241, 245, 249)
WHITE      = (255, 255, 255)
TEXT_DARK  = (30, 41, 59)
TEXT_MED   = (71, 85, 105)
RED_ACCENT = (220, 38, 38)
GREEN_ACC  = (22, 163, 74)
AMBER      = (217, 119, 6)


class SaaSDocPDF(FPDF):
    """Custom PDF class with professional headers, footers, and helper methods."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=25)
        self.chapter_num = 0
        self.toc_entries = []
        self._in_cover = False

    # -- Header / Footer -----------------------------------------------------
    def header(self):
        if self._in_cover or self.page_no() <= 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*TEXT_MED)
        self.cell(0, 6, "SaaS Revenue Intelligence  |  Technical Documentation", align="L")
        self.cell(0, 6, f"v1.0.0  |  March 2026", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*TEAL)
        self.set_line_width(0.5)
        self.line(10, 14, 200, 14)
        self.ln(4)

    def footer(self):
        if self._in_cover:
            return
        self.set_y(-20)
        self.set_draw_color(*LIGHT_GRAY)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*TEXT_MED)
        self.cell(0, 5, "Confidential  |  SaaS Revenue Intelligence", align="L")
        self.cell(0, 5, f"Page {self.page_no()}", align="R")

    # -- Reusable building blocks ---------------------------------------------
    def chapter_title(self, title):
        self.chapter_num += 1
        self.toc_entries.append((self.chapter_num, title, self.page_no()))
        self.add_page()
        # Chapter number badge
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 28)
        self.cell(18, 14, f"{self.chapter_num:02d}", fill=True, align="C")
        self.set_x(32)
        self.set_text_color(*NAVY)
        self.set_font("Helvetica", "B", 22)
        self.cell(0, 14, title, new_x="LMARGIN", new_y="NEXT")
        # Accent line
        self.set_draw_color(*TEAL)
        self.set_line_width(1)
        self.line(10, self.get_y() + 2, 120, self.get_y() + 2)
        self.ln(10)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*DARK_BLUE)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_section(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*TEAL)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*TEXT_DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet_list(self, items):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*TEXT_DARK)
        for item in items:
            self.cell(6, 5.5, "-")   # bullet
            self.multi_cell(0, 5.5, f"  {item}")
            self.ln(0.5)
        self.ln(2)

    def callout_box(self, title, text, color=TEAL):
        y_start = self.get_y()
        # Left accent bar
        self.set_fill_color(*color)
        self.rect(10, y_start, 3, 0.1, style="F")  # placeholder
        # Background
        self.set_fill_color(color[0], color[1], color[2])
        self.rect(10, y_start, 3, 22, style="F")
        bg = (color[0] // 8 + 224, color[1] // 8 + 224, color[2] // 8 + 224)
        self.set_fill_color(*bg)
        self.rect(13, y_start, 187, 22, style="F")
        # Text inside
        self.set_xy(16, y_start + 2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.cell(0, 5, title, new_x="LMARGIN", new_y="NEXT")
        self.set_x(16)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*TEXT_DARK)
        self.multi_cell(180, 4.5, text)
        self.set_y(y_start + 24)
        self.ln(2)

    def kpi_row(self, kpis):
        """Draw a row of KPI cards. kpis = [(label, value, color), ...]"""
        w = (190 - (len(kpis) - 1) * 4) / len(kpis)
        y = self.get_y()
        for i, (label, value, color) in enumerate(kpis):
            x = 10 + i * (w + 4)
            # Card background
            self.set_fill_color(*color)
            self.rect(x, y, w, 22, style="F")
            # Value
            self.set_xy(x, y + 3)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*WHITE)
            self.cell(w, 8, value, align="C")
            # Label
            self.set_xy(x, y + 12)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(255, 255, 255)
            self.cell(w, 5, label, align="C")
        self.set_y(y + 28)

    def code_block(self, code, title=None):
        if title:
            self.set_font("Helvetica", "BI", 9)
            self.set_text_color(*TEXT_MED)
            self.cell(0, 5, title, new_x="LMARGIN", new_y="NEXT")
        self.set_fill_color(*LIGHT_GRAY)
        self.set_text_color(*TEXT_DARK)
        self.set_font("Courier", "", 8)
        lines = code.strip().split("\n")
        y_start = self.get_y()
        block_h = len(lines) * 4 + 6
        # Check page break
        if y_start + block_h > 270:
            self.add_page()
            y_start = self.get_y()
        self.rect(10, y_start, 190, block_h, style="F")
        self.set_xy(14, y_start + 3)
        for line in lines:
            self.cell(0, 4, line[:120], new_x="LMARGIN", new_y="NEXT")
            self.set_x(14)
        self.set_y(y_start + block_h + 3)

    def data_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header row
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 9)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Data rows
        self.set_font("Helvetica", "", 8.5)
        fill = False
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
            if fill:
                self.set_fill_color(*LIGHT_GRAY)
            else:
                self.set_fill_color(*WHITE)
            self.set_text_color(*TEXT_DARK)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(3)

    def add_image_safe(self, path, caption="", w=180):
        if os.path.exists(path):
            x = (210 - w) / 2
            if self.get_y() + 100 > 270:
                self.add_page()
            self.image(path, x=x, w=w)
            if caption:
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(*TEXT_MED)
                self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(4)
        else:
            self.body_text(f"[Diagram: {caption} -- image not found at {path}]")


def build_pdf():
    pdf = SaaSDocPDF()

    # ========================================================================
    # COVER PAGE
    # ========================================================================
    pdf._in_cover = True
    pdf.add_page()

    # Large navy background
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 297, style="F")

    # Teal accent strip
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 80, 210, 4, style="F")

    # Title
    pdf.set_y(95)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 16, "SaaS Revenue", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 36)
    pdf.cell(0, 16, "Intelligence", align="C", new_x="LMARGIN", new_y="NEXT")

    # Subtitle
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 10, "Technical Documentation & System Guide", align="C", new_x="LMARGIN", new_y="NEXT")

    # Divider
    pdf.ln(8)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.8)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())

    # Key stats boxes
    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*WHITE)
    stats = [
        ("500 Accounts", "$11.3M MRR", "$136M ARR"),
        ("3 ML Models", "182 Tests", "100% Pass Rate"),
    ]
    for row in stats:
        x_start = 30
        for stat in row:
            pdf.set_x(x_start)
            pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(*NAVY)
            pdf.cell(45, 10, stat, align="C", fill=True)
            x_start += 50
        pdf.ln(14)

    # Version & date at bottom
    pdf.set_y(240)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 7, "Version 1.0.0  |  March 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Python 3.11  |  scikit-learn  |  XGBoost  |  FastAPI  |  Streamlit", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Confidential -- For Internal Use", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf._in_cover = False

    # ========================================================================
    # TABLE OF CONTENTS (placeholder -- will adjust after all pages built)
    # ========================================================================
    toc_page = pdf.page_no() + 1

    # ========================================================================
    # CHAPTER 1 -- EXECUTIVE SUMMARY
    # ========================================================================
    pdf.chapter_title("Executive Summary")

    pdf.section_title("Project Vision")
    pdf.body_text(
        "SaaS Revenue Intelligence is an end-to-end machine learning platform designed to "
        "transform how SaaS businesses manage subscription revenue. By combining predictive "
        "analytics, risk scoring, and automated action planning, the system empowers Customer "
        "Success teams to move from reactive firefighting to systematic, data-driven account "
        "prioritization."
    )

    pdf.callout_box(
        "KEY INSIGHT",
        "SaaS companies lose 15-25% of revenue annually to preventable churn. "
        "This platform identifies at-risk accounts, quantifies financial impact, "
        "and prescribes interventions -- protecting $2.16M in recoverable revenue.",
        TEAL
    )

    pdf.section_title("Business Impact")
    pdf.kpi_row([
        ("Total Accounts", "500", DARK_BLUE),
        ("Monthly MRR", "$11.3M", GREEN_ACC),
        ("Revenue at Risk", "$7.97M", RED_ACCENT),
        ("Recoverable", "$2.16M", TEAL),
    ])

    pdf.section_title("Risk Distribution")
    pdf.data_table(
        ["Risk Tier", "Accounts", "% of Total", "MRR at Risk", "Intervention", "Recovery Rate"],
        [
            ["CRITICAL", "73", "15%", "~$2.8M", "VP Escalation", "50%"],
            ["HIGH",     "184", "37%", "~$3.2M", "TAM Assignment", "40%"],
            ["MEDIUM",   "95",  "19%", "~$1.2M", "Executive QBR", "30%"],
            ["LOW",      "148", "30%", "~$0.8M", "Upsell Focus", "20%"],
        ],
        col_widths=[25, 22, 20, 28, 40, 28]
    )

    pdf.section_title("ROI Summary")
    pdf.callout_box(
        "RETURN ON INVESTMENT",
        "Revenue at Risk: $7,968,352  |  Expected Recovery: $2,155,634  |  "
        "Implementation: ~2 weeks  |  Payback: < 1 month  |  "
        "Net ARR Protected: $2.16M/year  |  ROI: ~10x in Year 1",
        GREEN_ACC
    )

    pdf.section_title("Core Capabilities")
    pdf.bullet_list([
        "Churn Prediction: ML-powered customer churn forecasting (LogisticRegression, XGBoost, RandomForest)",
        "Revenue Forecasting: MRR prediction with GradientBoosting (R2 = 0.92, MAE = $2,744)",
        "Customer Lifetime Value: Deterministic CLV formula (Mean CLV = $570,043)",
        "Risk Scoring: Composite risk engine with configurable weights (Churn 40% + Revenue 35% + Health 25%)",
        "Action Planning: Automated intervention mapping with urgency levels and expected recovery",
        "Real-time API: FastAPI-powered REST endpoints for on-demand scoring",
        "Interactive Dashboard: Streamlit-based portfolio visualization with drill-down capabilities",
        "MLOps Monitoring: Automated data drift detection, model health tracking, and retrain triggers",
    ])

    # ========================================================================
    # CHAPTER 2 -- SYSTEM ARCHITECTURE & DIAGRAMS
    # ========================================================================
    pdf.chapter_title("System Architecture & Diagrams")

    pdf.section_title("High-Level Architecture")
    pdf.body_text(
        "The system follows a modular architecture with five distinct layers: "
        "Data Ingestion, ML Analytics Engine, Business Logic, API Serving, and Presentation. "
        "Each layer communicates through well-defined interfaces, enabling independent scaling."
    )
    pdf.add_image_safe(IMG_ARCHITECTURE, "Figure 2.1 -- High-Level System Architecture")

    pdf.section_title("Development Timeline")
    pdf.data_table(
        ["Week", "Phase", "Deliverables", "Status"],
        [
            ["Week 1", "Feature Engineering", "account_level_features.csv (500 x 36)", "Complete"],
            ["Week 2", "Churn Model", "churn_model_v1.pkl (LogisticRegression)", "Complete"],
            ["Week 3", "Revenue + CLV", "revenue_model_v1.pkl, clv_model_v1.pkl", "Complete"],
            ["Week 4", "Intelligence Engine", "account_intelligence.csv (500 x 20)", "Complete"],
            ["Week 5", "API + Dashboard", "FastAPI :8000, Streamlit :8501", "Complete"],
        ],
        col_widths=[20, 40, 75, 25]
    )

    pdf.section_title("Data Pipeline Workflow")
    pdf.body_text(
        "Raw data flows through a multi-stage pipeline: ingestion of 5 CSV sources, "
        "schema standardization, constraint-based validation, and feature engineering "
        "that produces 31 account-level features for downstream modeling."
    )
    pdf.add_image_safe(IMG_PIPELINE, "Figure 2.2 -- Data Pipeline Workflow")

    pdf.section_title("Data Sources")
    pdf.data_table(
        ["Raw File", "Key Columns", "Join Key"],
        [
            ["accounts.csv", "account_id, industry, seats, plan_tier", "account_id"],
            ["subscriptions.csv", "total_mrr, tenure_months, auto_renew_ratio", "account_id"],
            ["feature_usage.csv", "engagement_score, unique_features_used, error_rate", "account_id"],
            ["support_tickets.csv", "ticket_count, escalation_ratio, avg_resolution_time", "account_id"],
            ["churn_events.csv", "churn_flag (target variable)", "account_id"],
        ],
        col_widths=[38, 85, 25]
    )

    pdf.section_title("Entity-Relationship Diagram")
    pdf.body_text(
        "All data sources share account_id as the primary join key, enabling "
        "a star-schema design centered on the accounts table."
    )
    pdf.add_image_safe(IMG_ER, "Figure 2.3 -- Database Entity-Relationship Diagram")

    pdf.section_title("Production Architecture")
    pdf.code_block(
        "+-----------------+     +------------------+\n"
        "|  FastAPI :8000  |---->| Streamlit :8501  |\n"
        "|  /health        |     | Portfolio view   |\n"
        "|  /accounts      |     | Risk charts      |\n"
        "|  /portfolio     |     | Top 20 accounts  |\n"
        "|  /score         |     | Action drilldown |\n"
        "+--------+--------+     +------------------+\n"
        "         |\n"
        "         v\n"
        "+------------------------------------------+\n"
        "| account_intelligence.csv (500 x 20)     |\n"
        "| Cached in memory (TTL = 300s)           |\n"
        "+------------------------------------------+\n"
        "         |\n"
        "         v\n"
        "+------------------------------------------+\n"
        "| Monitoring                               |\n"
        "| DataDriftDetector -> drift_report.json   |\n"
        "| ModelMonitor      -> model_report.json   |\n"
        "| AlertManager      -> alert_log.json      |\n"
        "+------------------------------------------+",
        "Production Deployment Architecture"
    )

    pdf.section_title("Technology Stack")
    pdf.data_table(
        ["Layer", "Technology", "Purpose"],
        [
            ["Language", "Python 3.11", "Core runtime"],
            ["ML", "scikit-learn, XGBoost", "Model training & evaluation"],
            ["Optimization", "Optuna", "Hyperparameter tuning (TPE)"],
            ["API", "FastAPI + Uvicorn", "REST API server"],
            ["Dashboard", "Streamlit + Plotly", "Interactive visualization"],
            ["Config", "PyYAML", "Configuration management"],
            ["Serialization", "joblib", "Model artifact persistence"],
            ["Testing", "pytest", "182 tests, 100% pass rate"],
            ["Containers", "Docker + docker-compose", "Containerized deployment"],
        ],
        col_widths=[35, 55, 70]
    )

    # ========================================================================
    # CHAPTER 3 -- FOLDER STRUCTURE & FILE ORGANIZATION
    # ========================================================================
    pdf.chapter_title("Folder Structure & File Organization")

    pdf.section_title("Project Tree")
    pdf.code_block(
        "saas_revenue_intelligence/\n"
        "|\n"
        "|-- config/                    # Configuration files\n"
        "|   |-- config.yaml            # App settings, paths, API config\n"
        "|   |-- model_config.yaml      # Model hyperparams, risk weights\n"
        "|   |-- logging_config.yaml    # Logging levels and handlers\n"
        "|   |-- monitoring_config.yaml # Drift thresholds, alert channels\n"
        "|\n"
        "|-- data/\n"
        "|   |-- raw/                   # Original CSV data files\n"
        "|   |-- processed/            # Engineered features & intelligence\n"
        "|\n"
        "|-- src/                       # Core business logic\n"
        "|   |-- data/                  # Ingestion, cleaning, validation, FE\n"
        "|   |-- models/               # Training (churn, revenue, CLV)\n"
        "|   |-- pipelines/            # End-to-end orchestration\n"
        "|   |-- inference/            # Real-time & batch prediction\n"
        "|   |-- monitoring/           # Drift detection, model health\n"
        "|   |-- risk/                 # Composite risk scoring\n"
        "|   |-- recommendation/       # Business rules & actions\n"
        "|   |-- api/                  # FastAPI routes & middleware\n"
        "|   |-- utils/                # Helpers, logging, metrics\n"
        "|\n"
        "|-- models/                    # Trained model artifacts (.pkl)\n"
        "|   |-- churn/                # Churn model artifacts\n"
        "|   |-- revenue/              # Revenue & CLV model artifacts\n"
        "|\n"
        "|-- dashboard/                # Streamlit UI application\n"
        "|   |-- main.py               # Dashboard entry point\n"
        "|   |-- pages/                # Multi-page dashboard views\n"
        "|\n"
        "|-- notebooks/                # Jupyter analysis notebooks\n"
        "|-- scripts/                  # Automation & integration tests\n"
        "|-- tests/                    # pytest unit & integration tests\n"
        "|-- docs/                     # Documentation & diagrams\n"
        "|-- monitoring/               # Monitoring output files\n"
        "|-- reports/                  # Generated HTML/JSON reports\n"
        "|\n"
        "|-- Dockerfile                # Container definition\n"
        "|-- docker-compose.yml        # Multi-service orchestration\n"
        "|-- Makefile                  # CLI command shortcuts\n"
        "|-- requirements.txt          # Python dependencies\n"
        "|-- setup.py                  # Package configuration\n"
        "|-- README.md                 # Project overview",
        "Project Directory Structure"
    )

    pdf.section_title("Directory Purposes")
    pdf.data_table(
        ["Directory", "Purpose", "Key Files"],
        [
            ["config/", "YAML configuration management", "config.yaml, model_config.yaml"],
            ["src/data/", "Data ingestion & feature engineering", "data_loader.py, feature_engineering.py"],
            ["src/models/", "ML model training & evaluation", "train_churn.py, train_revenue.py"],
            ["src/pipelines/", "End-to-end pipeline orchestration", "churn_pipeline.py, preprocessing.py"],
            ["src/inference/", "Prediction serving (real-time/batch)", "predict.py, batch_predict.py"],
            ["src/monitoring/", "MLOps monitoring & alerting", "data_drift_detector.py, alert_manager.py"],
            ["src/risk/", "Composite risk scoring engine", "risk_engine.py"],
            ["src/api/", "FastAPI REST endpoints", "main.py, routes.py"],
            ["models/", "Serialized model artifacts", "churn_model_v1.pkl, revenue_model_v1.pkl"],
            ["dashboard/", "Streamlit interactive UI", "main.py, pages/*.py"],
            ["scripts/", "Automation & testing scripts", "train_all.py, generate_report.py"],
            ["tests/", "Automated test suites", "test_churn_model.py, test_api.py"],
        ],
        col_widths=[32, 62, 62]
    )

    # ========================================================================
    # CHAPTER 4 -- COMMANDS & SETUP INSTRUCTIONS
    # ========================================================================
    pdf.chapter_title("Commands & Setup Instructions")

    pdf.section_title("1. Environment Setup")
    pdf.code_block(
        "# Clone the repository\n"
        "git clone <repo-url>\n"
        "cd saas_revenue_intelligence\n"
        "\n"
        "# Create and activate virtual environment\n"
        "python -m venv venv\n"
        "\n"
        "# Windows\n"
        "venv\\Scripts\\activate\n"
        "\n"
        "# Linux / macOS\n"
        "source venv/bin/activate\n"
        "\n"
        "# Install all dependencies\n"
        "pip install -r requirements.txt",
        "Step 1: Python Environment Setup"
    )

    pdf.section_title("2. Dependencies")
    pdf.data_table(
        ["Package", "Version", "Purpose"],
        [
            ["numpy", ">= 1.24.0", "Numerical computing"],
            ["pandas", ">= 2.0.0", "Data manipulation"],
            ["scikit-learn", ">= 1.3.0", "ML model training"],
            ["xgboost", ">= 2.0.0", "Gradient boosting models"],
            ["joblib", ">= 1.3.0", "Model serialization"],
            ["PyYAML", ">= 6.0.0", "Configuration parsing"],
            ["fastapi", ">= 0.115.0", "REST API framework"],
            ["uvicorn", ">= 0.34.0", "ASGI web server"],
            ["httpx", ">= 0.28.0", "HTTP client for testing"],
            ["pytest", ">= 9.0.0", "Testing framework"],
            ["pytest-cov", ">= 7.0.0", "Coverage reporting"],
        ],
        col_widths=[40, 35, 80]
    )

    pdf.section_title("3. Training Models")
    pdf.code_block(
        "# Train all models (fast mode, ~10 seconds)\n"
        "python scripts/train_all.py --skip-week1 --no-tune\n"
        "\n"
        "# Train with Optuna hyperparameter tuning\n"
        "python scripts/train_all.py --skip-week1\n"
        "\n"
        "# Full pipeline from raw data\n"
        "python scripts/train_all.py --no-tune\n"
        "\n"
        "# Verify all artifacts exist\n"
        "python scripts/train_all.py --verify-only",
        "Step 2: Model Training"
    )

    pdf.section_title("4. Starting Services")
    pdf.code_block(
        "# Terminal 1: Start FastAPI server\n"
        "python scripts/run_api.py\n"
        "# Access: http://localhost:8000/docs\n"
        "\n"
        "# Terminal 2: Start Streamlit dashboard\n"
        "python scripts/run_streamlit.py\n"
        "# Access: http://localhost:8501",
        "Step 3: Launch Services"
    )

    pdf.section_title("5. Docker Deployment")
    pdf.code_block(
        "# Build and start all services\n"
        "docker-compose up --build\n"
        "\n"
        "# Run in background\n"
        "docker-compose up -d\n"
        "\n"
        "# View logs\n"
        "docker-compose logs -f api\n"
        "docker-compose logs -f dashboard\n"
        "\n"
        "# Stop services\n"
        "docker-compose down",
        "Docker Deployment"
    )

    pdf.section_title("6. Makefile Quick Reference")
    pdf.data_table(
        ["Command", "Description"],
        [
            ["make install", "Install all Python dependencies"],
            ["make train", "Train models (fast, no tuning)"],
            ["make train-full", "Train with Optuna hyperparameter tuning"],
            ["make api", "Start FastAPI server (port 8000)"],
            ["make dashboard", "Start Streamlit dashboard (port 8501)"],
            ["make test", "Run all 182 tests"],
            ["make monitor", "Run monitoring health checks"],
            ["make report", "Generate HTML + JSON portfolio report"],
            ["make retrain", "Retrain if monitoring recommends it"],
            ["make docker", "Build and start Docker containers"],
            ["make clean", "Remove logs, reports, monitoring outputs"],
        ],
        col_widths=[45, 115]
    )

    # ========================================================================
    # CHAPTER 5 -- JUPYTER NOTEBOOKS & CODE WALKTHROUGHS
    # ========================================================================
    pdf.chapter_title("Jupyter Notebooks & Code Walkthroughs")

    pdf.section_title("Available Notebooks")
    pdf.data_table(
        ["Notebook", "Purpose", "Key Outputs"],
        [
            ["01_data_exploration.ipynb", "EDA on raw datasets", "Distribution plots, correlations"],
            ["02_feature_engineering.ipynb", "Feature creation walkthrough", "31 features from 5 sources"],
            ["04_revenue_modeling.ipynb", "Revenue model training", "GradientBoosting R2=0.92"],
            ["05_model_evaluation.ipynb", "Comprehensive model eval", "ROC curves, confusion matrices"],
            ["06_shap_interpretation.ipynb", "SHAP feature explanations", "Global & local importance"],
            ["07_model_drift_analysis.ipynb", "Drift detection analysis", "Feature stability reports"],
        ],
        col_widths=[55, 55, 55]
    )

    pdf.section_title("Code Walkthrough: Feature Engineering")
    pdf.body_text(
        "The FeatureEngineer class in src/data/feature_engineering.py orchestrates the creation "
        "of 31 account-level features by joining and aggregating data from 5 raw sources. "
        "The pipeline processes identity attributes, financial metrics, engagement signals, "
        "support quality indicators, and derived state/signal features."
    )
    pdf.code_block(
        'class FeatureEngineer:\n'
        '    """Build complete account-level dataset with 34 features."""\n'
        '\n'
        '    def __init__(self, reference_date=None):\n'
        '        self.reference_date = reference_date or datetime.now()\n'
        '\n'
        '    def build_account_level_dataset(self, datasets):\n'
        '        """\n'
        '        Joins 5 raw datasets on account_id and engineers:\n'
        '        - Subscription metrics (MRR, tenure, renewal)\n'
        '        - Engagement features (usage, errors, adoption)\n'
        '        - Support signals (tickets, resolution, CSAT)\n'
        '        - Derived signals (trends, pressure indicators)\n'
        '        Returns: DataFrame with 500 rows x 36 columns\n'
        '        """',
        "src/data/feature_engineering.py"
    )

    pdf.section_title("Code Walkthrough: Business Metrics")
    pdf.body_text(
        "The metrics module (src/utils/metrics.py) implements core SaaS business calculations "
        "used across the platform. Key formulas are transparent and configurable."
    )
    pdf.code_block(
        'def calculate_health_score(\n'
        '    engagement_score, support_risk_score,\n'
        '    churn_probability, tenure_months, auto_renew_ratio\n'
        ') -> float:\n'
        '    """Weighted health score (0-100)"""\n'
        '    score = (\n'
        '        0.30 * engagement_norm +\n'
        '        0.25 * support_health +\n'
        '        0.25 * churn_health +\n'
        '        0.10 * tenure_norm +\n'
        '        0.10 * auto_renew_norm\n'
        '    ) * 100\n'
        '    return round(score, 2)\n'
        '\n'
        'def calculate_clv(mrr, churn_probability,\n'
        '                  gross_margin=0.70, discount_rate=0.10):\n'
        '    """CLV = (MRR x gross_margin) / (monthly_churn + discount)"""\n'
        '    monthly_churn = churn_probability / 12\n'
        '    monthly_discount = discount_rate / 12\n'
        '    return (mrr * gross_margin) / (monthly_churn + monthly_discount)\n'
        '\n'
        'def calculate_composite_risk_score(\n'
        '    churn_probability, revenue_at_risk,\n'
        '    health_score, p95_revenue, weights=None\n'
        '):\n'
        '    """Risk = (w1*churn + w2*rev_norm + w3*(1-health)) * 100"""\n'
        '    # Weights: churn=0.40, revenue=0.35, health=0.25\n'
        '    return clipped to [0, 100]',
        "src/utils/metrics.py -- Core Business Formulas"
    )

    pdf.section_title("Code Walkthrough: Churn Model Training")
    pdf.body_text(
        "The ChurnModelTrainer uses Optuna for hyperparameter optimization across three "
        "model architectures. It includes cost-sensitive evaluation, probability calibration, "
        "and SHAP-based explainability."
    )
    pdf.code_block(
        'class ChurnModelTrainer:\n'
        '    """Elite churn model trainer with Optuna optimization."""\n'
        '\n'
        '    def __init__(self, tune_hyperparameters=True,\n'
        '                 n_trials=50, cost_fp=100.0, cost_fn=5000.0):\n'
        '        # cost_fp: False positive cost ($100 per retention offer)\n'
        '        # cost_fn: False negative cost ($5,000 per missed churner)\n'
        '\n'
        '    def train_logistic_regression(self, cv_folds=5):\n'
        '        """Optuna-tuned LR with isotonic calibration"""\n'
        '\n'
        '    def train_xgboost(self, cv_folds=5):\n'
        '        """XGBoost with early stopping, Optuna tuning"""\n'
        '\n'
        '    def compare_models(self):\n'
        '        """Compare all models on ROC-AUC, F1, Brier, Cost"""\n'
        '\n'
        '    def save_best_model(self, output_dir="models/churn"):\n'
        '        """Persist best model + metadata + feature importance"""',
        "src/models/train_churn.py -- Model Training Architecture"
    )

    # ========================================================================
    # CHAPTER 6 -- BUSINESS INTELLIGENCE LAYER
    # ========================================================================
    pdf.chapter_title("Business Intelligence Layer")

    pdf.section_title("Dashboard Overview")
    pdf.body_text(
        "The SaaS Revenue Intelligence dashboard provides real-time portfolio visibility "
        "through an interactive Streamlit interface. It connects to the FastAPI backend "
        "for live data and supports drill-down analysis of individual accounts."
    )
    pdf.add_image_safe(IMG_DASHBOARD, "Figure 6.1 -- Revenue Intelligence Dashboard (Mock)")

    pdf.section_title("Key Performance Indicators (KPIs)")
    pdf.data_table(
        ["KPI", "Formula", "Current Value", "Business Use"],
        [
            ["MRR", "Sum of monthly recurring revenue", "$11,338,747", "Revenue tracking"],
            ["ARR", "MRR x 12", "$136,064,964", "Annual forecasting"],
            ["Churn Rate", "Churned / Total accounts", "70%", "Retention health"],
            ["CLV", "(MRR x margin) / (churn + discount)", "Mean $570K", "Customer valuation"],
            ["Revenue at Risk", "MRR x churn_probability", "$7,968,352", "Risk quantification"],
            ["NRR", "(Start + Expansion - Churn) / Start", "Computed", "Growth efficiency"],
            ["Health Score", "Weighted composite (0-100)", "Per account", "Account health"],
            ["Risk Score", "40% churn + 35% rev + 25% health", "0-100", "Prioritization"],
        ],
        col_widths=[32, 50, 32, 42]
    )

    pdf.section_title("Risk Engine Configuration")
    pdf.body_text(
        "The risk engine uses configurable weights defined in config/model_config.yaml. "
        "The composite risk score combines three signals into a single 0-100 score."
    )
    pdf.code_block(
        "# config/model_config.yaml\n"
        "risk_engine:\n"
        "  weights:\n"
        "    churn_probability: 0.40   # Primary signal\n"
        "    revenue_at_risk:   0.35   # Financial impact\n"
        "    health_score:      0.25   # Product health\n"
        "\n"
        "  tier_thresholds:\n"
        "    critical: 75    # Score >= 75 -> CRITICAL\n"
        "    high:     60    # Score >= 60 -> HIGH\n"
        "    medium:   35    # Score >= 35 -> MEDIUM\n"
        "                    # Score <  35 -> LOW\n"
        "\n"
        "intervention_success_rates:\n"
        "  VP_ESCALATION:  0.50    # 50% recovery rate\n"
        "  TAM_ASSIGNMENT: 0.40    # 40% recovery rate\n"
        "  EXECUTIVE_QBR:  0.30    # 30% recovery rate\n"
        "  UPSELL:         0.20    # 20% recovery rate",
        "Risk Engine Configuration"
    )

    pdf.section_title("Model Performance Summary")
    pdf.sub_section("Churn Model (Classification)")
    pdf.data_table(
        ["Model", "ROC-AUC", "F1", "Precision", "Recall", "Brier Score", "Business Cost"],
        [
            ["LogisticRegression", "1.0000", "1.0000", "1.0000", "1.0000", "0.0005", "$0"],
            ["XGBoost", "1.0000", "1.0000", "1.0000", "1.0000", "0.0001", "$0"],
            ["RandomForest", "1.0000", "1.0000", "1.0000", "1.0000", "0.0331", "$0"],
        ],
        col_widths=[35, 22, 20, 22, 20, 25, 28]
    )
    pdf.callout_box(
        "NOTE ON PERFECT SCORES",
        "All models show ROC-AUC = 1.0 due to churn_probability data leakage in training features. "
        "After removing this feature, expected realistic AUC: 0.75 - 0.85. Winner: LogisticRegression (best calibration, Brier = 0.0005).",
        AMBER
    )

    pdf.sub_section("Revenue Model (Regression)")
    pdf.data_table(
        ["Model", "R2 Mean", "R2 Std", "MAE Mean", "MAE Std"],
        [
            ["GradientBoosting", "0.9182", "+/- 0.0413", "$2,744", "+/- $342"],
            ["RandomForest", "0.8193", "+/- 0.0698", "$4,260", "+/- $477"],
            ["Ridge", "0.6146", "+/- 0.0678", "$7,560", "+/- $707"],
        ],
        col_widths=[38, 30, 30, 30, 30]
    )

    # ========================================================================
    # CHAPTER 7 -- API INTEGRATIONS
    # ========================================================================
    pdf.chapter_title("API Integrations")

    pdf.section_title("API Overview")
    pdf.body_text(
        "The platform exposes a RESTful API built with FastAPI, providing programmatic access "
        "to all intelligence capabilities. The API supports health checks, portfolio queries, "
        "individual account scoring, and cache management."
    )
    pdf.data_table(
        ["Base URL", "Interactive Docs", "ReDoc"],
        [["http://localhost:8000", "http://localhost:8000/docs", "http://localhost:8000/redoc"]],
        col_widths=[55, 65, 55]
    )

    pdf.section_title("Health & Status Endpoints")
    pdf.code_block(
        'GET /health\n'
        '{\n'
        '  "status": "healthy",\n'
        '  "accounts_loaded": 500,\n'
        '  "pipeline_ready": true,\n'
        '  "model_version": "1.0.0",\n'
        '  "cache_status": "fresh"\n'
        '}\n'
        '\n'
        'GET /status\n'
        '{\n'
        '  "api_version": "1.0.0",\n'
        '  "churn_model": "LogisticRegression",\n'
        '  "revenue_model": "GradientBoosting",\n'
        '  "accounts_count": 500\n'
        '}',
        "Health & Status API Responses"
    )

    pdf.section_title("Portfolio Endpoints")
    pdf.code_block(
        'GET /portfolio\n'
        '{\n'
        '  "total_accounts": 500,\n'
        '  "tier_distribution": {\n'
        '    "CRITICAL": 73, "HIGH": 184,\n'
        '    "MEDIUM": 95, "LOW": 148\n'
        '  },\n'
        '  "total_revenue_at_risk": 7968352.23,\n'
        '  "total_recoverable": 2155634.15,\n'
        '  "top_20_accounts": [...]\n'
        '}',
        "Portfolio Summary Response"
    )

    pdf.section_title("Account Intelligence Endpoints")
    pdf.data_table(
        ["Endpoint", "Method", "Parameters", "Description"],
        [
            ["/accounts", "GET", "page, limit, tier, urgency", "Paginated account list with filters"],
            ["/accounts/{id}", "GET", "account_id", "Detailed account intelligence"],
            ["/accounts/{id}/risk", "GET", "account_id", "Risk breakdown for account"],
            ["/accounts/{id}/recommendation", "GET", "account_id", "Action recommendation"],
            ["/portfolio", "GET", "none", "Portfolio summary with tier distribution"],
            ["/portfolio/critical", "GET", "none", "Critical tier accounts only"],
            ["/score", "POST", "JSON body", "Real-time scoring of new data"],
        ],
        col_widths=[48, 18, 48, 52]
    )

    pdf.section_title("Real-time Scoring")
    pdf.code_block(
        '# POST /score -- Real-time account scoring\n'
        'curl -X POST http://localhost:8000/score \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{\n'
        '    "total_mrr": 15000,\n'
        '    "engagement_score": 25,\n'
        '    "ticket_count": 45,\n'
        '    "tenure_months": 12,\n'
        '    "auto_renew_ratio": 0.3\n'
        '  }\'\n'
        '\n'
        '# Response:\n'
        '{\n'
        '  "risk_score": 87.3,\n'
        '  "risk_tier": "CRITICAL",\n'
        '  "churn_probability": 0.91,\n'
        '  "health_score": 32.1,\n'
        '  "revenue_at_risk": 13650.00,\n'
        '  "recommended_action": "VP_ESCALATION",\n'
        '  "scored_at": "2026-03-07T19:40:00"\n'
        '}',
        "Real-time Scoring Example"
    )

    pdf.section_title("Cache Management")
    pdf.data_table(
        ["Endpoint", "Method", "Description"],
        [
            ["/cache/status", "GET", "Check cache freshness (TTL = 300 seconds)"],
            ["/cache/refresh", "POST", "Force cache invalidation and data reload"],
        ],
        col_widths=[50, 20, 90]
    )

    # ========================================================================
    # CHAPTER 8 -- BEST PRACTICES & RECOMMENDATIONS
    # ========================================================================
    pdf.chapter_title("Best Practices & Recommendations")

    pdf.section_title("Coding Standards")
    pdf.bullet_list([
        "Python 3.11+ with type hints on all public functions",
        "Docstrings on all classes and public methods (Google style)",
        "Modular architecture: one responsibility per module",
        "Configuration-driven: all magic numbers in YAML config files",
        "Consistent logging with structured colored output (src/utils/logger.py)",
        "Error handling with graceful fallbacks and informative messages",
    ])

    pdf.section_title("Version Control Workflow (Git)")
    pdf.bullet_list([
        "main branch: production-ready code only",
        "develop branch: integration branch for feature merges",
        "feature/* branches: individual feature development",
        "Semantic versioning: MAJOR.MINOR.PATCH (currently v1.0.0)",
        "Commit messages: conventional commits format (feat:, fix:, docs:)",
        "Pull requests require passing CI checks before merge",
    ])

    pdf.section_title("CI/CD Pipeline Overview")
    pdf.code_block(
        "Pipeline Stages:\n"
        "\n"
        "1. LINT      -> flake8, black, isort\n"
        "2. TEST      -> pytest tests/ -v (182 tests)\n"
        "3. COVERAGE  -> pytest --cov=src --cov-report=html\n"
        "4. BUILD     -> docker build -t saas-intelligence .\n"
        "5. DEPLOY    -> docker-compose up -d (staging/prod)\n"
        "6. MONITOR   -> python scripts/monitor_model.py\n"
        "7. REPORT    -> python scripts/generate_report.py",
        "Recommended CI/CD Pipeline"
    )

    pdf.section_title("Security Considerations")
    pdf.data_table(
        ["Area", "Implementation", "Status"],
        [
            ["Data Privacy", "No PII in model features, anonymized account IDs", "Implemented"],
            ["API Rate Limiting", "slowapi integration for request throttling", "Implemented"],
            ["Authentication", "API key / OAuth2 for production endpoints", "Planned"],
            ["Data Encryption", "TLS for API, encrypted storage for credentials", "Planned"],
            ["Access Control", "Role-based access for dashboard views", "Planned"],
            ["Audit Logging", "All API requests logged with timestamps", "Implemented"],
            ["Model Security", "Model artifacts stored locally, version controlled", "Implemented"],
        ],
        col_widths=[40, 80, 30]
    )

    pdf.section_title("Monitoring Best Practices")
    pdf.bullet_list([
        "Run monitoring daily (automated via Docker cron or scripts/monitor_model.py)",
        "Data drift detection: normalized mean shift with 5% (WARNING) and 10% (CRITICAL) thresholds",
        "Model health: 4-check system (churn rate sanity, calibration, AUC estimate, portfolio drift)",
        "Alert channels: Console (always on), File logging (always on), Email (configurable SMTP)",
        "Auto-retrain triggers: 1+ CRITICAL = RETRAIN_REQUIRED, 2+ WARNING = RETRAIN_RECOMMENDED",
        "Baseline management: auto-created on first run, manual reset after intentional data changes",
    ])

    # ========================================================================
    # CHAPTER 9 -- FUTURE ENHANCEMENTS
    # ========================================================================
    pdf.chapter_title("Future Enhancements")

    pdf.section_title("Scaling Roadmap")
    pdf.data_table(
        ["Scale", "Accounts", "Infrastructure Required"],
        [
            ["Current", "500", "2 CPU, 4GB RAM (single machine)"],
            ["Phase 1", "5,000", "+ Redis cache + PostgreSQL database"],
            ["Phase 2", "50,000", "+ Kubernetes + S3 storage + read replicas"],
            ["Phase 3", "500,000", "+ Apache Spark + distributed ML training"],
        ],
        col_widths=[30, 30, 100]
    )

    pdf.section_title("Predictive Analytics Enhancements")
    pdf.bullet_list([
        "Time-series churn prediction: forecast churn probability trajectory over next 30/60/90 days",
        "Cohort analysis: automated segmentation by industry, plan tier, and engagement patterns",
        "Upsell propensity modeling: predict accounts most likely to upgrade",
        "Revenue trajectory forecasting: project MRR growth/decline per account",
        "Seasonal pattern detection: identify recurring churn/expansion cycles",
    ])

    pdf.section_title("AI-Driven Insights")
    pdf.bullet_list([
        "Natural language explanations: GPT-powered account risk narratives for CSM briefings",
        "Automated playbook generation: AI-generated intervention strategies per risk profile",
        "Anomaly detection: unsupervised learning to flag unusual account behavior patterns",
        "Churn root cause analysis: automated identification of top 3 drivers per account",
        "Sentiment analysis: integrate NLP on support ticket text for early warning signals",
    ])

    pdf.section_title("Platform Improvements")
    pdf.bullet_list([
        "Real-time streaming: Apache Kafka for live event processing and scoring",
        "A/B testing framework: test intervention effectiveness with controlled experiments",
        "Multi-tenant support: serve multiple SaaS products from single infrastructure",
        "Mobile dashboard: responsive design or native app for on-the-go monitoring",
        "Webhook integrations: push alerts to Slack, PagerDuty, Salesforce, HubSpot",
        "Data warehouse integration: connect to Snowflake, BigQuery, or Redshift",
        "Feature store: centralized feature management with versioning and lineage",
    ])

    pdf.section_title("Production Improvements")
    pdf.callout_box(
        "IMMEDIATE ACTION ITEMS",
        "1. Remove churn_probability leakage from training features  |  "
        "2. Add SHAP explanations to /explain/{account_id} endpoint  |  "
        "3. Track feature importance weekly via drift detector  |  "
        "4. A/B test risk engine weights quarterly  |  "
        "5. Implement OAuth2 authentication for API endpoints",
        ORANGE
    )

    # ========================================================================
    # CHAPTER 10 -- APPENDIX: TESTING & TROUBLESHOOTING
    # ========================================================================
    pdf.chapter_title("Testing & Troubleshooting")

    pdf.section_title("Testing Strategy")
    pdf.data_table(
        ["Phase", "Script", "Tests", "Purpose"],
        [
            ["Phase 1: Data", "test_week1_pipeline.py", "Integration", "E2E data flow validation"],
            ["Phase 2: Models", "test_week2_models.py", "Integration", "Training orchestration"],
            ["Phase 3: Inference", "test_inference.py", "Integration", "Pipeline prediction logic"],
            ["Phase 4: Risk", "test_week4_risk.py", "Integration", "Risk engine & intelligence"],
            ["Unit Tests", "tests/test_churn_model.py", "Unit", "Model evaluator & pipeline"],
            ["API Tests", "tests/test_api.py", "Unit", "REST endpoint validation"],
        ],
        col_widths=[32, 45, 25, 55]
    )

    pdf.section_title("Running Tests")
    pdf.code_block(
        "# Run all 182 tests\n"
        "pytest tests/ -v\n"
        "\n"
        "# Run with coverage report\n"
        "pytest tests/ -v --cov=src --cov-report=html\n"
        "\n"
        "# Quick test (stop on first failure)\n"
        "pytest tests/ -x -q\n"
        "\n"
        "# Run specific integration test\n"
        "python scripts/test_week2_models.py",
        "Test Commands"
    )

    pdf.section_title("Common Troubleshooting")
    pdf.data_table(
        ["Error", "Cause", "Solution"],
        [
            ["FileNotFoundError: churn_model_v1.pkl", "Models not trained", "python scripts/train_all.py --no-tune"],
            ["KeyError: revenue_at_risk", "Pipeline order issue", "Run revenue pipeline before intelligence"],
            ["ROC-AUC = 1.0000", "Data leakage", "Remove churn_probability from features"],
            ["API not reachable", "API not started", "Start API first: python scripts/run_api.py"],
            ["No module named X", "Missing deps", "pip install -r requirements.txt"],
            ["Baseline not found", "First monitoring run", "Expected; auto-creates on first run"],
            ["Port already in use", "Stale process", "Kill process: netstat -ano | findstr :8000"],
        ],
        col_widths=[52, 38, 65]
    )

    pdf.section_title("Useful Debug Commands")
    pdf.code_block(
        "# Verify all model artifacts exist\n"
        "python scripts/train_all.py --verify-only\n"
        "\n"
        "# Check API health\n"
        "curl http://localhost:8000/health\n"
        "\n"
        "# Check account count\n"
        'python -c "import pandas as pd; print(pd.read_csv(\n'
        "    'data/processed/account_intelligence.csv').shape)\"\n"
        "\n"
        "# Check risk distribution\n"
        'python -c "import pandas as pd; print(pd.read_csv(\n'
        "    'data/processed/account_intelligence.csv')['risk_tier'].value_counts())\"",
        "Debug Commands"
    )

    # ========================================================================
    # SAVE PDF
    # ========================================================================
    print(f"Generating PDF with {pdf.page_no()} pages...")
    pdf.output(str(OUTPUT_PATH))
    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"PDF saved to: {OUTPUT_PATH}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    return str(OUTPUT_PATH)


if __name__ == "__main__":
    build_pdf()
