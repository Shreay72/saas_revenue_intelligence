"""
SaaS Revenue Risk Intelligence API
Week 5 — FastAPI Application Entry Point

Middleware stack:
    GZip → CORS → Logging → Rate Limiting

Routers:
    health.py    → /health, /api/v1/status
    accounts.py  → /api/v1/accounts/*
    portfolio.py → /api/v1/portfolio/*, /api/v1/cache/*
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.dependencies import load_intelligence
from src.api.middleware.error_handler import (
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.api.routers import health, accounts, portfolio

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("api.main")

# ─────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# ─────────────────────────────────────────────
# LIFESPAN — replaces deprecated @app.on_event
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────
    logger.info("=" * 60)
    logger.info("🚀 SaaS Revenue Intelligence API starting up...")
    logger.info("=" * 60)
    load_intelligence()
    logger.info("✅ API ready.")
    logger.info("📖 Swagger docs: http://localhost:8000/docs")
    logger.info("📖 ReDoc:        http://localhost:8000/redoc")

    yield   # ← API is live and serving requests here

    # ── SHUTDOWN ─────────────────────────────
    logger.info("🛑 API shutting down.")


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="SaaS Revenue Risk Intelligence API",
    version="1.0.0",
    description=(
        "Production-grade risk scoring and retention recommendation engine "
        "for SaaS customer portfolios. "
        "Powered by ML churn prediction (Week 2), deterministic CLV (Week 3), "
        "and rule-based intervention recommendations (Week 4)."
    ),
    contact={
        "name": "SaaS Intelligence Team",
    },
    license_info={
        "name": "Internal Use Only",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,          # ← modern lifespan handler
)

# ─────────────────────────────────────────────
# RATE LIMITER STATE
# ─────────────────────────────────────────────

app.state.limiter = limiter

# ─────────────────────────────────────────────
# MIDDLEWARE STACK (order matters — outermost first)
# ─────────────────────────────────────────────

# 1. GZip compression — compress responses ≥ 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 2. CORS — allow Streamlit dashboard (Week 6)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",     # Streamlit
        "http://localhost:3000",     # React (future)
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 3. Structured JSON request logging
app.add_middleware(LoggingMiddleware)

# ─────────────────────────────────────────────
# EXCEPTION HANDLERS
# ─────────────────────────────────────────────

app.add_exception_handler(StarletteHTTPException,  http_exception_handler)
app.add_exception_handler(RequestValidationError,  validation_exception_handler)
app.add_exception_handler(Exception,               generic_exception_handler)
app.add_exception_handler(RateLimitExceeded,       _rate_limit_exceeded_handler)

# ─────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────

app.include_router(health.router)
app.include_router(accounts.router)
app.include_router(portfolio.router)
