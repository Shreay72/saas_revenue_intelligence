"""
API Server Launcher
Run:
    python scripts/run_api.py
"""

import sys
import uvicorn
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 SaaS Revenue Intelligence API")
    print("=" * 60)
    print("📖 Swagger UI  → http://localhost:8000/docs")
    print("📖 ReDoc       → http://localhost:8000/redoc")
    print("❤️  Health      → http://localhost:8000/health")
    print("=" * 60)

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
