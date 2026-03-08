"""
Streamlit Dashboard Launcher
SaaS Revenue Intelligence System

Checks API health before launching dashboard.

Usage:
    python scripts/run_streamlit.py
    python scripts/run_streamlit.py --no-check   (skip API check)
    python scripts/run_streamlit.py --port 8502
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

API_BASE_URL  = "http://localhost:8000"
DEFAULT_PORT  = 8501
DASHBOARD_ENTRY = str(project_root / "dashboard" / "main.py")


def check_api_health(timeout: int = 5) -> bool:
    """Check if the FastAPI backend is reachable and healthy."""
    try:
        import requests
        resp = requests.get(f"{API_BASE_URL}/health", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            status   = data.get("status", "unknown")
            accounts = data.get("accounts_loaded", "?")
            version  = data.get("version", "?")
            print(f"  ✅ API is healthy")
            print(f"     Status:   {status}")
            print(f"     Accounts: {accounts}")
            print(f"     Version:  {version}")
            return True
        else:
            print(f"  ⚠️  API returned HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ API not reachable: {e}")
        return False


def launch_dashboard(port: int):
    """Launch the Streamlit dashboard."""
    print(f"\n  Launching Streamlit dashboard on port {port}...")
    print(f"  Entry: {DASHBOARD_ENTRY}")
    print(f"\n  Dashboard URL: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        DASHBOARD_ENTRY,
        "--server.port", str(port),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]

    try:
        subprocess.run(cmd, cwd=str(project_root), check=True)
    except KeyboardInterrupt:
        print("\n\n  Dashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\n  ❌ Streamlit failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Streamlit dashboard"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip API health check before launching"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Streamlit port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=API_BASE_URL,
        help=f"FastAPI base URL (default: {API_BASE_URL})"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  📊 SaaS Revenue Intelligence — Dashboard Launcher")
    print("=" * 60)

    # API health check
    if not args.no_check:
        print(f"\n  Checking API health at {args.api_url}...")
        healthy = check_api_health()

        if not healthy:
            print("""
  ⚠️  API is not running. The dashboard requires the API.

  Start the API first:
    python scripts/run_api.py

  Or skip this check (dashboard will show errors):
    python scripts/run_streamlit.py --no-check
""")
            sys.exit(1)
    else:
        print("\n  ⏭️  Skipping API health check (--no-check flag set)")

    # Launch
    launch_dashboard(port=args.port)


if __name__ == "__main__":
    main()
