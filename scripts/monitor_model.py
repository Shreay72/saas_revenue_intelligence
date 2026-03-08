"""
Model Monitor Script
SaaS Revenue Intelligence System — Week 5

Standalone monitoring runner. Runs drift detection + model health
checks and dispatches alerts. Designed for daily cron execution.

Usage:
    python scripts/monitor_model.py
    python scripts/monitor_model.py --features-csv data/processed/account_level_features.csv
    python scripts/monitor_model.py --intelligence-csv data/processed/account_intelligence.csv
    python scripts/monitor_model.py --reset-baseline
    python scripts/monitor_model.py --report
"""

import sys
import time
import argparse
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.utils.logger import get_logger
from src.monitoring.data_drift_detector import DataDriftDetector
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.alert_manager import AlertManager

logger = get_logger(__name__)


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_monitoring(
    features_csv: str = "data/processed/account_level_features.csv",
    intelligence_csv: str = "data/processed/account_intelligence.csv",
    reset_baseline: bool = False,
    generate_report: bool = False,
) -> dict:
    """
    Run full monitoring pipeline:
        1. Data drift detection on intelligence CSV
        2. Model health monitoring on features CSV
        3. Alert dispatching
        4. Optionally generate HTML report

    Returns combined results dict.
    """
    start = time.time()

    # ── Load data ─────────────────────────────────────────────────────
    print_section("Loading Data")

    features_path     = Path(features_csv)
    intelligence_path = Path(intelligence_csv)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features CSV not found: {features_path}\n"
            "Run: python scripts/train_all.py --skip-week1 --no-tune"
        )
    if not intelligence_path.exists():
        raise FileNotFoundError(
            f"Intelligence CSV not found: {intelligence_path}\n"
            "Run: python scripts/train_all.py --skip-week1 --no-tune"
        )

    df_features     = pd.read_csv(features_path)
    df_intelligence = pd.read_csv(intelligence_path)

    print(f"  Features CSV    : {len(df_features)} accounts × {df_features.shape[1]} cols")
    print(f"  Intelligence CSV: {len(df_intelligence)} accounts × {df_intelligence.shape[1]} cols")

    manager = AlertManager()

    # ── Reset baseline if requested ───────────────────────────────────
    if reset_baseline:
        print_section("Resetting Baseline")
        detector = DataDriftDetector()
        detector.create_baseline(df_intelligence)
        print("  ✅ Baseline reset from current intelligence data.")

    # ── Step 1: Data Drift ────────────────────────────────────────────
    print_section("Step 1 — Data Drift Detection")
    detector    = DataDriftDetector()
    drift_report = detector.run(df_intelligence)

    print(f"\n  Overall Status  : {drift_report['overall_status']}")
    print(f"  Features Checked: {drift_report['features_checked']}")
    print(f"  Critical        : {drift_report['n_critical']}")
    print(f"  Warning         : {drift_report['n_warning']}")
    print(f"  OK              : {drift_report['n_ok']}")

    # ── Step 2: Model Health ──────────────────────────────────────────
    print_section("Step 2 — Model Health Monitor")
    monitor      = ModelMonitor()
    model_report = monitor.run(df_features)

    print(f"\n  Recommendation  : {model_report['recommendation']}")
    for check, result in model_report["checks"].items():
        status = result.get("status", "?")
        emoji  = "✅" if status == "OK" else ("⚠️ " if status == "WARNING" else "❌")
        print(f"  {emoji} {check:<28} → {status}")

    # ── Step 3: Alerts ────────────────────────────────────────────────
    print_section("Step 3 — Alert Dispatching")
    manager.process_drift_report(drift_report)
    manager.process_model_report(model_report)
    manager.flush()

    summary = manager.summary()
    print(f"\n  Alerts dispatched:")
    print(f"    INFO     : {summary.get('INFO', 0)}")
    print(f"    WARNING  : {summary.get('WARNING', 0)}")
    print(f"    CRITICAL : {summary.get('CRITICAL', 0)}")

    # ── Step 4: Optional Report ───────────────────────────────────────
    if generate_report:
        print_section("Step 4 — Generating Report")
        from scripts.generate_report import load_intelligence, load_monitoring_reports, build_json_report, build_html_report
        from datetime import datetime

        df        = load_intelligence()
        monitoring = load_monitoring_reports()
        report    = build_json_report(df, monitoring)

        import json
        from pathlib import Path as P
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out  = P("reports")
        out.mkdir(parents=True, exist_ok=True)

        (out / f"portfolio_report_{ts}.json").write_text(json.dumps(report, indent=2))
        (out / f"portfolio_report_{ts}.html").write_text(
            build_html_report(report, df), encoding="utf-8"
        )
        print(f"  ✅ Reports saved to reports/portfolio_report_{ts}.*")

    # ── Final summary ─────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")

    overall_ok = (
        drift_report["overall_status"] == "OK" and
        model_report["recommendation"] == "OK"
    )
    if overall_ok:
        print(f"  ✅ ALL CHECKS PASSED — System is healthy ({elapsed:.1f}s)")
    else:
        print(f"  ⚠️  ISSUES DETECTED — Review alerts above ({elapsed:.1f}s)")
        if model_report["recommendation"] in ("RETRAIN_REQUIRED", "RETRAIN_RECOMMENDED"):
            print(f"\n  Run: python scripts/retrain_model.py --check-first --no-tune")

    print(f"{'=' * 60}\n")

    return {
        "drift":  drift_report,
        "model":  model_report,
        "alerts": summary,
        "healthy": overall_ok,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SaaS Intelligence model monitoring",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python scripts/monitor_model.py
  python scripts/monitor_model.py --reset-baseline
  python scripts/monitor_model.py --report
  python scripts/monitor_model.py --features-csv data/processed/account_level_features.csv
        """
    )
    parser.add_argument(
        "--features-csv",
        default="data/processed/account_level_features.csv",
        help="Path to account_level_features.csv"
    )
    parser.add_argument(
        "--intelligence-csv",
        default="data/processed/account_intelligence.csv",
        help="Path to account_intelligence.csv"
    )
    parser.add_argument(
        "--reset-baseline",
        action="store_true",
        help="Reset drift baseline to current data before checking"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML + JSON report after monitoring"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🔍 SaaS Revenue Intelligence — Model Monitor")
    print("=" * 60)

    try:
        results = run_monitoring(
            features_csv=args.features_csv,
            intelligence_csv=args.intelligence_csv,
            reset_baseline=args.reset_baseline,
            generate_report=args.report,
        )
        sys.exit(0 if results["healthy"] else 1)

    except Exception as e:
        print(f"\n  ❌ Monitoring FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
