"""
Generate Report Script
SaaS Revenue Intelligence System — Week 5

Generates a full HTML + JSON portfolio intelligence report.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output-dir reports/custom
    python scripts/generate_report.py --run-monitoring   # run drift check first
    python scripts/generate_report.py --format json      # JSON only
    python scripts/generate_report.py --format html      # HTML only
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_intelligence() -> pd.DataFrame:
    path = Path("data/processed/account_intelligence.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Intelligence CSV not found: {path}\n"
            "Run: python scripts/train_all.py --skip-week1 --no-tune"
        )
    df = pd.read_csv(path)
    logger.info(f"  Intelligence loaded: {len(df)} accounts × {df.shape[1]} columns")
    return df


def load_monitoring_reports() -> dict:
    reports = {}
    for name, path in [
        ("drift",  "monitoring/drift_report.json"),
        ("model",  "monitoring/model_health_report.json"),
        ("alerts", "monitoring/alert_log.json"),
    ]:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                reports[name] = json.load(f)
        else:
            reports[name] = None
    return reports


# ─────────────────────────────────────────────────────────────────────────────
# JSON REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_json_report(df: pd.DataFrame, monitoring: dict) -> dict:
    """Build a structured JSON portfolio report."""

    def _col(col):
        return df[col].sum() if col in df.columns else 0

    tier_dist = df["risk_tier"].value_counts().to_dict() if "risk_tier" in df.columns else {}
    action_dist = df["recommended_action"].value_counts().to_dict() if "recommended_action" in df.columns else {}

    top20 = df.nlargest(20, "priority_score") if "priority_score" in df.columns else df.head(20)
    top20_cols = [c for c in [
        "account_id", "account_name", "total_mrr",
        "risk_tier", "risk_score", "churn_probability",
        "revenue_at_risk", "expected_recovery",
        "recommended_action", "urgency", "priority_score",
    ] if c in top20.columns]

    report = {
        "generated_at":   datetime.now().isoformat(),
        "total_accounts": len(df),
        "portfolio": {
            "total_mrr":           round(float(_col("total_mrr")), 2),
            "total_revenue_at_risk": round(float(_col("revenue_at_risk")), 2),
            "total_recoverable":    round(float(_col("expected_recovery")), 2),
            "avg_health_score":     round(float(df["health_score"].mean()), 2) if "health_score" in df.columns else None,
            "avg_churn_probability": round(float(df["churn_probability"].mean()), 4) if "churn_probability" in df.columns else None,
        },
        "risk_distribution":   tier_dist,
        "action_distribution": action_dist,
        "top_20_accounts":     top20[top20_cols].to_dict("records"),
        "monitoring": {
            "drift_status":     monitoring["drift"]["overall_status"] if monitoring.get("drift") else "NOT_RUN",
            "model_health":     monitoring["model"]["recommendation"] if monitoring.get("model") else "NOT_RUN",
            "total_alerts":     len(monitoring["alerts"]) if monitoring.get("alerts") else 0,
        },
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_html_report(report: dict, df: pd.DataFrame) -> str:
    """Build a self-contained HTML intelligence report."""
    ts          = report["generated_at"]
    total       = report["total_accounts"]
    p           = report["portfolio"]
    tier_dist   = report["risk_distribution"]
    top20       = report["top_20_accounts"]

    tier_rows = "".join(
        f"<tr><td><b>{tier}</b></td><td>{count}</td>"
        f"<td>{count / total * 100:.1f}%</td></tr>"
        for tier, count in tier_dist.items()
    )

    account_rows = ""
    for acc in top20:
        tier  = acc.get("risk_tier", "")
        color = {"CRITICAL": "#ff4444", "HIGH": "#ff8800", "MEDIUM": "#ffcc00", "LOW": "#44bb44"}.get(tier, "#999")
        account_rows += (
            f"<tr>"
            f"<td>{acc.get('account_name', acc.get('account_id', ''))}</td>"
            f"<td><span style='color:{color};font-weight:bold'>{tier}</span></td>"
            f"<td>${acc.get('total_mrr', 0):,.0f}</td>"
            f"<td>{acc.get('churn_probability', 0):.2%}</td>"
            f"<td>${acc.get('revenue_at_risk', 0):,.0f}</td>"
            f"<td>${acc.get('expected_recovery', 0):,.0f}</td>"
            f"<td>{acc.get('recommended_action', '')}</td>"
            f"<td>{acc.get('urgency', '')}</td>"
            f"</tr>"
        )

    drift_status  = report["monitoring"]["drift_status"]
    model_health  = report["monitoring"]["model_health"]
    drift_color   = {"OK": "green", "WARNING": "orange", "CRITICAL": "red"}.get(drift_status, "grey")
    model_color   = {"OK": "green", "RETRAIN_RECOMMENDED": "orange", "RETRAIN_REQUIRED": "red"}.get(model_health, "grey")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SaaS Revenue Intelligence Report — {ts[:10]}</title>
  <style>
    body   {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
              margin: 40px; color: #222; background: #f9f9f9; }}
    h1     {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 8px; }}
    h2     {{ color: #16213e; margin-top: 32px; }}
    .kpi   {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
    .card  {{ background: white; border-radius: 8px; padding: 20px 28px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); min-width: 180px; }}
    .card .val {{ font-size: 2em; font-weight: bold; color: #e94560; }}
    .card .lbl {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
    table  {{ width: 100%; border-collapse: collapse; background: white;
              border-radius: 8px; overflow: hidden;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    th     {{ background: #16213e; color: white; padding: 10px 14px; text-align: left; }}
    td     {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f0f4ff; }}
    .badge {{ padding: 2px 10px; border-radius: 12px; font-size: 0.8em;
              font-weight: bold; color: white; }}
    footer {{ margin-top: 40px; font-size: 0.8em; color: #aaa; text-align: center; }}
  </style>
</head>
<body>
  <h1>📊 SaaS Revenue Intelligence Report</h1>
  <p>Generated: <b>{ts}</b> &nbsp;|&nbsp; Total Accounts: <b>{total}</b></p>

  <h2>Portfolio Overview</h2>
  <div class="kpi">
    <div class="card"><div class="val">${p['total_mrr']:,.0f}</div><div class="lbl">Total MRR</div></div>
    <div class="card"><div class="val">${p['total_revenue_at_risk']:,.0f}</div><div class="lbl">Revenue at Risk</div></div>
    <div class="card"><div class="val">${p['total_recoverable']:,.0f}</div><div class="lbl">Recoverable</div></div>
    <div class="card"><div class="val">{p.get('avg_health_score', 0):.1f}</div><div class="lbl">Avg Health Score</div></div>
    <div class="card"><div class="val">{(p.get('avg_churn_probability') or 0):.1%}</div><div class="lbl">Avg Churn Probability</div></div>
  </div>

  <h2>Risk Distribution</h2>
  <table style="max-width:400px">
    <thead><tr><th>Tier</th><th>Accounts</th><th>%</th></tr></thead>
    <tbody>{tier_rows}</tbody>
  </table>

  <h2>Monitoring Status</h2>
  <div class="kpi">
    <div class="card">
      <div class="val" style="color:{drift_color}">{drift_status}</div>
      <div class="lbl">Data Drift</div>
    </div>
    <div class="card">
      <div class="val" style="color:{model_color}">{model_health}</div>
      <div class="lbl">Model Health</div>
    </div>
    <div class="card">
      <div class="val">{report['monitoring']['total_alerts']}</div>
      <div class="lbl">Total Alerts</div>
    </div>
  </div>

  <h2>Top 20 Accounts by Priority</h2>
  <table>
    <thead>
      <tr>
        <th>Account</th><th>Risk Tier</th><th>MRR</th>
        <th>Churn Prob</th><th>Revenue at Risk</th>
        <th>Expected Recovery</th><th>Action</th><th>Urgency</th>
      </tr>
    </thead>
    <tbody>{account_rows}</tbody>
  </table>

  <footer>SaaS Revenue Intelligence System &copy; {ts[:4]}</footer>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate portfolio intelligence report")
    parser.add_argument("--output-dir",     default="reports",   help="Output directory")
    parser.add_argument("--run-monitoring", action="store_true", help="Run drift detection first")
    parser.add_argument("--format",         default="both",      choices=["json", "html", "both"])
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  📄 SaaS Revenue Intelligence — Report Generator")
    print("=" * 60)

    # ── Optional: run monitoring first ──────────────────────────────
    if args.run_monitoring:
        print("\n  Running monitoring checks first...")
        from src.monitoring.data_drift_detector import DataDriftDetector
        from src.monitoring.model_monitor import ModelMonitor
        from src.monitoring.alert_manager import AlertManager

        df_features = pd.read_csv("data/processed/account_level_features.csv")
        df_intel    = pd.read_csv("data/processed/account_intelligence.csv")

        drift_report = DataDriftDetector().run(df_intel)
        model_report = ModelMonitor().run(df_features)

        manager = AlertManager()
        manager.process_drift_report(drift_report)
        manager.process_model_report(model_report)
        manager.flush()

    # ── Load data ────────────────────────────────────────────────────
    df         = load_intelligence()
    monitoring = load_monitoring_reports()

    # ── Build report ─────────────────────────────────────────────────
    report = build_json_report(df, monitoring)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.format in ("json", "both"):
        json_path = output_dir / f"portfolio_report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  ✅ JSON report: {json_path}")

    if args.format in ("html", "both"):
        html      = build_html_report(report, df)
        html_path = output_dir / f"portfolio_report_{timestamp}.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"  ✅ HTML report: {html_path}")

    print(f"\n  Portfolio Summary:")
    print(f"    Total Accounts    : {report['total_accounts']}")
    print(f"    Total MRR         : ${report['portfolio']['total_mrr']:,.2f}")
    print(f"    Revenue at Risk   : ${report['portfolio']['total_revenue_at_risk']:,.2f}")
    print(f"    Recoverable       : ${report['portfolio']['total_recoverable']:,.2f}")
    print(f"    Drift Status      : {report['monitoring']['drift_status']}")
    print(f"    Model Health      : {report['monitoring']['model_health']}")
    print()


if __name__ == "__main__":
    main()
