"""
Alert Manager
SaaS Revenue Intelligence System — Week 5

Centralised alert dispatcher. Reads drift + model health reports
and dispatches alerts to configured channels (console, file, email).

Usage:
    manager = AlertManager()
    manager.process_drift_report(drift_report)
    manager.process_model_report(model_report)
    manager.flush()   # write alert log to disk
"""

import sys
import json
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def _load_alert_config(config_path: str = "config/monitoring_config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class Alert:
    """Single alert entry."""

    def __init__(self, severity: str, source: str, message: str, detail: Optional[Dict] = None):
        self.severity  = severity.upper()   # INFO | WARNING | CRITICAL
        self.source    = source
        self.message   = message
        self.detail    = detail or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "severity":  self.severity,
            "source":    self.source,
            "message":   self.message,
            "detail":    self.detail,
        }

    def __str__(self) -> str:
        return f"[{self.severity}] {self.source}: {self.message}"


class AlertManager:
    """
    Processes monitoring reports and dispatches alerts.

    Channels:
        - console  : logs to stdout via logging
        - file     : appends to monitoring/alert_log.json
        - email    : sends via SMTP (disabled by default)
    """

    def __init__(self, config_path: str = "config/monitoring_config.yaml"):
        cfg = _load_alert_config(config_path)
        self.alert_cfg    = cfg.get("alerts", {})
        self.mon_cfg      = cfg.get("monitoring", {})

        self.enabled:          bool = self.alert_cfg.get("enabled", True)
        self.console_enabled:  bool = self.alert_cfg.get("channels", {}).get("console", True)
        self.file_enabled:     bool = self.alert_cfg.get("channels", {}).get("file", True)
        self.email_enabled:    bool = self.alert_cfg.get("channels", {}).get("email", False)
        self.smtp_cfg:         Dict = self.alert_cfg.get("smtp", {})
        self.alert_log_path:   Path = Path(self.mon_cfg.get("output", {}).get("alert_log", "monitoring/alert_log.json"))

        self._alerts: List[Alert] = []
        logger.info("AlertManager initialised.")

    # ─────────────────────────────────────────────
    # ALERT CREATION
    # ─────────────────────────────────────────────

    def add_alert(self, severity: str, source: str, message: str, detail: Optional[Dict] = None) -> Alert:
        alert = Alert(severity=severity, source=source, message=message, detail=detail)
        self._alerts.append(alert)
        self._dispatch(alert)
        return alert

    # ─────────────────────────────────────────────
    # REPORT PROCESSORS
    # ─────────────────────────────────────────────

    def process_drift_report(self, report: Dict) -> None:
        """Read a drift report dict and raise alerts for drifted features."""
        overall = report.get("overall_status", "OK")

        if overall == "CRITICAL":
            self.add_alert(
                severity="CRITICAL",
                source="DataDriftDetector",
                message=f"CRITICAL data drift detected — {report.get('n_critical', 0)} feature(s) critical",
                detail={"overall_status": overall, "n_critical": report.get("n_critical"), "n_warning": report.get("n_warning")},
            )
        elif overall == "WARNING":
            self.add_alert(
                severity="WARNING",
                source="DataDriftDetector",
                message=f"Data drift warning — {report.get('n_warning', 0)} feature(s) drifting",
                detail={"overall_status": overall, "n_warning": report.get("n_warning")},
            )
        else:
            self.add_alert(
                severity="INFO",
                source="DataDriftDetector",
                message="No significant data drift detected.",
                detail={"overall_status": overall},
            )

        # Per-feature CRITICAL alerts
        for feature, result in report.get("feature_results", {}).items():
            if result["status"] == "CRITICAL":
                self.add_alert(
                    severity="CRITICAL",
                    source="DataDriftDetector",
                    message=f"Feature '{feature}' drift={result['shift']:.4f} (threshold={self.mon_cfg.get('drift_thresholds', {}).get('critical', 0.10)})",
                    detail=result,
                )

    def process_model_report(self, report: Dict) -> None:
        """Read a model health report dict and raise alerts for failing checks."""
        recommendation = report.get("recommendation", "OK")

        if recommendation == "RETRAIN_REQUIRED":
            self.add_alert(
                severity="CRITICAL",
                source="ModelMonitor",
                message="Model retraining REQUIRED — critical health checks failing",
                detail={"recommendation": recommendation},
            )
        elif recommendation == "RETRAIN_RECOMMENDED":
            self.add_alert(
                severity="WARNING",
                source="ModelMonitor",
                message="Model retraining RECOMMENDED — multiple health warnings",
                detail={"recommendation": recommendation},
            )
        else:
            self.add_alert(
                severity="INFO",
                source="ModelMonitor",
                message="Model health is OK — no retraining needed",
                detail={"recommendation": recommendation},
            )

        # Per-check alerts
        for check_name, result in report.get("checks", {}).items():
            status = result.get("status", "OK")
            if status == "CRITICAL":
                self.add_alert(
                    severity="CRITICAL",
                    source=f"ModelMonitor.{check_name}",
                    message=result.get("reason", f"Check '{check_name}' failed"),
                    detail=result,
                )
            elif status == "WARNING":
                self.add_alert(
                    severity="WARNING",
                    source=f"ModelMonitor.{check_name}",
                    message=result.get("reason", f"Check '{check_name}' warning"),
                    detail=result,
                )

    # ─────────────────────────────────────────────
    # DISPATCH
    # ─────────────────────────────────────────────

    def _dispatch(self, alert: Alert) -> None:
        if not self.enabled:
            return
        if self.console_enabled:
            self._log_to_console(alert)
        if self.email_enabled and alert.severity == "CRITICAL":
            self._send_email(alert)

    def _log_to_console(self, alert: Alert) -> None:
        msg = str(alert)
        if alert.severity == "CRITICAL":
            logger.critical(msg)
        elif alert.severity == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)

    def _send_email(self, alert: Alert) -> None:
        try:
            subject = f"[{alert.severity}] SaaS Intelligence Alert — {alert.source}"
            body    = f"{alert.message}\n\nDetail:\n{json.dumps(alert.detail, indent=2)}"
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = self.smtp_cfg.get("from_addr", "alerts@saas-intelligence.com")
            msg["To"]      = ", ".join(self.smtp_cfg.get("to_addrs", []))

            with smtplib.SMTP(self.smtp_cfg["host"], self.smtp_cfg["port"]) as server:
                server.starttls()
                server.login(self.smtp_cfg["username"], self.smtp_cfg["password"])
                server.sendmail(msg["From"], self.smtp_cfg["to_addrs"], msg.as_string())

            logger.info(f"  📧 Email alert sent: {alert.severity}")
        except Exception as e:
            logger.warning(f"  ⚠️  Email send failed: {e}")

    # ─────────────────────────────────────────────
    # FLUSH — WRITE ALERT LOG
    # ─────────────────────────────────────────────

    def flush(self) -> None:
        """Write all alerts collected this session to the alert log file."""
        if not self.file_enabled:
            return

        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing log
        existing: List[Dict] = []
        if self.alert_log_path.exists():
            try:
                with open(self.alert_log_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        # Append new alerts
        new_entries = [a.to_dict() for a in self._alerts]
        combined    = existing + new_entries

        with open(self.alert_log_path, "w") as f:
            json.dump(combined, f, indent=2)

        logger.info(f"  Alert log updated: {self.alert_log_path} ({len(new_entries)} new alerts)")
        self._alerts.clear()

    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────

    def summary(self) -> Dict:
        """Return count of alerts by severity in this session."""
        counts = {"INFO": 0, "WARNING": 0, "CRITICAL": 0}
        for a in self._alerts:
            counts[a.severity] = counts.get(a.severity, 0) + 1
        return counts
