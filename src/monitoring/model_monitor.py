"""
Model Health Monitor
SaaS Revenue Intelligence System — Week 5

Checks model health using:
    - Churn rate sanity check (suspicious if out of expected range)
    - AUC proxy via churn_probability calibration
    - Portfolio metrics vs baseline
    - Retrain recommendation trigger

Usage:
    monitor = ModelMonitor()
    report  = monitor.run(current_df)
    print(report["recommendation"])   # OK | RETRAIN_RECOMMENDED | RETRAIN_REQUIRED
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.metrics import calculate_portfolio_metrics

logger = logging.getLogger(__name__)


def _load_monitoring_config(config_path: str = "config/monitoring_config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ModelMonitor:
    """
    Monitors model health and portfolio-level metric shifts.

    Checks:
        1. Churn rate sanity (expected range: 5%–95%)
        2. Churn probability calibration (mean prob vs actual churn rate)
        3. AUC estimate via separation of churn prob distributions
        4. Portfolio metric deltas vs baseline snapshot
        5. Issues retrain recommendation
    """

    def __init__(self, config_path: str = "config/monitoring_config.yaml"):
        cfg = _load_monitoring_config(config_path)
        self.cfg        = cfg["monitoring"]
        health_cfg      = self.cfg["model_health"]

        self.min_auc:          float = health_cfg["min_auc"]
        self.max_auc_drop:     float = health_cfg["max_auc_drop"]
        self.min_accounts:     int   = health_cfg["min_accounts"]
        self.churn_rate_min:   float = health_cfg["churn_rate_min"]
        self.churn_rate_max:   float = health_cfg["churn_rate_max"]

        self.baseline_path:    Path  = Path(self.cfg["baseline"]["path"])
        self.report_path:      Path  = Path(self.cfg["output"]["model_report"])

        logger.info("ModelMonitor initialised.")

    # ─────────────────────────────────────────────
    # CHECKS
    # ─────────────────────────────────────────────

    def _check_churn_rate(self, df: pd.DataFrame) -> Dict:
        """Verify churn rate is within expected bounds."""
        if "churn_flag" not in df.columns:
            return {"status": "SKIP", "reason": "churn_flag column missing"}

        rate = float(df["churn_flag"].mean())
        if rate < self.churn_rate_min:
            status = "WARNING"
            reason = f"Churn rate {rate:.1%} below minimum {self.churn_rate_min:.1%}"
        elif rate > self.churn_rate_max:
            status = "WARNING"
            reason = f"Churn rate {rate:.1%} above maximum {self.churn_rate_max:.1%}"
        else:
            status = "OK"
            reason = f"Churn rate {rate:.1%} within expected range"

        return {"status": status, "churn_rate": round(rate, 4), "reason": reason}

    def _check_probability_calibration(self, df: pd.DataFrame) -> Dict:
        """
        Check if churn_probability mean aligns with actual churn_flag rate.
        A well-calibrated model should have mean(prob) ≈ mean(churn_flag).
        """
        if "churn_probability" not in df.columns or "churn_flag" not in df.columns:
            return {"status": "SKIP", "reason": "Required columns missing"}

        mean_prob  = float(df["churn_probability"].mean())
        actual_rate = float(df["churn_flag"].mean())
        calibration_error = abs(mean_prob - actual_rate)

        if calibration_error > 0.20:
            status = "CRITICAL"
        elif calibration_error > 0.10:
            status = "WARNING"
        else:
            status = "OK"

        return {
            "status":             status,
            "mean_predicted_prob": round(mean_prob, 4),
            "actual_churn_rate":   round(actual_rate, 4),
            "calibration_error":   round(calibration_error, 4),
        }

    def _estimate_auc(self, df: pd.DataFrame) -> Dict:
        """
        Estimate AUC via Mann-Whitney U statistic.
        AUC = P(score_positive > score_negative)

        Requires churn_probability + churn_flag columns.
        """
        if "churn_probability" not in df.columns or "churn_flag" not in df.columns:
            return {"status": "SKIP", "reason": "Required columns missing"}

        churned     = df[df["churn_flag"] == 1]["churn_probability"].values
        not_churned = df[df["churn_flag"] == 0]["churn_probability"].values

        if len(churned) == 0 or len(not_churned) == 0:
            return {"status": "SKIP", "reason": "Insufficient class samples"}

        # Mann-Whitney AUC
        n_pos = len(churned)
        n_neg = len(not_churned)
        concordant = sum(
            1 for p in churned for n in not_churned if p > n
        ) + 0.5 * sum(
            1 for p in churned for n in not_churned if p == n
        )
        auc = concordant / (n_pos * n_neg)

        if auc < self.min_auc:
            status = "CRITICAL"
            reason = f"AUC {auc:.4f} below minimum {self.min_auc}"
        elif auc < self.min_auc + 0.05:
            status = "WARNING"
            reason = f"AUC {auc:.4f} near minimum threshold"
        else:
            status = "OK"
            reason = f"AUC {auc:.4f} is healthy"

        return {
            "status":    status,
            "auc":       round(auc, 4),
            "n_churned": n_pos,
            "n_active":  n_neg,
            "reason":    reason,
        }

    def _check_portfolio_drift(self, df: pd.DataFrame) -> Dict:
        """Compare current portfolio metrics vs baseline snapshot."""
        if not self.baseline_path.exists():
            return {"status": "SKIP", "reason": "No baseline snapshot available"}

        baseline_df = pd.read_csv(self.baseline_path)
        current_metrics  = calculate_portfolio_metrics(df)
        baseline_metrics = calculate_portfolio_metrics(baseline_df)

        deltas = {}
        for key in ["total_mrr", "churn_rate", "avg_mrr"]:
            if key in current_metrics and key in baseline_metrics:
                base_val = baseline_metrics[key]
                curr_val = current_metrics[key]
                delta_pct = (curr_val - base_val) / (abs(base_val) + 1e-9)
                deltas[key] = {
                    "baseline": round(base_val, 4),
                    "current":  round(curr_val, 4),
                    "delta_pct": round(delta_pct, 4),
                }

        return {
            "status":          "OK",
            "current_metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
            "deltas":           deltas,
        }

    # ─────────────────────────────────────────────
    # RECOMMENDATION
    # ─────────────────────────────────────────────

    def _make_recommendation(self, checks: Dict) -> str:
        statuses = [v.get("status", "OK") for v in checks.values() if v.get("status") != "SKIP"]
        if statuses.count("CRITICAL") >= 1:
            return "RETRAIN_REQUIRED"
        if statuses.count("WARNING") >= 2:
            return "RETRAIN_RECOMMENDED"
        return "OK"

    # ─────────────────────────────────────────────
    # MAIN RUN
    # ─────────────────────────────────────────────

    def run(self, current_df: pd.DataFrame) -> Dict:
        """
        Run full model health monitoring pipeline.

        Returns health report dict and persists it to disk.
        """
        logger.info("=" * 60)
        logger.info("Running Model Health Monitor")
        logger.info("=" * 60)
        logger.info(f"  Accounts: {len(current_df)}")

        if len(current_df) < self.min_accounts:
            logger.warning(f"  ⚠️  Only {len(current_df)} accounts — below minimum {self.min_accounts}")

        checks = {
            "churn_rate":      self._check_churn_rate(current_df),
            "calibration":     self._check_probability_calibration(current_df),
            "auc_estimate":    self._estimate_auc(current_df),
            "portfolio_drift": self._check_portfolio_drift(current_df),
        }

        recommendation = self._make_recommendation(checks)

        for name, result in checks.items():
            status = result.get("status", "?")
            logger.info(f"  {name:<25} → {status}")

        logger.info(f"\n  RECOMMENDATION: {recommendation}")

        report = {
            "run_timestamp":  datetime.now().isoformat(),
            "account_count":  len(current_df),
            "checks":         checks,
            "recommendation": recommendation,
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"  Model health report saved: {self.report_path}")

        return report
