"""
Data Drift Detector
SaaS Revenue Intelligence System — Week 5

Detects statistical drift between a baseline snapshot and
current account-level features.

Drift metric: normalised mean shift
    shift = |mean_current - mean_baseline| / (|mean_baseline| + ε)

Usage:
    detector = DataDriftDetector()
    report   = detector.run(current_df)
    print(report["overall_status"])   # OK | WARNING | CRITICAL
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

logger = logging.getLogger(__name__)


def _load_monitoring_config(config_path: str = "config/monitoring_config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataDriftDetector:
    """
    Detects feature-level data drift versus a stored baseline snapshot.

    Steps:
        1. Load (or auto-create) baseline snapshot
        2. Compute per-feature mean shift
        3. Classify each feature: OK | WARNING | CRITICAL
        4. Produce an overall status + drift report dict
        5. Persist report to monitoring/drift_report.json
    """

    def __init__(self, config_path: str = "config/monitoring_config.yaml"):
        cfg = _load_monitoring_config(config_path)
        self.cfg      = cfg["monitoring"]
        self.alert_cfg = cfg.get("alerts", {})

        self.monitored_features: list = self.cfg["monitored_features"]
        self.warn_threshold: float    = self.cfg["drift_thresholds"]["warning"]
        self.crit_threshold: float    = self.cfg["drift_thresholds"]["critical"]
        self.baseline_path: Path      = Path(self.cfg["baseline"]["path"])
        self.drift_report_path: Path  = Path(self.cfg["output"]["drift_report"])
        self.auto_create_baseline: bool = self.cfg["baseline"]["auto_create"]

        logger.info("DataDriftDetector initialised.")
        logger.info(f"  Monitored features : {self.monitored_features}")
        logger.info(f"  Warn threshold     : {self.warn_threshold}")
        logger.info(f"  Critical threshold : {self.crit_threshold}")

    # ─────────────────────────────────────────────
    # BASELINE
    # ─────────────────────────────────────────────

    def create_baseline(self, df: pd.DataFrame) -> None:
        """Save current data as the drift baseline."""
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in self.monitored_features if c in df.columns]
        df[cols].to_csv(self.baseline_path, index=False)
        logger.info(f"✅ Baseline snapshot saved: {self.baseline_path} ({len(df)} accounts)")

    def load_baseline(self) -> Optional[pd.DataFrame]:
        """Load baseline snapshot from disk. Returns None if not found."""
        if not self.baseline_path.exists():
            logger.warning(f"⚠️  Baseline not found: {self.baseline_path}")
            return None
        df = pd.read_csv(self.baseline_path)
        logger.info(f"  Baseline loaded: {len(df)} accounts × {len(df.columns)} features")
        return df

    # ─────────────────────────────────────────────
    # DRIFT COMPUTATION
    # ─────────────────────────────────────────────

    def compute_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> Dict[str, Dict]:
        """
        Compute normalised mean shift per feature.

        Returns:
            {
              "total_mrr": {
                  "baseline_mean": 22000.0,
                  "current_mean":  24500.0,
                  "shift":         0.113,
                  "status":        "WARNING"
              }, ...
            }
        """
        results = {}

        for feature in self.monitored_features:
            if feature not in baseline_df.columns or feature not in current_df.columns:
                logger.debug(f"  Skipping {feature} — not in both dataframes")
                continue

            baseline_mean = float(baseline_df[feature].mean())
            current_mean  = float(current_df[feature].mean())

            # Normalised mean shift (ε=1e-9 guards zero baseline)
            shift = abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-9)

            if shift >= self.crit_threshold:
                status = "CRITICAL"
            elif shift >= self.warn_threshold:
                status = "WARNING"
            else:
                status = "OK"

            results[feature] = {
                "baseline_mean": round(baseline_mean, 4),
                "current_mean":  round(current_mean,  4),
                "shift":         round(shift, 4),
                "status":        status,
            }

            logger.info(
                f"  {feature:<30} shift={shift:.4f}  [{status}]"
                f"  baseline={baseline_mean:.2f}  current={current_mean:.2f}"
            )

        return results

    def _overall_status(self, feature_results: Dict) -> str:
        statuses = [v["status"] for v in feature_results.values()]
        if "CRITICAL" in statuses:
            return "CRITICAL"
        if "WARNING" in statuses:
            return "WARNING"
        return "OK"

    # ─────────────────────────────────────────────
    # MAIN RUN
    # ─────────────────────────────────────────────

    def run(self, current_df: pd.DataFrame) -> Dict:
        """
        Run full drift detection pipeline.

        Returns drift report dict and persists it to disk.
        """
        logger.info("=" * 60)
        logger.info("Running Data Drift Detection")
        logger.info("=" * 60)

        # Load or auto-create baseline
        baseline_df = self.load_baseline()
        if baseline_df is None:
            if self.auto_create_baseline:
                logger.info("  Auto-creating baseline from current data...")
                self.create_baseline(current_df)
                baseline_df = current_df.copy()
            else:
                raise FileNotFoundError(
                    f"Baseline not found at {self.baseline_path}. "
                    "Run create_baseline(df) first or set auto_create: true in config."
                )

        feature_results = self.compute_feature_drift(baseline_df, current_df)
        overall         = self._overall_status(feature_results)

        n_critical = sum(1 for v in feature_results.values() if v["status"] == "CRITICAL")
        n_warning  = sum(1 for v in feature_results.values() if v["status"] == "WARNING")
        n_ok       = sum(1 for v in feature_results.values() if v["status"] == "OK")

        report = {
            "run_timestamp":    datetime.now().isoformat(),
            "overall_status":   overall,
            "baseline_accounts": len(baseline_df),
            "current_accounts":  len(current_df),
            "features_checked":  len(feature_results),
            "n_critical":        n_critical,
            "n_warning":         n_warning,
            "n_ok":              n_ok,
            "feature_results":   feature_results,
        }

        # Persist
        self.drift_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.drift_report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"  Drift report saved: {self.drift_report_path}")

        logger.info(
            f"\n  OVERALL: {overall} | "
            f"CRITICAL: {n_critical} | WARNING: {n_warning} | OK: {n_ok}"
        )

        return report
