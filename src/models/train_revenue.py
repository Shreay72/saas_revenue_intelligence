"""
Revenue Model Training
SaaS Revenue Intelligence System - Week 3 (Revised)

Key changes:
    - Removed total_arr from MRR features (no trivial ARR/12 leakage)
    - Uses actual churn model probabilities (no hard-coded mapping)
    - CLV is deterministic formula-based
    - Uses engagement/support/revenue state/signal features
    - Stability check across seeds + confidence intervals
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.metrics import calculate_revenue_at_risk
from src.models.clv_model import CLVModel
from src.pipelines.churn_pipeline import ChurnPredictionPipeline  # Week 2 churn pipeline

logger = get_logger(__name__)

MRR_FEATURES = [
    "seats",
    "tenure_months",
    "engagement_score",
    "support_risk_score",
    "auto_renew_ratio",
    "upgrade_count",
    "downgrade_count",
    "ticket_count",
    "avg_satisfaction_score",
    "error_rate",
    "revenue_per_seat",
    "unique_features_used",
    "total_usage",
    "escalation_ratio",
    "engagement_state",
    "revenue_change_signal",
    "support_pressure_signal",
]


class RevenueModelTrainer:
    """
    Trains MRR forecasting model and initializes deterministic CLV model.
    """

    def __init__(self, output_dir: str = "models/revenue"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_model = None
        self.best_model_name = None
        self.feature_cols = None
        logger.info("RevenueModelTrainer initialized.")

    # ─────────────────────────────────────────────
    # DATA LOADING + CHURN PROBABILITIES
    # ─────────────────────────────────────────────

    def load_data(self, data_path: str = "data/processed/account_level_features.csv") -> pd.DataFrame:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded: {df.shape}")
        logger.info(
            f"MRR Stats — Min: ${df['total_mrr'].min():,.2f} | "
            f"Max: ${df['total_mrr'].max():,.2f} | "
            f"Mean: ${df['total_mrr'].mean():,.2f}"
        )

        # Attach churn_probability from churn model if missing
        if "churn_probability" not in df.columns:
            logger.info("Scoring churn probabilities via Week 2 churn pipeline...")
            churn_pipeline = ChurnPredictionPipeline(model_dir="models/churn")
            churn_probs = churn_pipeline.predict_proba(df)
            df["churn_probability"] = churn_probs
            logger.info("churn_probability column added from churn model.")

        # Ensure state/signal features exist (backwards compatibility)
        for col in ["engagement_state", "revenue_change_signal", "support_pressure_signal"]:
            if col not in df.columns:
                df[col] = 0
                logger.info(f"  ⚠️  Added missing state column: {col} = 0")

        return df

    # ─────────────────────────────────────────────
    # MRR MODEL TRAINING
    # ─────────────────────────────────────────────

    def train_mrr_model(self, df: pd.DataFrame) -> Dict:
        logger.info("=" * 60)
        logger.info("Training MRR Prediction Model")
        logger.info("=" * 60)

        self.feature_cols = [c for c in MRR_FEATURES if c in df.columns]
        logger.info(f"Features: {len(self.feature_cols)} columns")

        X = df[self.feature_cols].fillna(0)
        y = df["total_mrr"]

        models = {
            "GradientBoosting": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=100,
                            max_depth=3,
                            learning_rate=0.1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "RandomForest": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=100,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "Ridge": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        }

        results = {}
        for name, model in models.items():
            cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2")
            cv_mae = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
            results[name] = {
                "model": model,
                "r2_mean": cv_r2.mean(),
                "r2_std": cv_r2.std(),
                "mae_mean": -cv_mae.mean(),
                "mae_std": cv_mae.std(),
            }
            logger.info(
                f"  {name}: R²={cv_r2.mean():.4f} ± {cv_r2.std():.4f} | "
                f"MAE=${-cv_mae.mean():,.2f} ± ${cv_mae.std():,.2f}"
            )

        best_name = max(results, key=lambda k: results[k]["r2_mean"])
        best_info = results[best_name]
        best_model = best_info["model"]
        logger.info(
            f"\n🏆 Best MRR Model: {best_name} "
            f"(R²={best_info['r2_mean']:.4f} ± {best_info['r2_std']:.4f})"
        )

        # Stability check across seeds
        logger.info("\n🔬 Stability Check (3 seeds: 42, 7, 99)...")
        stability_r2 = []
        kf = KFold(n_splits=5, shuffle=True)

        for seed in [42, 7, 99]:
            kf.random_state = seed
            seed_scores = cross_val_score(best_model, X, y, cv=kf, scoring="r2")
            stability_r2.append(seed_scores.mean())
            logger.info(
                f"  Seed {seed:>2}: R²={seed_scores.mean():.4f} ± {seed_scores.std():.4f}"
            )

        stability_mean = float(np.mean(stability_r2))
        stability_std = float(np.std(stability_r2))
        stability_flag = (
            "✅ STABLE" if stability_std < 0.05 else "⚠️  UNSTABLE (high seed variance)"
        )
        logger.info(
            f"  Cross-seed R²: {stability_mean:.4f} ± {stability_std:.4f}  → {stability_flag}"
        )

        best_model.fit(X, y)
        self.best_model = best_model
        self.best_model_name = best_name

        return {
            "model_name": best_name,
            "r2_mean": round(best_info["r2_mean"], 4),
            "r2_std": round(best_info["r2_std"], 4),
            "mae_mean": round(best_info["mae_mean"], 2),
            "mae_std": round(best_info["mae_std"], 2),
            "stability_r2_mean": round(stability_mean, 4),
            "stability_r2_std": round(stability_std, 4),
            "stability_flag": stability_flag,
            "feature_count": len(self.feature_cols),
        }

    # ─────────────────────────────────────────────
    # CLV — Deterministic
    # ─────────────────────────────────────────────

    def train_clv_model(self, df: pd.DataFrame) -> Tuple[CLVModel, Dict]:
        logger.info("=" * 60)
        logger.info("CLV Model — Deterministic (Formula-Based)")
        logger.info("=" * 60)

        clv_model = CLVModel(output_dir=str(self.output_dir))
        metrics = clv_model.fit(df)
        return clv_model, metrics

    # ─────────────────────────────────────────────
    # REVENUE AT RISK
    # ─────────────────────────────────────────────

    def calculate_revenue_at_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating Revenue At Risk (Expected Loss Formula)...")

        result = df[["account_id", "account_name", "total_mrr", "churn_probability"]].copy()

        result["revenue_at_risk"] = result.apply(
            lambda r: calculate_revenue_at_risk(r["total_mrr"], r["churn_probability"]),
            axis=1,
        )

        total_at_risk = result["revenue_at_risk"].sum()
        logger.info(f"Total Monthly Revenue At Risk: ${total_at_risk:,.2f}")
        logger.info("  (Formula: MRR × churn_probability — expected value)")

        return result.sort_values("revenue_at_risk", ascending=False)

    # ─────────────────────────────────────────────
    # SAVE ARTIFACTS
    # ─────────────────────────────────────────────

    def save_artifacts(self, mrr_metrics: Dict, clv_metrics: Dict) -> None:
        model_path = self.output_dir / "revenue_model_v1.pkl"
        joblib.dump(
            {"model": self.best_model, "feature_cols": self.feature_cols},
            model_path,
        )
        logger.info(f"✅ MRR model saved: {model_path}")

        metadata = {
            "model_name": "MRR Prediction Model",
            "model_type": self.best_model_name,
            "train_date": datetime.now().isoformat(),
            "feature_cols": self.feature_cols,
            "mrr_metrics": mrr_metrics,
            "clv_metrics": clv_metrics,
            "version": "v3",
            "changes": [
                "Removed total_arr from features to avoid trivial leakage",
                "Attached churn_probability from churn model (no hard-coded mapping)",
                "Renamed pseudo-trend features to *_state / *_signal",
                "Revenue at risk uses expected loss formula",
                "Stability check across 3 seeds + confidence intervals",
            ],
        }
        meta_path = self.output_dir / "revenue_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Revenue metadata saved: {meta_path}")

    # ─────────────────────────────────────────────
    # FULL TRAINING PIPELINE
    # ─────────────────────────────────────────────

    def run(self, data_path: str = "data/processed/account_level_features.csv"):
        df = self.load_data(data_path)

        mrr_metrics = self.train_mrr_model(df)
        clv_model, clv_metrics = self.train_clv_model(df)

        self.save_artifacts(mrr_metrics, clv_metrics)
        clv_model.save(clv_metrics)

        logger.info("\n✅ All revenue models trained successfully!")
        return mrr_metrics, clv_metrics
