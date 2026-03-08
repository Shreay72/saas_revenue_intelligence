"""
Retrain Model Script
SaaS Revenue Intelligence System — Week 5

Triggers a full model retrain when:
    - Called manually
    - Drift or model health reports recommend it

Usage:
    python scripts/retrain_model.py                  # always retrain
    python scripts/retrain_model.py --check-first    # only retrain if recommended
    python scripts/retrain_model.py --no-tune        # skip Optuna
    python scripts/retrain_model.py --churn-only     # retrain churn model only
    python scripts/retrain_model.py --revenue-only   # retrain revenue model only
"""

import sys
import time
import json
import shutil
import argparse
import traceback
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BACKUP
# ─────────────────────────────────────────────────────────────────────────────

def backup_existing_models() -> Path:
    """Back up current models/ before overwriting."""
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir  = Path(f"models/backups/retrain_{timestamp}")
    models_dir  = Path("models")

    if not models_dir.exists():
        logger.info("  No existing models to back up.")
        return backup_dir

    backup_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["churn", "revenue"]:
        src = models_dir / subdir
        if src.exists():
            dst = backup_dir / subdir
            shutil.copytree(src, dst)
            logger.info(f"  Backed up: {src} → {dst}")

    logger.info(f"  ✅ Backup complete: {backup_dir}")
    return backup_dir


# ─────────────────────────────────────────────────────────────────────────────
# CHECK — should we retrain?
# ─────────────────────────────────────────────────────────────────────────────

def should_retrain() -> tuple[bool, str]:
    """
    Read latest drift + model health reports.
    Returns (True, reason) if retraining is recommended, else (False, reason).
    """
    drift_path  = Path("monitoring/drift_report.json")
    model_path  = Path("monitoring/model_health_report.json")

    reasons = []

    if drift_path.exists():
        with open(drift_path) as f:
            drift = json.load(f)
        if drift.get("overall_status") in ("CRITICAL", "WARNING"):
            reasons.append(f"Data drift detected: {drift['overall_status']}")
    else:
        logger.warning("  No drift report found — run monitoring first.")

    if model_path.exists():
        with open(model_path) as f:
            model = json.load(f)
        rec = model.get("recommendation", "OK")
        if rec in ("RETRAIN_REQUIRED", "RETRAIN_RECOMMENDED"):
            reasons.append(f"Model health: {rec}")
    else:
        logger.warning("  No model health report found — run monitoring first.")

    if reasons:
        return True, " | ".join(reasons)
    return False, "No retrain triggers found — models are healthy."


# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN STEPS
# ─────────────────────────────────────────────────────────────────────────────

def retrain_churn(tune: bool) -> None:
    """Retrain churn model from latest account_level_features.csv."""
    import pandas as pd
    from src.models.train_churn import ChurnModelTrainer

    logger.info("  Retraining churn model...")
    trainer = ChurnModelTrainer(
        tune_hyperparameters=tune,
        n_trials=50 if tune else 5,
        cost_fp=100.0,
        cost_fn=5000.0,
    )
    trainer.load_data(data_path="data/processed/account_level_features.csv")
    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_xgboost()
    comparison = trainer.compare_models()
    trainer.save_best_model(output_dir="models/churn")

    best = comparison.iloc[0]
    logger.info(f"  ✅ Churn model retrained — Best: {best['Model']} (AUC={best['ROC-AUC']:.4f})")


def retrain_revenue() -> None:
    """Retrain revenue + CLV models."""
    from src.models.train_revenue import RevenueModelTrainer

    logger.info("  Retraining revenue model...")
    trainer = RevenueModelTrainer(output_dir="models/revenue")
    mrr_metrics, clv_metrics = trainer.run(
        data_path="data/processed/account_level_features.csv"
    )
    logger.info(f"  ✅ Revenue model retrained — R²={mrr_metrics['r2_mean']:.4f}")


def regenerate_intelligence() -> None:
    """Re-run Week 4 intelligence generation after retrain."""
    import pandas as pd
    from src.pipelines.churn_pipeline import load_pipeline
    from src.pipelines.revenue_pipeline import load_revenue_pipeline
    from src.recommendation.recommendation_engine import RecommendationEngine

    logger.info("  Regenerating account intelligence...")
    df = pd.read_csv("data/processed/account_level_features.csv")

    churn_pipeline = load_pipeline()
    df["churn_probability"] = churn_pipeline.predict_proba(df)

    rev_pipeline = load_revenue_pipeline()
    df["revenue_at_risk"] = rev_pipeline.calculate_revenue_at_risk(df)
    df["health_score"]    = rev_pipeline.calculate_health_scores(df)
    df["clv"]             = rev_pipeline.predict_clv(df)

    rec_engine = RecommendationEngine()
    intelligence_df = rec_engine.generate_intelligence(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    intelligence_df.to_csv("data/processed/account_intelligence.csv", index=False)
    logger.info(f"  ✅ Intelligence regenerated: {len(intelligence_df)} accounts")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrain SaaS Intelligence models")
    parser.add_argument("--check-first",   action="store_true", help="Only retrain if monitoring recommends it")
    parser.add_argument("--no-tune",       action="store_true", help="Skip Optuna hyperparameter tuning")
    parser.add_argument("--churn-only",    action="store_true", help="Only retrain churn model")
    parser.add_argument("--revenue-only",  action="store_true", help="Only retrain revenue model")
    parser.add_argument("--no-backup",     action="store_true", help="Skip model backup")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🔄 SaaS Revenue Intelligence — Model Retrain")
    print("=" * 60)

    # ── Check first? ──────────────────────────────────────────────────
    if args.check_first:
        needs_retrain, reason = should_retrain()
        if not needs_retrain:
            print(f"\n  ✅ No retrain needed: {reason}")
            return
        print(f"\n  ⚠️  Retrain triggered: {reason}")

    tune = not args.no_tune
    start = time.time()

    try:
        # ── Backup ────────────────────────────────────────────────────
        if not args.no_backup:
            print("\n  📦 Backing up existing models...")
            backup_existing_models()

        # ── Retrain ───────────────────────────────────────────────────
        if not args.revenue_only:
            print("\n  🧠 Retraining Churn Model...")
            retrain_churn(tune)

        if not args.churn_only:
            print("\n  💰 Retraining Revenue & CLV Models...")
            retrain_revenue()

        # ── Re-generate intelligence ──────────────────────────────────
        print("\n  📊 Regenerating Account Intelligence...")
        regenerate_intelligence()

        elapsed = time.time() - start
        print(f"\n{'=' * 60}")
        print(f"  🎉 Retrain complete in {elapsed:.1f}s")
        print(f"{'=' * 60}\n")

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ Retrain FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
