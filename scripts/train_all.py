"""
Full Training Pipeline — One Command to Rule Them All
SaaS Revenue Intelligence System

Runs all training steps in correct order:
    Step 1: Build account-level features   (Week 1)
    Step 2: Train churn model              (Week 2)
    Step 3: Train revenue & CLV models     (Week 3)
    Step 4: Generate account intelligence  (Week 4)
    Step 5: Verify all artifacts exist     (Pre-flight)

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --skip-week1   (if features already built)
    python scripts/train_all.py --verify-only  (just check artifacts exist)
    python scripts/train_all.py --no-tune      (skip Optuna, use defaults — fast)
"""

import sys
import time
import argparse
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)

ARTIFACTS = {
    "Week 1 — Account Features":   "data/processed/account_level_features.csv",
    "Week 2 — Churn Model":        "models/churn/churn_model_v1.pkl",
    "Week 2 — Preprocessor":       "models/churn/preprocessing_pipeline.pkl",
    "Week 2 — Feature Names":      "models/churn/feature_names.json",
    "Week 2 — Churn Metadata":     "models/churn/model_metadata.json",
    "Week 3 — Revenue Model":      "models/revenue/revenue_model_v1.pkl",
    "Week 3 — CLV Model":          "models/revenue/clv_model_v1.pkl",
    "Week 3 — Revenue Metadata":   "models/revenue/revenue_metadata.json",
    "Week 3 — CLV Metadata":       "models/revenue/clv_metadata.json",
    "Week 4 — Intelligence CSV":   "data/processed/account_intelligence.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def run_step(step_num: int, total: int, description: str, fn):
    """Run a single pipeline step with timing and full error reporting."""
    print(f"\n{'=' * 70}")
    print(f"  [{step_num}/{total}] {description}")
    print(f"{'=' * 70}")
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        print(f"\n  ✅ Done in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ FAILED after {elapsed:.1f}s")
        print(f"     Error: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def step_build_features():
    """Week 1 — Build account-level feature dataset from 5 raw CSVs."""
    from src.data.data_loader import DataLoader
    from src.data.data_cleaning import DataCleaner
    from src.data.data_validator import DataValidator
    from src.data.feature_engineering import FeatureEngineer

    loader  = DataLoader("data/raw")
    raw     = loader.load_all()
    print(f"  Loaded {len(raw)} datasets:")
    for name, df in raw.items():
        print(f"    {name:<20} → {df.shape[0]:,} rows × {df.shape[1]} cols")

    cleaner = DataCleaner()
    cleaned = cleaner.clean_all(raw)

    validator = DataValidator()
    validator.validate_raw_datasets(cleaned)

    engineer = FeatureEngineer()
    df = engineer.build_account_level_dataset(cleaned)

    validator.validate_account_level_dataset(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/account_level_features.csv", index=False)

    print(f"\n  Saved : data/processed/account_level_features.csv")
    print(f"  Shape : {df.shape[0]} accounts × {df.shape[1]} features")
    print(f"  Churn : {df['churn_flag'].mean() * 100:.1f}%")
    print(f"  MRR   : ${df['total_mrr'].min():,.0f} – ${df['total_mrr'].max():,.0f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CHURN MODEL
# ─────────────────────────────────────────────────────────────────────────────

def _make_churn_step(tune: bool):
    """Return a closure so the tune flag is captured cleanly."""
    def step_train_churn():
        """Week 2 — Train churn prediction model (3 models, pick best)."""
        from src.models.train_churn import ChurnModelTrainer

        trainer = ChurnModelTrainer(
            tune_hyperparameters=tune,
            n_trials=50 if tune else 5,
            cost_fp=100.0,
            cost_fn=5000.0,
        )

        trainer.load_data(data_path="data/processed/account_level_features.csv")
        print(f"  Train : {trainer.X_train.shape} | "
              f"Val : {trainer.X_val.shape} | "
              f"Test : {trainer.X_test.shape}")

        trainer.train_logistic_regression()
        trainer.train_random_forest()
        trainer.train_xgboost()

        comparison = trainer.compare_models()

        for model_name in ("XGBoost", "Random Forest"):
            if model_name in trainer.models:
                trainer.get_feature_importance(model_name, top_n=10)
                break

        trainer.save_best_model(output_dir="models/churn")

        best = comparison.iloc[0]
        print(f"\n  Best model : {best['Model']}")
        print(f"  ROC-AUC    : {best['ROC-AUC']:.4f}")

        return comparison.to_dict("records")

    return step_train_churn


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — REVENUE & CLV MODELS
# ─────────────────────────────────────────────────────────────────────────────

def step_train_revenue():
    """Week 3 — Train MRR forecast model + deterministic CLV."""
    import pandas as pd
    from src.models.train_revenue import RevenueModelTrainer
    from src.pipelines.churn_pipeline import load_pipeline

    df = pd.read_csv("data/processed/account_level_features.csv")

    if "churn_probability" not in df.columns:
        print("  Attaching churn probabilities from churn model...")
        churn_pipeline = load_pipeline()
        df["churn_probability"] = churn_pipeline.predict_proba(df)
        df.to_csv("data/processed/account_level_features.csv", index=False)
        print(f"  churn_probability : "
              f"{df['churn_probability'].min():.3f} – "
              f"{df['churn_probability'].max():.3f}")

    trainer = RevenueModelTrainer(output_dir="models/revenue")
    mrr_metrics, clv_metrics = trainer.run(
        data_path="data/processed/account_level_features.csv"
    )

    print(f"\n  MRR Model  : {mrr_metrics['model_name']}")
    print(f"  R²         : {mrr_metrics['r2_mean']:.4f} ± {mrr_metrics['r2_std']:.4f}")
    print(f"  MAE        : ${mrr_metrics['mae_mean']:,.2f}")
    print(f"\n  CLV Method : {clv_metrics['method']}")
    print(f"  CLV Mean   : ${clv_metrics['clv_mean']:,.2f}")
    print(f"  CLV Median : ${clv_metrics['clv_median']:,.2f}")

    return mrr_metrics, clv_metrics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ACCOUNT INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def step_generate_intelligence():
    """
    Week 4 — Score every account: risk tier + recommendations.

    Column contract (what RiskEngine + RecommendationEngine expect on df):
        churn_probability  ← from ChurnPipeline
        revenue_at_risk    ← from RevenuePipeline.calculate_revenue_at_risk()
        health_score       ← from RevenuePipeline.calculate_health_scores()
        clv                ← from RevenuePipeline.predict_clv()
                             NOTE: must be named 'clv', NOT 'predicted_clv'

    generate_intelligence(df) internally calls score_portfolio(df),
    which requires all four columns above to already exist on df.
    """
    import pandas as pd
    from src.pipelines.churn_pipeline import load_pipeline
    from src.pipelines.revenue_pipeline import load_revenue_pipeline
    from src.recommendation.recommendation_engine import RecommendationEngine

    df = pd.read_csv("data/processed/account_level_features.csv")
    print(f"  Loaded {len(df)} accounts")

    # ── Step 1: Churn probabilities ──────────────────────────────────────
    print("  Scoring churn probabilities...")
    churn_pipeline = load_pipeline()
    df["churn_probability"] = churn_pipeline.predict_proba(df)

    # ── Step 2: Revenue metrics ──────────────────────────────────────────
    # Must compute and assign BEFORE calling generate_intelligence()
    # because RiskEngine.score_portfolio() reads these columns directly.
    print("  Calculating CLV + Revenue at Risk + Health Score...")
    rev_pipeline = load_revenue_pipeline()

    df["revenue_at_risk"] = rev_pipeline.calculate_revenue_at_risk(df)
    df["health_score"]    = rev_pipeline.calculate_health_scores(df)
    # ✅ Named 'clv' — this is what RiskEngine._compute_percentiles()
    #    and assign_risk_type() read. 'predicted_clv' would be ignored.
    df["clv"]             = rev_pipeline.predict_clv(df)

    print(f"  Revenue at Risk : ${df['revenue_at_risk'].sum():,.2f} total")
    print(f"  Health Score    : {df['health_score'].mean():.1f} avg")
    print(f"  CLV             : ${df['clv'].mean():,.2f} avg")

    # ── Step 3: Full intelligence table ─────────────────────────────────
    # generate_intelligence(df) handles:
    #   → RiskEngine.score_portfolio(df)    (risk scores + tiers)
    #   → BusinessRules.evaluate_portfolio(df)  (recommendations)
    #   → expected recovery calculations
    print("  Generating intelligence table (risk + recommendations)...")
    rec_engine = RecommendationEngine()
    intelligence_df = rec_engine.generate_intelligence(df)

    # ── Step 4: Save ─────────────────────────────────────────────────────
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    intelligence_df.to_csv("data/processed/account_intelligence.csv", index=False)

    def _count(col, val):
        return (intelligence_df[col] == val).sum() \
               if col in intelligence_df.columns else "?"

    critical    = _count("risk_tier", "CRITICAL")
    high        = _count("risk_tier", "HIGH")
    medium      = _count("risk_tier", "MEDIUM")
    low         = _count("risk_tier", "LOW")
    total_risk  = intelligence_df["revenue_at_risk"].sum()   \
                  if "revenue_at_risk"  in intelligence_df.columns else 0
    recoverable = intelligence_df["expected_recovery"].sum() \
                  if "expected_recovery" in intelligence_df.columns else 0

    print(f"\n  Saved: data/processed/account_intelligence.csv")
    print(f"  Accounts scored : {len(intelligence_df)}")
    print(f"\n  Risk Distribution:")
    print(f"    CRITICAL : {critical}")
    print(f"    HIGH     : {high}")
    print(f"    MEDIUM   : {medium}")
    print(f"    LOW      : {low}")
    print(f"\n  Total Revenue at Risk : ${total_risk:>12,.2f}")
    print(f"  Total Recoverable     : ${recoverable:>12,.2f}")

    return intelligence_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — VERIFY ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

def step_verify_artifacts():
    """Pre-flight check — verify every model artifact exists on disk."""
    print("\n  Verifying all artifacts...\n")

    all_ok  = True
    missing = []

    for name, path in ARTIFACTS.items():
        p      = Path(path)
        exists = p.exists()
        status = "✅" if exists else "❌ MISSING"
        size   = ""
        if exists:
            b    = p.stat().st_size
            size = (f"  ({b / 1_000_000:.1f} MB)" if b > 1_000_000
                    else f"  ({b / 1_000:.1f} KB)")
        print(f"    {status}  {name:<35} → {path}{size}")
        if not exists:
            all_ok = False
            missing.append(path)

    if all_ok:
        print("\n  ✅ All artifacts present — system is ready.")
        print("\n  ─────────────────────────────────────────────────────")
        print("  Start API:        python scripts/run_api.py")
        print("  Start Dashboard:  python scripts/run_streamlit.py")
        print("  Run all tests:    pytest tests/ -v")
        print("  ─────────────────────────────────────────────────────")
    else:
        print(f"\n  ❌ {len(missing)} artifact(s) missing:")
        for p in missing:
            print(f"     • {p}")
        print("\n  Re-run: python scripts/train_all.py")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train all models end-to-end",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python scripts/train_all.py                        # Full pipeline (all 5 steps)
  python scripts/train_all.py --skip-week1           # Skip feature engineering
  python scripts/train_all.py --verify-only          # Only check artifacts exist
  python scripts/train_all.py --no-tune              # Skip Optuna (fast, ~2 min)
  python scripts/train_all.py --skip-week1 --no-tune # Fastest re-train
        """
    )
    parser.add_argument(
        "--skip-week1",
        action="store_true",
        help="Skip feature engineering (account_level_features.csv already exists)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify artifacts exist — do not retrain anything"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable Optuna hyperparameter tuning (uses defaults, much faster)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  🚀 SaaS Revenue Intelligence — Full Training Pipeline")
    print("=" * 70)

    if args.verify_only:
        print()
        step_verify_artifacts()
        return

    if args.skip_week1:
        print("\n  ⏭️  --skip-week1  : Feature engineering will be skipped")
    if args.no_tune:
        print("  ⚡  --no-tune     : Optuna disabled — using default hyperparameters")

    tune         = not args.no_tune
    total_steps  = 4 if args.skip_week1 else 5
    step_counter = [0]

    def next_step():
        step_counter[0] += 1
        return step_counter[0]

    total_start = time.time()

    try:
        # ── Step 1 : Features ────────────────────────────────────────────
        if not args.skip_week1:
            run_step(
                next_step(), total_steps,
                "Building Account-Level Features (Week 1)",
                step_build_features
            )
        else:
            step_counter[0] += 1
            features_path = Path("data/processed/account_level_features.csv")
            if not features_path.exists():
                print("\n  ❌ --skip-week1 set but account_level_features.csv not found!")
                print("     Run without --skip-week1 to build it first.")
                sys.exit(1)
            print(f"\n  ⏭️  Skipping Week 1 — using: {features_path}")

        # ── Step 2 : Churn Model ─────────────────────────────────────────
        run_step(
            next_step(), total_steps,
            "Training Churn Model (Week 2)",
            _make_churn_step(tune)
        )

        # ── Step 3 : Revenue & CLV ───────────────────────────────────────
        run_step(
            next_step(), total_steps,
            "Training Revenue & CLV Models (Week 3)",
            step_train_revenue
        )

        # ── Step 4 : Account Intelligence ────────────────────────────────
        run_step(
            next_step(), total_steps,
            "Generating Account Intelligence (Week 4)",
            step_generate_intelligence
        )

        # ── Step 5 : Verify ──────────────────────────────────────────────
        run_step(
            next_step(), total_steps,
            "Verifying All Artifacts (Pre-flight Check)",
            step_verify_artifacts
        )

    except Exception as e:
        total_elapsed = time.time() - total_start
        print(f"\n{'=' * 70}")
        print(f"  ❌ TRAINING FAILED after {total_elapsed:.1f}s")
        print(f"  Error: {e}")
        print(f"{'=' * 70}")
        print("\n  Full traceback:")
        traceback.print_exc()
        sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  🎉 ALL STEPS COMPLETED SUCCESSFULLY in {total_elapsed:.1f}s")
    print(f"{'=' * 70}")
    print("""
  ┌──────────────────────────────────────────────────────┐
  │  Next steps:                                         │
  │                                                      │
  │  Start API:        python scripts/run_api.py         │
  │  Start Dashboard:  python scripts/run_streamlit.py   │
  │  Run Tests:        pytest tests/ -v                  │
  └──────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
