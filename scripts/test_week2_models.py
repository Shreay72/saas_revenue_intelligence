"""
Elite Week 2 Integration Test
Run: python scripts/test_week2_models.py [options]

Options:
  --tune          Enable Optuna hyperparameter tuning (recommended)
  --quick         Quick mode without tuning (for testing)
  --trials N      Number of Optuna trials (default: 50)
  --cost-fp N     Cost of false positive (default: 100)
  --cost-fn N     Cost of false negative (default: 5000)
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.train_churn import train_all_models

def main():
    parser = argparse.ArgumentParser(
        description='Train elite churn models with business cost optimization'
    )
    parser.add_argument('--tune', action='store_true', 
                       help='Enable Optuna hyperparameter tuning')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode without tuning')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of Optuna trials (default: 50)')
    parser.add_argument('--cost-fp', type=float, default=100.0,
                       help='Cost of false positive in USD (default: 100)')
    parser.add_argument('--cost-fn', type=float, default=5000.0,
                       help='Cost of false negative in USD (default: 5000)')
    
    args = parser.parse_args()
    
    # Determine tuning setting
    if args.tune:
        tune = True
    elif args.quick:
        tune = False
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("🎯 ELITE WEEK 2 MODEL TRAINING")
        print("="*80)
        print("\n⚙️  Configuration Options:")
        print(f"   • Hyperparameter Tuning: Optuna TPE sampler")
        print(f"   • Early Stopping: XGBoost validation-based")
        print(f"   • Threshold Optimization: Business cost minimization")
        print(f"   • Calibration: Sigmoid calibration")
        print(f"   • Feature Mapping: Real feature names")
        print(f"   • SHAP Ready: Memory-optimized")
        
        response = input("\n⚙️  Enable hyperparameter tuning? (slower, ~10-15 min) [Y/n]: ")
        tune = response.lower() != 'n'
    
    print("\n" + "="*80)
    print("🚀 ELITE WEEK 2 MODEL TRAINING")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"   Hyperparameter Tuning: {'✅ ENABLED' if tune else '❌ DISABLED'}")
    if tune:
        print(f"   Optuna Trials: {args.trials}")
    print(f"   Early Stopping: ✅ ENABLED (XGBoost)")
    print(f"   Threshold Optimization: ✅ ENABLED (Business Cost)")
    print(f"   Calibration: ✅ ENABLED (Sigmoid)")
    print(f"   Feature Name Mapping: ✅ ENABLED")
    print(f"   SHAP Support: ✅ READY (Memory-Optimized)")
    print(f"\n💰 Business Assumptions:")
    print(f"   Cost of False Positive: ${args.cost_fp:,.2f}")
    print(f"   Cost of False Negative: ${args.cost_fn:,.2f}")
    print("\n" + "="*80 + "\n")
    
    try:
        # Run training
        trainer, comparison = train_all_models(
            tune_hyperparameters=tune,
            n_trials=args.trials,
            cost_fp=args.cost_fp,
            cost_fn=args.cost_fn
        )
        
        print("\n" + "="*80)
        print("✅ ELITE WEEK 2 MODELS SUCCESSFULLY TRAINED!")
        print("="*80)
        
        print("\n📦 Artifacts Created:")
        print("   • models/churn/churn_model_v1.pkl")
        print("   • models/churn/preprocessing_pipeline.pkl")
        print("   • models/churn/model_metadata.json")
        print("   • models/churn/feature_names.json")
        
        print("\n📊 Model Performance:")
        best_model = comparison.iloc[0]
        print(f"   🏆 Best Model: {best_model['Model']}")
        print(f"   📈 ROC-AUC: {best_model['ROC-AUC']:.4f}")
        print(f"   💰 Business Cost: ${best_model['Business Cost']:,.2f}")
        
        print("\n💡 Next Steps:")
        print("   1. Run SHAP analysis for interpretability")
        print("      → python scripts/run_shap_analysis.py")
        print("   2. Create visualization notebooks")
        print("      → notebooks/03_churn_modeling.ipynb")
        print("   3. Test inference pipeline")
        print("      → python scripts/test_inference.py")
        print("   4. Deploy to production")
        print("      → python app/backend/main.py")
        
        print("\n" + "="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
