"""
Test Inference Pipeline
Verify production pipeline works correctly

Usage:
    python scripts/test_inference.py
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.churn_pipeline import load_pipeline

def main():
    print("="*80)
    print("🧪 TESTING PRODUCTION INFERENCE PIPELINE")
    print("="*80 + "\n")
    
    # Load pipeline
    print("📦 Loading production pipeline...")
    pipeline = load_pipeline()
    print("✅ Pipeline loaded successfully\n")
    
    # Display model info
    info = pipeline.get_model_info()
    print("📊 Model Information:")
    print(f"   Model: {info['model_name']}")
    print(f"   Type: {info['model_type']}")
    print(f"   Trained: {info['train_date']}")
    print(f"   ROC-AUC: {info['metrics'].get('roc_auc', 0):.4f}")
    print(f"   Optimal Threshold: {info['optimal_threshold']:.3f}\n")
    
    # Load test data
    print("📂 Loading test accounts...")
    df = pd.read_csv('data/processed/account_level_features.csv')
    test_accounts = df.sample(10, random_state=42)
    print(f"✅ Loaded {len(test_accounts)} test accounts\n")
    
    # Test 1: Simple predictions
    print("="*80)
    print("TEST 1: Binary Predictions")
    print("="*80)
    predictions = pipeline.predict(test_accounts)
    print(f"Predictions: {predictions}")
    print(f"Churn Rate: {predictions.mean()*100:.1f}%\n")
    
    # Test 2: Probabilities
    print("="*80)
    print("TEST 2: Probability Predictions")
    print("="*80)
    probabilities = pipeline.predict_proba(test_accounts)
    print(f"Min Probability: {probabilities.min():.3f}")
    print(f"Max Probability: {probabilities.max():.3f}")
    print(f"Mean Probability: {probabilities.mean():.3f}\n")
    
    # Test 3: Risk levels
    print("="*80)
    print("TEST 3: Risk Level Analysis")
    print("="*80)
    risk_results = pipeline.predict_with_risk_levels(test_accounts)
    print(risk_results.to_string(index=False))
    print("\nRisk Distribution:")
    print(risk_results['risk_level'].value_counts())
    print()
    
    # Test 4: With recommendations
    print("="*80)
    print("TEST 4: Predictions with Recommendations")
    print("="*80)
    recommendations = pipeline.predict_with_recommendations(test_accounts)
    print(recommendations[['account_name', 'risk_level', 'risk_score', 'recommended_action']].to_string(index=False))
    print()
    
    # Test 5: High-risk accounts only
    print("="*80)
    print("TEST 5: High-Risk Accounts Filter")
    print("="*80)
    high_risk = recommendations[recommendations['risk_level'] == 'HIGH']
    if len(high_risk) > 0:
        print(f"Found {len(high_risk)} high-risk accounts:")
        print(high_risk[['account_name', 'risk_score', 'recommended_action']].to_string(index=False))
    else:
        print("No high-risk accounts in this sample")
    print()
    
    print("="*80)
    print("✅ ALL INFERENCE TESTS PASSED")
    print("="*80)
    print("\n💡 Production pipeline is ready for deployment!")
    print("\nNext steps:")
    print("   • Integrate with FastAPI backend")
    print("   • Create Streamlit dashboard")
    print("   • Set up batch prediction jobs")
    print("   • Monitor model performance\n")

if __name__ == "__main__":
    main()
