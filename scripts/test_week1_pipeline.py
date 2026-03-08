"""
Week 1 Integration Test - Fixed version
Run: python scripts/test_week1_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now imports will work
from src.data.data_loader import DataLoader
from src.data.data_cleaning import DataCleaner
from src.data.data_validator import DataValidator
from src.data.feature_engineering import FeatureEngineer
from src.pipelines.preprocessing_pipeline import build_preprocessing_pipeline


def main():
    print("\n" + "="*80)
    print("🧪 WEEK 1 FULL PIPELINE TEST")
    print("="*80 + "\n")
    
    try:
        # Step 1: Load
        print("[1/6] Loading raw datasets...")
        loader = DataLoader("data/raw")
        raw_data = loader.load_all()
        print("✅ Loaded successfully\n")
        
        # Step 2: Clean
        print("[2/6] Cleaning datasets...")
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_all(raw_data)
        print("✅ Cleaned successfully\n")
        
        # Step 3: Validate Raw
        print("[3/6] Validating raw datasets...")
        validator = DataValidator()
        validator.validate_raw_datasets(cleaned_data)
        print("✅ Validation passed\n")
        
        # Step 4: Feature Engineering
        print("[4/6] Building features...")
        engineer = FeatureEngineer()
        account_df = engineer.build_account_level_dataset(cleaned_data)
        print(f"✅ Dataset shape: {account_df.shape}\n")
        
        # Step 5: Validate Account-Level
        print("[5/6] Validating account dataset...")
        validator.validate_account_level_dataset(account_df)
        print("✅ Validation passed\n")
        
        # Step 6: Preprocessing
        print("[6/6] Running preprocessing pipeline...")
        preprocessor, X_train, X_test, y_train, y_test = build_preprocessing_pipeline(account_df)
        
        print(f"\nTrain Shape: {X_train.shape}")
        print(f"Test Shape: {X_test.shape}")
        print(f"\nChurn Distribution (Train):")
        print(y_train.value_counts())
        
        # Save processed dataset
        print("\n" + "="*80)
        account_df.to_csv('data/processed/account_level_features.csv', index=False)
        print("✅ Saved: data/processed/account_level_features.csv")
        
        print("\n🎉 WEEK 1 PIPELINE SUCCESSFULLY EXECUTED!")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
