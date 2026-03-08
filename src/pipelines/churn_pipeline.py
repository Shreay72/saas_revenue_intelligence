"""
Production Churn Prediction Pipeline
Handles model loading, inference, and business recommendations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ChurnPredictionPipeline:
    """
    Production-grade churn prediction pipeline.
    
    Features:
    - Model loading with validation
    - Batch prediction
    - Risk level categorization
    - Business recommendations
    - SHAP-ready for explainability
    """
    
    def __init__(self, model_dir: str = 'models/churn'):
        self.logger = self._setup_logger()
        self.model_dir = Path(model_dir)
        
        # Load model artifacts
        self.model = self._load_model()
        self.preprocessor = self._load_preprocessor()
        self.feature_names = self._load_feature_names()
        self.metadata = self._load_metadata()
        
        # Business thresholds
        self.risk_thresholds = {
            'critical': 0.7,
            'high': 0.5,
            'medium': 0.3,
            'low': 0.0
        }
    
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _load_model(self):
        """Load trained model."""
        model_path = self.model_dir / 'churn_model_v1.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        self.logger.info(f"✅ Model loaded: {model_path}")
        
        return model
    
    def _load_preprocessor(self):
        """Load preprocessing pipeline."""
        preprocessor_path = self.model_dir / 'preprocessing_pipeline.pkl'
        
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        preprocessor = joblib.load(preprocessor_path)
        self.logger.info("✅ Preprocessor loaded")
        
        return preprocessor
    
    def _load_feature_names(self) -> Dict:
        """Load feature names."""
        feature_path = self.model_dir / 'feature_names.json'
        
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                feature_names = json.load(f)
            return feature_names
        else:
            self.logger.warning("Feature names not found")
            return {}
    
    def _load_metadata(self) -> Dict:
        """Load model metadata."""
        metadata_path = self.model_dir / 'model_metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.logger.info(f"✅ Metadata loaded: {metadata.get('model_name', 'Unknown')}")
            return metadata
        else:
            self.logger.warning("Metadata not found")
            return {}
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict churn (binary: 0 or 1).
        
        Args:
            df: DataFrame with account features
            
        Returns:
            Array of binary predictions (0 = No Churn, 1 = Churn)
        """
        X = self._prepare_features(df)
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probabilities.
        
        Args:
            df: DataFrame with account features
            
        Returns:
            Array of churn probabilities (probability of class 1)
        """
        X = self._prepare_features(df)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            probabilities = self.model.decision_function(X)
            probabilities = 1 / (1 + np.exp(-probabilities))
        
        return probabilities
    
    def predict_with_risk_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with risk level categorization.
        
        Args:
            df: DataFrame with account features
            
        Returns:
            DataFrame with predictions, probabilities, and risk levels
        """
        predictions = self.predict(df)
        probabilities = self.predict_proba(df)
        
        results = pd.DataFrame({
            'account_id': df['account_id'] if 'account_id' in df.columns else range(len(df)),
            'account_name': df['account_name'] if 'account_name' in df.columns else ['Unknown'] * len(df),
            'churn_prediction': predictions,
            'churn_probability': probabilities,
            'risk_score': probabilities,
            'risk_level': self._categorize_risk(probabilities)
        })
        
        # Sort by churn_probability descending
        results = results.sort_values('churn_probability', ascending=False).reset_index(drop=True)
        
        return results
    
    def _categorize_risk(self, probabilities: np.ndarray) -> np.ndarray:
        """Categorize churn risk into levels."""
        risk_levels = np.empty(len(probabilities), dtype=object)
        
        for i, prob in enumerate(probabilities):
            if prob >= self.risk_thresholds['critical']:
                risk_levels[i] = 'HIGH'
            elif prob >= self.risk_thresholds['high']:
                risk_levels[i] = 'HIGH'
            elif prob >= self.risk_thresholds['medium']:
                risk_levels[i] = 'MEDIUM'
            else:
                risk_levels[i] = 'LOW'
        
        return risk_levels
    
    def predict_with_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with business recommendations.
        
        Args:
            df: DataFrame with account features
            
        Returns:
            DataFrame with predictions, risk levels, and recommendations
        """
        # ✅ FIX: Reset index to ensure alignment after sorting
        df_reset = df.reset_index(drop=True)
        
        # Get base predictions with risk levels
        results = self.predict_with_risk_levels(df_reset)
        
        # ✅ FIX: Use df_reset with aligned index — avoids KeyError
        results['recommended_action'] = results.apply(
            lambda row: self._get_recommendation(row, df_reset.loc[row.name]),
            axis=1
        )
        
        # Add priority score
        results['priority_score'] = self._calculate_priority(results, df_reset)
        
        # Convert to categorical priority (1=High, 2=Medium, 3=Low)
        results['priority'] = results['risk_level'].map({
            'HIGH': 1,
            'MEDIUM': 2,
            'LOW': 3
        })
        
        # Sort by priority score
        results = results.sort_values('priority_score', ascending=False).reset_index(drop=True)
        
        return results
    
    def _get_recommendation(self, row: pd.Series, account_data: pd.Series) -> str:
        """Generate business recommendation based on risk level."""
        
        risk_level = row['risk_level']
        probability = row['churn_probability']
        
        if risk_level == 'HIGH':
            return (
                f"🚨 URGENT ACTION REQUIRED (Risk: {probability:.1%})\n"
                "• Schedule immediate executive intervention call\n"
                "• Offer retention package (discount/upgrade)\n"
                "• Assign dedicated account manager\n"
                "• Review support tickets for unresolved issues"
            )
        
        elif risk_level == 'MEDIUM':
            return (
                f"⚡ MEDIUM RISK - Monitor closely (Risk: {probability:.1%})\n"
                "• Add to weekly check-in list\n"
                "• Send usage analytics and best practices\n"
                "• Invite to customer success webinar\n"
                "• Check engagement trends"
            )
        
        else:  # LOW
            return (
                f"✅ LOW RISK - Standard engagement (Risk: {probability:.1%})\n"
                "• Continue regular touchpoints\n"
                "• Share product updates and new features\n"
                "• Identify upsell opportunities\n"
                "• Request testimonial/referral"
            )
    
    def _calculate_priority(self, results: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate action priority score.
        
        Factors:
        - Churn probability    (0-100 points)
        - Revenue/MRR          (0-50  points)
        - Engagement level     (0-30  points)
        - Support risk         (0-20  points)
        """
        priority_scores = results['churn_probability'] * 100
        
        if 'mrr' in df.columns:
            revenue_normalized = (df['mrr'] - df['mrr'].min()) / (df['mrr'].max() - df['mrr'].min() + 1e-8)
            priority_scores += revenue_normalized.values * 50
        
        if 'engagement_score' in df.columns:
            engagement_normalized = 1 - (df['engagement_score'] / 100)
            priority_scores += engagement_normalized.values * 30
        
        if 'support_risk_score' in df.columns:
            support_normalized = (df['support_risk_score'] - df['support_risk_score'].min()) / \
                               (df['support_risk_score'].max() - df['support_risk_score'].min() + 1e-8)
            priority_scores += support_normalized.values * 20
        
        return priority_scores.values
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Keeps account_name since preprocessor was trained with it.
        Only drops target (churn_flag) and identifier (account_id).
        """
        df = df.copy()
        
        cols_to_drop = ['churn_flag', 'account_id']
        X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        X_processed = self.preprocessor.transform(X)
        
        return X_processed
    
    def get_model_info(self) -> Dict:
        """Get model information and metadata."""
        
        info = {
            'model_type': type(self.model).__name__,
            'model_name': self.metadata.get('model_name', 'Unknown'),
            'train_date': self.metadata.get('train_date', 'Unknown'),
            'metrics': self.metadata.get('metrics', {}),
            'optimal_threshold': self.metadata.get('metrics', {}).get('optimal_threshold', 0.5),
            'model_dir': str(self.model_dir),
            'metadata': self.metadata,
            'risk_thresholds': self.risk_thresholds,
            'feature_count': len(self.feature_names.get('encoded_features', [])),
            'has_shap_support': True
        }
        
        return info
    
    def batch_predict(
        self,
        df: pd.DataFrame,
        batch_size: int = 100,
        include_recommendations: bool = True
    ) -> pd.DataFrame:
        """
        Predict in batches for large datasets.
        
        Args:
            df: DataFrame with account features
            batch_size: Number of rows per batch
            include_recommendations: Whether to include recommendations
            
        Returns:
            DataFrame with all predictions
        """
        results_list = []
        total_batches = (len(df) - 1) // batch_size + 1
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            
            if include_recommendations:
                batch_results = self.predict_with_recommendations(batch)
            else:
                batch_results = self.predict_with_risk_levels(batch)
            
            results_list.append(batch_results)
            self.logger.info(f"Processed batch {i // batch_size + 1}/{total_batches}")
        
        return pd.concat(results_list, ignore_index=True)


def load_pipeline(model_dir: str = 'models/churn') -> ChurnPredictionPipeline:
    """
    Load churn prediction pipeline.
    
    Usage:
        pipeline = load_pipeline()
        predictions = pipeline.predict(df)
        results    = pipeline.predict_with_recommendations(df)
    """
    return ChurnPredictionPipeline(model_dir=model_dir)


if __name__ == "__main__":
    print("Churn Prediction Pipeline ready.")
    print("\nUsage:")
    print("  from src.pipelines.churn_pipeline import load_pipeline")
    print("  pipeline = load_pipeline()")
    print("  predictions = pipeline.predict(df)")
