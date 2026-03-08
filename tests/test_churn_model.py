"""
Elite Unit Tests for Churn Models
Run: pytest tests/test_churn_model.py -v
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.evaluate import ModelEvaluator
from src.pipelines.churn_pipeline import load_pipeline


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initializes with correct costs."""
        evaluator = ModelEvaluator(cost_fp=100, cost_fn=5000)
        assert evaluator.cost_fp == 100
        assert evaluator.cost_fn == 5000
    
    def test_evaluate_classification(self):
        """Test classification evaluation returns all required metrics."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_pred_proba = np.array([0.1, 0.4, 0.8, 0.9, 0.3])
        
        metrics = evaluator.evaluate_classification(y_true, y_pred, y_pred_proba)
        
        # Check all required metrics exist
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert 'total_business_cost' in metrics
        assert 'cost_fp_total' in metrics
        assert 'cost_fn_total' in metrics
        
        # Check values are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert metrics['total_business_cost'] >= 0
    
    def test_optimize_threshold_business_cost(self):
        """Test threshold optimization for business cost."""
        evaluator = ModelEvaluator(cost_fp=100, cost_fn=5000)
        
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        threshold, metrics = evaluator.optimize_threshold(
            y_true, y_pred_proba, metric='business_cost'
        )
        
        # Check threshold is valid
        assert 0 < threshold < 1
        
        # Check metrics exist
        assert 'business_cost' in metrics
        assert 'threshold' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'optimized_score' in metrics
        
        # Check values are reasonable
        assert metrics['threshold'] == threshold
        assert metrics['business_cost'] >= 0
    
    def test_optimize_threshold_f1(self):
        """Test threshold optimization for F1 score."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        threshold, metrics = evaluator.optimize_threshold(
            y_true, y_pred_proba, metric='f1'
        )
        
        assert 0 < threshold < 1
        assert 'f1_score' in metrics
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_create_risk_levels(self):
        """Test risk level categorization."""
        evaluator = ModelEvaluator()
        
        probabilities = np.array([0.1, 0.4, 0.7, 0.9])
        risk_df = evaluator.create_risk_levels(probabilities)
        
        # Check DataFrame structure
        assert len(risk_df) == 4
        assert 'churn_probability' in risk_df.columns
        assert 'risk_level' in risk_df.columns
        
        # Check risk levels are correctly assigned
        assert risk_df.iloc[0]['risk_level'] == 'LOW'
        assert risk_df.iloc[1]['risk_level'] == 'MEDIUM'
        assert risk_df.iloc[2]['risk_level'] == 'HIGH'
        assert risk_df.iloc[3]['risk_level'] == 'HIGH'
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        evaluator = ModelEvaluator()
        
        # Mock metrics for two models
        results = {
            'Model A': {
                'roc_auc': 0.85,
                'f1_score': 0.75,
                'precision': 0.80,
                'recall': 0.70,
                'total_business_cost': 10000
            },
            'Model B': {
                'roc_auc': 0.90,
                'f1_score': 0.80,
                'precision': 0.85,
                'recall': 0.75,
                'total_business_cost': 8000
            }
        }
        
        comparison = evaluator.compare_models(results)
        
        # Check comparison DataFrame
        assert len(comparison) == 2
        assert 'Model' in comparison.columns
        assert 'ROC-AUC' in comparison.columns
        
        # Check sorting (should be by ROC-AUC descending)
        assert comparison.iloc[0]['Model'] == 'Model B'
        assert comparison.iloc[0]['ROC-AUC'] == 0.90


class TestChurnPipeline:
    """Test production pipeline."""
    
    @pytest.fixture
    def check_models_exist(self):
        """Check if models are trained."""
        model_path = Path('models/churn/churn_model_v1.pkl')
        if not model_path.exists():
            pytest.skip("Models not trained yet - run train_churn.py first")
    
    @pytest.fixture
    def check_data_exists(self):
        """Check if processed data exists."""
        data_path = Path('data/processed/account_level_features.csv')
        if not data_path.exists():
            pytest.skip("Processed data not found - run Week 1 pipeline first")
    
    def test_pipeline_load(self, check_models_exist):
        """Test if pipeline loads successfully."""
        pipeline = load_pipeline()
        
        assert pipeline is not None
        assert pipeline.model is not None
        assert pipeline.preprocessor is not None
        assert pipeline.metadata is not None
    
    def test_predict(self, check_models_exist, check_data_exists):
        """Test prediction functionality."""
        pipeline = load_pipeline()
        df = pd.read_csv('data/processed/account_level_features.csv')
        df_sample = df.head(10)
        
        predictions = pipeline.predict(df_sample)
        
        # Check predictions
        assert len(predictions) == 10
        assert set(predictions).issubset({0, 1})
        assert predictions.dtype in [np.int32, np.int64]
    
    def test_predict_proba(self, check_models_exist, check_data_exists):
        """Test probability prediction."""
        pipeline = load_pipeline()
        df = pd.read_csv('data/processed/account_level_features.csv')
        df_sample = df.head(10)
        
        probabilities = pipeline.predict_proba(df_sample)
        
        # Check probabilities
        assert len(probabilities) == 10
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_predict_with_risk_levels(self, check_models_exist, check_data_exists):
        """Test risk level prediction."""
        pipeline = load_pipeline()
        df = pd.read_csv('data/processed/account_level_features.csv')
        df_sample = df.head(10)
        
        results = pipeline.predict_with_risk_levels(df_sample)
        
        # Check results structure
        assert len(results) == 10
        assert 'account_id' in results.columns
        assert 'churn_probability' in results.columns
        assert 'risk_level' in results.columns
        assert 'risk_score' in results.columns
        
        # Check risk levels are valid
        assert set(results['risk_level'].unique()).issubset({'LOW', 'MEDIUM', 'HIGH'})
        
        # Check sorting (should be by probability descending)
        assert results['churn_probability'].is_monotonic_decreasing
    
    def test_predict_with_recommendations(self, check_models_exist, check_data_exists):
        """Test predictions with recommendations."""
        pipeline = load_pipeline()
        df = pd.read_csv('data/processed/account_level_features.csv')
        df_sample = df.head(5)
        
        results = pipeline.predict_with_recommendations(df_sample)
        
        # Check results structure
        assert 'recommended_action' in results.columns
        assert 'priority' in results.columns
        assert 'risk_level' in results.columns
        
        # Check priority values
        assert set(results['priority'].unique()).issubset({1, 2, 3})
        
        # Check recommendations are strings
        assert all(isinstance(action, str) for action in results['recommended_action'])
    
    def test_get_model_info(self, check_models_exist):
        """Test model information retrieval."""
        pipeline = load_pipeline()
        
        info = pipeline.get_model_info()
        
        # Check info structure
        assert 'model_name' in info
        assert 'model_type' in info
        assert 'train_date' in info
        assert 'metrics' in info
        assert 'optimal_threshold' in info
        
        # Check metrics exist
        assert isinstance(info['metrics'], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
