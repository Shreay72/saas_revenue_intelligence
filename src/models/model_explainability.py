"""
Elite Model Explainability Module with Memory-Optimized SHAP
SaaS Revenue Risk & Retention Intelligence System
Week 2 - Production-grade SHAP analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import logging
from typing import Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ModelExplainer:
    """
    SHAP-based model explainability with memory optimization.
    
    Features:
    - Memory-efficient SHAP computation
    - Global feature importance
    - Individual prediction explanations
    - Feature interaction analysis
    """
    
    def __init__(self, model_dir: str = 'models/churn'):
        self.logger = self._setup_logger()
        self.model_dir = Path(model_dir)
        
        # Load artifacts
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
    
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
    
    def _load_feature_names(self):
        """Load feature names."""
        import json
        feature_path = self.model_dir / 'feature_names.json'
        
        if not feature_path.exists():
            self.logger.warning("Feature names file not found")
            return None
        
        with open(feature_path, 'r') as f:
            feature_data = json.load(f)
        
        return feature_data.get('encoded_features', [])
    
    def create_explainer(
        self, 
        X_background: np.ndarray,
        model_type: str = 'tree',
        max_background_samples: int = 100
    ):
        """
        Create SHAP explainer with memory optimization.
        
        Parameters:
        -----------
        X_background : array-like
            Background data for SHAP (use training set)
        model_type : str
            'tree' for tree models, 'linear' for linear models
        max_background_samples : int
            Max samples for background (prevents memory issues)
        """
        
        self.logger.info("Creating memory-optimized SHAP explainer...")
        
        # Handle calibrated models
        model = self.model
        if hasattr(model, 'calibrated_classifiers_'):
            model = model.calibrated_classifiers_[0].estimator
            self.logger.info("Detected calibrated model, using base estimator")
        
        # FIXED: Use background sample for memory efficiency
        if X_background.shape[0] > max_background_samples:
            self.logger.info(f"Sampling {max_background_samples} background samples for memory efficiency")
            indices = np.random.choice(X_background.shape[0], max_background_samples, replace=False)
            X_background_sampled = X_background[indices]
        else:
            X_background_sampled = X_background
        
        if model_type == 'tree':
            # FIXED: Use feature_perturbation for better memory management
            self.explainer = shap.TreeExplainer(
                model,
                data=X_background_sampled,
                feature_perturbation='interventional'
            )
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(model, X_background_sampled)
        else:
            # Kernel explainer (slowest but universal)
            self.explainer = shap.KernelExplainer(
                model.predict_proba, 
                X_background_sampled
            )
        
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, np.ndarray):
            self.expected_value = self.expected_value[1]  # Positive class
        
        self.logger.info("✅ SHAP explainer created")
        return self.explainer
    
    def compute_shap_values(self, X: np.ndarray, check_additivity: bool = False):
        """
        Compute SHAP values for dataset.
        
        Parameters:
        -----------
        X : array-like
            Data to explain
        check_additivity : bool
            Verify SHAP values sum correctly (slower but more accurate)
        """
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        self.logger.info(f"Computing SHAP values for {X.shape[0]} samples...")
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        
        # Handle multi-output models (binary classification)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Positive class (churn)
        
        self.logger.info("✅ SHAP values computed")
        return self.shap_values
    
    def plot_summary(self, X: np.ndarray, max_display: int = 20, plot_type: str = 'dot'):
        """
        Plot SHAP summary.
        
        Parameters:
        -----------
        X : array-like
            Data (used for feature values in plot)
        max_display : int
            Number of top features to show
        plot_type : str
            'dot' (beeswarm) or 'bar' (mean absolute SHAP)
        """
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X, 
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type=plot_type,
            show=False
        )
        plt.tight_layout()
        return plt.gcf()
    
    def plot_bar(self, X: np.ndarray, max_display: int = 20):
        """Plot mean absolute SHAP values (global feature importance)."""
        return self.plot_summary(X, max_display=max_display, plot_type='bar')
    
    def explain_prediction(
        self, 
        X: np.ndarray, 
        index: int = 0,
        plot_type: str = 'waterfall'
    ):
        """
        Explain individual prediction.
        
        Parameters:
        -----------
        X : array-like
            Test data
        index : int
            Index of sample to explain
        plot_type : str
            'waterfall' or 'force'
        """
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        if plot_type == 'waterfall':
            # Create explanation object
            shap_explanation = shap.Explanation(
                values=self.shap_values[index],
                base_values=self.expected_value,
                data=X[index],
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap_explanation, show=False)
            plt.tight_layout()
            return plt.gcf()
        
        elif plot_type == 'force':
            # Force plot (interactive)
            shap.force_plot(
                self.expected_value,
                self.shap_values[index],
                X[index],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            return plt.gcf()
    
    def plot_dependence(
        self, 
        X: np.ndarray, 
        feature_name: str,
        interaction_feature: Optional[str] = None
    ):
        """
        Plot SHAP dependence for a specific feature.
        
        Shows how feature value affects prediction.
        
        Parameters:
        -----------
        X : array-like
            Data
        feature_name : str
            Feature to analyze
        interaction_feature : str, optional
            Feature to color by (for interaction detection)
        """
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        return plt.gcf()
    
    def get_top_features_for_account(
        self, 
        X: np.ndarray, 
        index: int = 0, 
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N features driving prediction for specific account.
        
        Returns DataFrame with feature names, SHAP values, and feature values.
        """
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        shap_vals = self.shap_values[index]
        feature_vals = X[index]
        
        # Create DataFrame
        feature_impact = pd.DataFrame({
            'feature': self.feature_names[:len(shap_vals)],
            'shap_value': shap_vals,
            'feature_value': feature_vals,
            'abs_shap': np.abs(shap_vals)
        }).sort_values('abs_shap', ascending=False).head(top_n)
        
        # Add interpretation
        feature_impact['impact'] = feature_impact['shap_value'].apply(
            lambda x: 'Increases Churn Risk' if x > 0 else 'Decreases Churn Risk'
        )
        
        return feature_impact[['feature', 'feature_value', 'shap_value', 'impact']]
    
    def create_risk_explanation(
        self, 
        X: np.ndarray, 
        index: int,
        churn_probability: float
    ) -> str:
        """
        Generate human-readable explanation for churn risk.
        
        Returns:
        --------
        str : Business-friendly explanation
        """
        
        top_features = self.get_top_features_for_account(X, index, top_n=5)
        
        risk_level = 'HIGH' if churn_probability > 0.6 else 'MEDIUM' if churn_probability > 0.3 else 'LOW'
        
        explanation = f"""
🎯 CHURN RISK ANALYSIS
{'='*60}

Account Risk Level: {risk_level} ({churn_probability*100:.1f}% probability)

Top Risk Factors:
"""
        
        for idx, row in top_features.iterrows():
            direction = "↑" if row['shap_value'] > 0 else "↓"
            explanation += f"\n{direction} {row['feature']}: {row['feature_value']:.2f} ({row['impact']})"
        
        explanation += "\n\n"
        
        # Recommendations
        if risk_level == 'HIGH':
            explanation += """
💡 RECOMMENDED ACTIONS:
   • Immediate account manager intervention
   • Review recent support tickets
   • Offer custom retention package
   • Schedule executive business review
"""
        elif risk_level == 'MEDIUM':
            explanation += """
💡 RECOMMENDED ACTIONS:
   • Trigger engagement campaign
   • Offer product training session
   • Monitor usage patterns closely
   • Proactive check-in within 1 week
"""
        else:
            explanation += """
💡 RECOMMENDED ACTIONS:
   • Continue standard monitoring
   • Encourage feature adoption
   • Quarterly business review
"""
        
        return explanation


def analyze_model_with_shap(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    model_dir: str = 'models/churn',
    max_background_samples: int = 100
):
    """
    Create full SHAP analysis with memory optimization.
    
    Usage:
        explainer = analyze_model_with_shap(X_train, X_test)
        explainer.plot_summary(X_test)
    """
    
    explainer = ModelExplainer(model_dir)
    
    # Detect model type
    model = explainer.model
    if hasattr(model, 'calibrated_classifiers_'):
        model = model.calibrated_classifiers_[0].estimator
    
    model_type = 'tree' if hasattr(model, 'feature_importances_') else 'linear'
    
    explainer.create_explainer(
        X_train, 
        model_type=model_type,
        max_background_samples=max_background_samples
    )
    explainer.compute_shap_values(X_test)
    
    print("\n✅ SHAP analysis complete")
    print(f"   Model type: {model_type}")
    print(f"   Background samples: {min(X_train.shape[0], max_background_samples)}")
    print(f"   Test samples analyzed: {X_test.shape[0]}")
    print(f"\n💡 Use explainer.plot_summary(X_test) to visualize")
    
    return explainer


if __name__ == "__main__":
    print("Elite Model Explainability module ready.")
