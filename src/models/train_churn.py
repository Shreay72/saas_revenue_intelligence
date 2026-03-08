"""
Elite Churn Model Training with Optuna Hyperparameter Optimization
SaaS Revenue Risk & Retention Intelligence System
Week 2 - Production-grade with early stopping and cost-sensitive optimization
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import joblib
from datetime import datetime
from typing import Tuple, Dict, List
import json

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# Optuna for hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.preprocessing_pipeline import build_preprocessing_pipeline
from src.models.evaluate import ModelEvaluator, evaluate_model


# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ChurnModelTrainer:
    """Elite churn model trainer with Optuna optimization."""
    
    def __init__(
        self, 
        tune_hyperparameters: bool = True,
        n_trials: int = 50,
        cost_fp: float = 100.0,
        cost_fn: float = 5000.0
    ):
        self.logger = self._setup_logger()
        self.evaluator = ModelEvaluator(cost_fp=cost_fp, cost_fn=cost_fn)
        self.models = {}
        self.results = {}
        self.tune_hyperparameters = tune_hyperparameters
        self.n_trials = n_trials
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.feature_names = None
        
        # Data placeholders
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None  # For early stopping
        self.y_val = None
        self.original_feature_names = None
    
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def load_data(self, data_path: str = 'data/processed/account_level_features.csv'):
        """Load data with validation split for early stopping."""
        
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Store original feature names
        features_df = df.drop(columns=['churn_flag', 'account_id', 'account_name'], errors='ignore')
        self.original_feature_names = features_df.columns.tolist()
        
        # Build preprocessing pipeline
        preprocessor, X_train, X_test, y_train, y_test = build_preprocessing_pipeline(df)
        
        # Split train into train + validation for early stopping
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        self.preprocessor = preprocessor
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Extract feature names
        self.feature_names = self._get_feature_names_from_preprocessor()
        
        self.logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        self.logger.info(f"Churn rate - Train: {y_train.mean()*100:.2f}%, Test: {y_test.mean()*100:.2f}%")
        
        return self
    
    def _get_feature_names_from_preprocessor(self) -> List[str]:
        """Extract feature names after preprocessing."""
        
        feature_names = []
        
        try:
            for name, transformer, columns in self.preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(columns)
                elif name == 'cat':
                    if hasattr(transformer, 'get_feature_names_out'):
                        cat_features = transformer.get_feature_names_out(columns)
                        feature_names.extend(cat_features)
        except Exception as e:
            self.logger.warning(f"Could not extract feature names: {e}")
            feature_names = [f"feature_{i}" for i in range(self.X_train.shape[1])]
        
        return feature_names
    
    def train_logistic_regression(self, cv_folds: int = 5) -> Tuple:
        """Train Logistic Regression with proper calibration."""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("Training Logistic Regression (Baseline)")
        self.logger.info("="*80)
        
        if self.tune_hyperparameters:
            self.logger.info("🔧 Hyperparameter tuning with Optuna...")
            
            def objective(trial):
                params = {
                    'C': trial.suggest_float('C', 0.001, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l2']),
                    'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
                    'class_weight': 'balanced',
                    'max_iter': 1000,
                    'random_state': 42
                }
                
                model = LogisticRegression(**params)
                
                # Cross-validation score
                scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=3, scoring='roc_auc', n_jobs=-1
                )
                
                return scores.mean()
            
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
            
            best_params = study.best_params
            best_params.update({'class_weight': 'balanced', 'max_iter': 1000, 'random_state': 42})
            
            self.logger.info(f"✅ Best parameters: {best_params}")
            self.logger.info(f"✅ Best CV ROC-AUC: {study.best_value:.4f}")
            
            base_model = LogisticRegression(**best_params)
        else:
            base_model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        
        # FIXED: Calibrate the model properly
        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
        calibrated_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        metrics = evaluate_model(
            calibrated_model, self.X_test, self.y_test, 
            "Logistic Regression",
            cost_fp=self.cost_fp, cost_fn=self.cost_fn
        )
        
        # Store
        self.models['Logistic Regression'] = calibrated_model
        self.results['Logistic Regression'] = metrics
        
        return calibrated_model, metrics
    
    def train_random_forest(self, cv_folds: int = 5) -> Tuple:
        """Train Random Forest with Optuna."""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("Training Random Forest")
        self.logger.info("="*80)
        
        if self.tune_hyperparameters:
            self.logger.info("🔧 Hyperparameter tuning with Optuna...")
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                model = RandomForestClassifier(**params)
                
                scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=3, scoring='roc_auc', n_jobs=-1
                )
                
                return scores.mean()
            
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
            
            best_params = study.best_params
            best_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})
            
            self.logger.info(f"✅ Best parameters: {best_params}")
            self.logger.info(f"✅ Best CV ROC-AUC: {study.best_value:.4f}")
            
            model = RandomForestClassifier(**best_params)
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        metrics = evaluate_model(
            model, self.X_test, self.y_test, 
            "Random Forest",
            cost_fp=self.cost_fp, cost_fn=self.cost_fn
        )
        
        # Store
        self.models['Random Forest'] = model
        self.results['Random Forest'] = metrics
        
        return model, metrics
    
    def train_xgboost(self, cv_folds: int = 5) -> Tuple:
        """Train XGBoost with Optuna and early stopping (XGBoost 3.x compatible)."""
        
        self.logger.info("\n" + "="*80)
        self.logger.info("Training XGBoost with Early Stopping")
        self.logger.info("="*80)
        
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        if self.tune_hyperparameters:
            self.logger.info("🔧 Hyperparameter tuning with Optuna...")
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'scale_pos_weight': scale_pos_weight,
                    'eval_metric': 'logloss',
                    'random_state': 42,
                    'tree_method': 'hist',
                    'device': 'cpu'
                }
                
                model = XGBClassifier(**params)
                
                # FIXED: XGBoost 3.x compatible early stopping
                try:
                    # Try XGBoost 3.x callback API
                    from xgboost.callback import EarlyStopping
                    
                    model.fit(
                        self.X_train, self.y_train,
                        eval_set=[(self.X_val, self.y_val)],
                        callbacks=[EarlyStopping(rounds=20, save_best=True)],
                        verbose=False
                    )
                except (ImportError, TypeError):
                    # Fallback for XGBoost 2.x
                    model.fit(
                        self.X_train, self.y_train,
                        eval_set=[(self.X_val, self.y_val)],
                        verbose=False
                    )
                
                # Score on validation set
                from sklearn.metrics import roc_auc_score
                y_pred_proba = model.predict_proba(self.X_val)[:, 1]
                score = roc_auc_score(self.y_val, y_pred_proba)
                
                return score
            
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
            
            best_params = study.best_params
            best_params.update({
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'logloss',
                'random_state': 42,
                'tree_method': 'hist',
                'device': 'cpu'
            })
            
            self.logger.info(f"✅ Best parameters: {best_params}")
            self.logger.info(f"✅ Best CV ROC-AUC: {study.best_value:.4f}")
            
            model = XGBClassifier(**best_params)
        else:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=42,
                tree_method='hist',
                device='cpu'
            )
        
        # Final training with early stopping
        self.logger.info("Training final model with early stopping...")
        
        try:
            # XGBoost 3.x API
            from xgboost.callback import EarlyStopping
            
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                callbacks=[EarlyStopping(rounds=20, save_best=True)],
                verbose=True
            )
            self.logger.info(f"✅ Early stopping at iteration: {model.best_iteration}")
        except (ImportError, TypeError, AttributeError):
            # Fallback for XGBoost 2.x or if callbacks don't work
            self.logger.warning("Early stopping not available, training without it")
            model.fit(self.X_train, self.y_train, verbose=True)
        
        # Evaluate
        metrics = evaluate_model(
            model, self.X_test, self.y_test, 
            "XGBoost",
            cost_fp=self.cost_fp, cost_fn=self.cost_fn
        )
        
        # Store
        self.models['XGBoost'] = model
        self.results['XGBoost'] = metrics
        
        return model, metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models."""
        
        comparison = self.evaluator.compare_models(self.results)
        
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON")
        print("="*80)
        print(comparison.to_string(index=False))
        print("="*80 + "\n")
        
        # Find best model
        best_model_name = comparison.iloc[0]['Model']
        best_roc_auc = comparison.iloc[0]['ROC-AUC']
        
        print(f"🏆 Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})\n")
        
        return comparison
    
    def get_feature_importance(self, model_name: str = 'XGBoost', top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from tree-based model."""
        
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            self.logger.warning(f"Model {model_name} doesn't have feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        print(f"\n🔥 Top {top_n} Features ({model_name}):")
        print("="*80)
        print(feature_importance.to_string(index=False))
        print("="*80 + "\n")
        
        return feature_importance
    
    def save_best_model(self, output_dir: str = 'models/churn', model_name: str = None):
        """Save best model and artifacts."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine best model
        if model_name is None:
            comparison = self.evaluator.compare_models(self.results)
            model_name = comparison.iloc[0]['Model']
        
        best_model = self.models[model_name]
        best_metrics = self.results[model_name]
        
        # Save model
        model_file = output_path / 'churn_model_v1.pkl'
        joblib.dump(best_model, model_file)
        self.logger.info(f"✅ Model saved: {model_file}")
        
        # Save preprocessor
        preprocessor_file = output_path / 'preprocessing_pipeline.pkl'
        joblib.dump(self.preprocessor, preprocessor_file)
        self.logger.info(f"✅ Preprocessor saved: {preprocessor_file}")
        
        # Save feature names
        feature_names_data = {
            'original_features': self.original_feature_names,
            'encoded_features': self.feature_names
        }
        feature_file = output_path / 'feature_names.json'
        with open(feature_file, 'w') as f:
            json.dump(feature_names_data, f, indent=2)
        self.logger.info(f"✅ Feature names saved: {feature_file}")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': type(best_model).__name__,
            'train_date': datetime.now().isoformat(),
            'hyperparameter_tuning': {
                'enabled': self.tune_hyperparameters,
                'method': 'Optuna TPE' if self.tune_hyperparameters else None,
                'n_trials': self.n_trials if self.tune_hyperparameters else None
            },
            'data_shape': {
                'train': self.X_train.shape,
                'validation': self.X_val.shape,
                'test': self.X_test.shape
            },
            'metrics': {
                'accuracy': float(best_metrics['accuracy']),
                'precision': float(best_metrics['precision']),
                'recall': float(best_metrics['recall']),
                'f1_score': float(best_metrics['f1_score']),
                'roc_auc': float(best_metrics['roc_auc']) if best_metrics['roc_auc'] else None,
                'optimal_threshold': float(best_metrics.get('optimal_threshold', 0.5)),
                'business_cost_total': float(best_metrics['total_business_cost']),
                'business_cost_per_prediction': float(best_metrics['cost_per_prediction'])
            },
            'business_assumptions': {
                'cost_false_positive': self.cost_fp,
                'cost_false_negative': self.cost_fn
            }
        }
        
        metadata_file = output_path / 'model_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"✅ Metadata saved: {metadata_file}")
        
        return output_path


def train_all_models(
    tune_hyperparameters: bool = True,
    n_trials: int = 50,
    cost_fp: float = 100.0,
    cost_fn: float = 5000.0
) -> Tuple[ChurnModelTrainer, pd.DataFrame]:
    """
    Train all churn models and return trainer + comparison.
    
    Usage:
        trainer, comparison = train_all_models(tune_hyperparameters=True, n_trials=50)
    """
    
    # Initialize trainer
    trainer = ChurnModelTrainer(
        tune_hyperparameters=tune_hyperparameters,
        n_trials=n_trials,
        cost_fp=cost_fp,
        cost_fn=cost_fn
    )
    
    # Load data
    trainer.load_data()
    
    # Train models
    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_xgboost()
    
    # Compare models
    comparison = trainer.compare_models()
    
    # Get feature importance
    if 'XGBoost' in trainer.models:
        trainer.get_feature_importance('XGBoost', top_n=20)
    elif 'Random Forest' in trainer.models:
        trainer.get_feature_importance('Random Forest', top_n=20)
    
    # Save best model
    trainer.save_best_model()
    
    return trainer, comparison


if __name__ == "__main__":
    print("Elite Churn Model Training module ready.")
    print("\nUsage:")
    print("  trainer, comparison = train_all_models(tune_hyperparameters=True, n_trials=50)")
