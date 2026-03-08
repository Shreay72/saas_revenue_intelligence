"""
Elite Model Evaluation Module with Cost-Sensitive Business Metrics
SaaS Revenue Risk & Retention Intelligence System
Week 2 - Production-grade evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import logging
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Elite model evaluation with business cost analysis.
    
    Features:
    - Standard ML metrics
    - Threshold optimization
    - Calibration analysis
    - Cost-sensitive business evaluation
    """
    
    def __init__(
        self, 
        cost_fp: float = 100.0,  # Cost of unnecessary retention offer
        cost_fn: float = 5000.0  # Cost of lost customer (avg LTV)
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
    
    def evaluate_classification(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray] = None, 
        model_name: str = "Model"
    ) -> Dict:
        """Comprehensive classification evaluation with business metrics."""
        
        metrics = {}
        
        # Standard ML metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Probabilistic metrics
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = None
            metrics['avg_precision'] = None
            metrics['brier_score'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        # Business metrics
        metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        metrics['false_positive_rate'] = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        # BUSINESS COST ANALYSIS
        metrics['cost_fp_total'] = metrics['fp'] * self.cost_fp
        metrics['cost_fn_total'] = metrics['fn'] * self.cost_fn
        metrics['total_business_cost'] = metrics['cost_fp_total'] + metrics['cost_fn_total']
        metrics['cost_per_prediction'] = metrics['total_business_cost'] / len(y_true) if len(y_true) > 0 else 0
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred)
        
        return metrics
    
    def optimize_threshold(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        metric: str = 'business_cost'
    ) -> Tuple[float, Dict]:
        """
        Find optimal classification threshold.
        
        Metrics:
        - 'f1': Optimize F1-score
        - 'precision': Optimize precision
        - 'recall': Optimize recall
        - 'youden': Optimize Youden's J statistic
        - 'business_cost': Minimize business cost (RECOMMENDED)
        """
        
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_score = float('inf') if metric == 'business_cost' else 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'business_cost':
                # Minimize business cost
                cm = confusion_matrix(y_true, y_pred)
                cost = (cm[0, 1] * self.cost_fp) + (cm[1, 0] * self.cost_fn)
                score = cost  # Use actual cost for comparison
                actual_score = cost
                
                # Update best (lower cost is better)
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = self._get_metrics_at_threshold(
                        y_true, y_pred_proba, threshold, metric, actual_score
                    )
            else:
                if metric == 'f1':
                    score = f1_score(y_true, y_pred, zero_division=0)
                elif metric == 'precision':
                    score = precision_score(y_true, y_pred, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_true, y_pred, zero_division=0)
                elif metric == 'youden':
                    cm = confusion_matrix(y_true, y_pred)
                    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
                    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
                    score = sensitivity + specificity - 1
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                actual_score = score
                
                # Update best (higher is better for other metrics)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = self._get_metrics_at_threshold(
                        y_true, y_pred_proba, threshold, metric, actual_score
                    )
        
        # Log result
        if best_metrics:
            self.logger.info(f"✅ Optimal threshold: {best_threshold:.3f} ({metric}: {best_metrics.get('optimized_score', 0):.2f})")
        
        return best_threshold, best_metrics
    
    def _get_metrics_at_threshold(self, y_true, y_pred_proba, threshold, metric_name, score):
        """Helper to get metrics at specific threshold."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'business_cost': (cm[0, 1] * self.cost_fp) + (cm[1, 0] * self.cost_fn),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'optimized_metric': metric_name,
            'optimized_score': score
        }
    
    def plot_business_cost_curve(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        model_name: str = "Model"
    ):
        """Plot business cost across different thresholds."""
        
        thresholds = np.arange(0.0, 1.01, 0.01)
        costs = []
        fp_counts = []
        fn_counts = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            cost = (cm[0, 1] * self.cost_fp) + (cm[1, 0] * self.cost_fn)
            costs.append(cost)
            fp_counts.append(cm[0, 1])
            fn_counts.append(cm[1, 0])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Cost curve
        ax1.plot(thresholds, costs, linewidth=2, color='red')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Total Business Cost ($)')
        ax1.set_title(f'Business Cost vs Threshold - {model_name}')
        ax1.grid(alpha=0.3)
        
        # Optimal point
        optimal_idx = np.argmin(costs)
        ax1.axvline(thresholds[optimal_idx], color='green', linestyle='--', 
                   label=f'Optimal: {thresholds[optimal_idx]:.2f}')
        ax1.legend()
        
        # Error breakdown
        ax2.plot(thresholds, fp_counts, label='False Positives', linewidth=2)
        ax2.plot(thresholds, fn_counts, label='False Negatives', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Error Count')
        ax2.set_title('FP vs FN Trade-off')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_threshold_analysis(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        model_name: str = "Model"
    ):
        """Plot precision, recall, F1 across thresholds."""
        
        thresholds = np.arange(0.0, 1.01, 0.01)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Threshold Analysis - {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_calibration_curve(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        n_bins: int = 10
    ):
        """Plot calibration curve (reliability diagram)."""
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )
        
        brier = brier_score_loss(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
                 label=f'{model_name} (Brier: {brier:.4f})')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve - {model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def print_evaluation_report(self, metrics: Dict, model_name: str = "Model"):
        """Print comprehensive evaluation report."""
        
        print("="*80)
        print(f"📊 EVALUATION REPORT: {model_name}")
        print("="*80)
        
        print(f"\n🎯 ML Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        
        if metrics['roc_auc'] is not None:
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"   Avg Precision: {metrics['avg_precision']:.4f}")
            print(f"   Brier Score:   {metrics['brier_score']:.4f}")
        
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Negatives:  {metrics['tn']}")
        print(f"   False Positives: {metrics['fp']}")
        print(f"   False Negatives: {metrics['fn']}")
        print(f"   True Positives:  {metrics['tp']}")
        
        print(f"\n💰 BUSINESS COST ANALYSIS:")
        print(f"   Cost per FP (retention offer): ${self.cost_fp:,.2f}")
        print(f"   Cost per FN (lost customer):   ${self.cost_fn:,.2f}")
        print(f"   Total FP Cost: ${metrics['cost_fp_total']:,.2f}")
        print(f"   Total FN Cost: ${metrics['cost_fn_total']:,.2f}")
        print(f"   TOTAL COST:    ${metrics['total_business_cost']:,.2f}")
        print(f"   Cost per prediction: ${metrics['cost_per_prediction']:.2f}")
        
        print("="*80 + "\n")
    
    def create_risk_levels(self, y_pred_proba: np.ndarray) -> pd.DataFrame:
        """Create risk level categories."""
        
        risk_levels = []
        for prob in y_pred_proba:
            if prob < 0.3:
                risk_levels.append('LOW')
            elif prob < 0.6:
                risk_levels.append('MEDIUM')
            else:
                risk_levels.append('HIGH')
        
        return pd.DataFrame({
            'churn_probability': y_pred_proba,
            'risk_level': risk_levels
        })
    
    def compare_models(self, results_dict: Dict) -> pd.DataFrame:
        """Compare multiple models."""
        
        comparison = []
        for model_name, metrics in results_dict.items():
            comparison.append({
                'Model': model_name,
                'ROC-AUC': metrics.get('roc_auc', 0),
                'F1-Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Business Cost': metrics.get('total_business_cost', 0),
                'Brier': metrics.get('brier_score', None)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str = "Model"):
        """Plot confusion matrix."""
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Retained', 'Churned'],
                    yticklabels=['Retained', 'Churned'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       model_name: str = "Model"):
        """Plot ROC curve."""
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                   model_name: str = "Model",
                   cost_fp: float = 100.0, cost_fn: float = 5000.0) -> Dict:
    """
    Evaluate model with business cost analysis.
    
    Parameters:
    -----------
    cost_fp : float
        Cost of false positive (unnecessary retention offer)
    cost_fn : float
        Cost of false negative (lost customer, typically avg LTV)
    """
    
    evaluator = ModelEvaluator(cost_fp=cost_fp, cost_fn=cost_fn)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Optimize threshold for business cost
        optimal_threshold, threshold_metrics = evaluator.optimize_threshold(
            y_test, y_pred_proba, metric='business_cost'
        )
        
        # Re-predict with optimal threshold
        y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    else:
        y_pred_proba = None
        optimal_threshold = 0.5
        y_pred_optimized = y_pred
    
    # Evaluate
    metrics = evaluator.evaluate_classification(y_test, y_pred, y_pred_proba, model_name)
    
    # Add optimized metrics
    if y_pred_proba is not None:
        metrics['optimal_threshold'] = optimal_threshold
        metrics['optimized_precision'] = precision_score(y_test, y_pred_optimized, zero_division=0)
        metrics['optimized_recall'] = recall_score(y_test, y_pred_optimized, zero_division=0)
        metrics['optimized_f1'] = f1_score(y_test, y_pred_optimized, zero_division=0)
        
        # Optimized business cost
        cm_opt = confusion_matrix(y_test, y_pred_optimized)
        metrics['optimized_business_cost'] = (cm_opt[0, 1] * cost_fp) + (cm_opt[1, 0] * cost_fn)
    
    evaluator.print_evaluation_report(metrics, model_name)
    
    return metrics


if __name__ == "__main__":
    print("Elite Model Evaluation module ready.")
