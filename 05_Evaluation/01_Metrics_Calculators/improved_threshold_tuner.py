"""
Improved Threshold Tuner for Binary Classification

Finds optimal decision thresholds that balance:
- Detection rate (recall for attacks)
- False positive rate (specificity for benign)
- Precision-recall trade-off
- Security requirements
"""

from typing import Tuple, Dict, Optional, List
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, f1_score, confusion_matrix,
    precision_score, recall_score, roc_curve, auc
)
import logging

logger = logging.getLogger(__name__)


class ImprovedThresholdTuner:
    """
    Advanced threshold tuning for binary classification
    
    Supports multiple optimization objectives:
    - F1-score maximization
    - Precision-recall balance
    - Security-focused (maximize detection, minimize FP)
    - Custom weighted objectives
    """
    
    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Initialize threshold tuner
        
        Args:
            y_true: True binary labels (0 or 1)
            y_proba: Predicted probabilities for positive class
        """
        self.y_true = y_true
        self.y_proba = y_proba
        
        # Calculate precision-recall curve
        self.precisions, self.recalls, self.thresholds = precision_recall_curve(y_true, y_proba)
        
        # Ensure thresholds array has same length as precisions and recalls
        if len(self.thresholds) == len(self.precisions) - 1:
            self.thresholds = np.append(self.thresholds, 1.0)
        
        # Calculate ROC curve
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(y_true, y_proba)
        self.roc_auc = auc(self.fpr, self.tpr)
        
        logger.info(f"Threshold tuner initialized with {len(y_true)} samples")
        logger.info(f"Positive class ratio: {np.sum(y_true) / len(y_true):.4f}")
        logger.info(f"ROC AUC: {self.roc_auc:.4f}")
    
    def find_optimal_threshold_f1(self) -> Tuple[float, float]:
        """
        Find threshold that maximizes F1-score
        
        Returns:
            threshold: Optimal threshold
            f1_score: F1-score at optimal threshold
        """
        f1_scores = 2 * (self.precisions * self.recalls) / (self.precisions + self.recalls + 1e-10)
        f1_scores = np.nan_to_num(f1_scores)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = self.thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        logger.info(f"F1-optimal threshold: {optimal_threshold:.4f}, F1: {optimal_f1:.4f}")
        
        return optimal_threshold, optimal_f1
    
    def find_optimal_threshold_for_recall(self, target_recall: float) -> Tuple[Optional[float], float]:
        """
        Find threshold that achieves at least target recall
        
        Useful for security: "catch at least 95% of attacks"
        
        Args:
            target_recall: Minimum desired recall (e.g., 0.95)
            
        Returns:
            threshold: Threshold achieving target recall (or None if impossible)
            actual_recall: Actual recall achieved
        """
        valid_indices = np.where(self.recalls >= target_recall)[0]
        
        if len(valid_indices) == 0:
            logger.warning(f"Cannot achieve recall of {target_recall}. Max: {np.max(self.recalls):.4f}")
            return None, np.max(self.recalls)
        
        # Among valid thresholds, choose one with highest precision
        best_idx = valid_indices[0]
        threshold = self.thresholds[best_idx]
        actual_recall = self.recalls[best_idx]
        precision = self.precisions[best_idx]
        
        logger.info(f"Threshold for recall >= {target_recall}: {threshold:.4f}")
        logger.info(f"Actual recall: {actual_recall:.4f}, Precision: {precision:.4f}")
        
        return threshold, actual_recall
    
    def find_optimal_threshold_for_precision(self, target_precision: float) -> Tuple[Optional[float], float]:
        """
        Find threshold that achieves at least target precision
        
        Useful for reducing false positives: "at least 99% of alerts are real"
        
        Args:
            target_precision: Minimum desired precision (e.g., 0.99)
            
        Returns:
            threshold: Threshold achieving target precision (or None if impossible)
            actual_precision: Actual precision achieved
        """
        valid_indices = np.where(self.precisions >= target_precision)[0]
        
        if len(valid_indices) == 0:
            logger.warning(f"Cannot achieve precision of {target_precision}. Max: {np.max(self.precisions):.4f}")
            return None, np.max(self.precisions)
        
        # Among valid thresholds, choose one with highest recall
        best_idx = valid_indices[-1]
        threshold = self.thresholds[best_idx]
        actual_precision = self.precisions[best_idx]
        recall = self.recalls[best_idx]
        
        logger.info(f"Threshold for precision >= {target_precision}: {threshold:.4f}")
        logger.info(f"Actual precision: {actual_precision:.4f}, Recall: {recall:.4f}")
        
        return threshold, actual_precision
    
    def find_optimal_threshold_for_fpr(self, target_fpr: float) -> Tuple[Optional[float], float]:
        """
        Find threshold that achieves at most target false positive rate
        
        Args:
            target_fpr: Maximum desired FPR (e.g., 0.02 for 2%)
            
        Returns:
            threshold: Threshold achieving target FPR (or None if impossible)
            actual_fpr: Actual FPR achieved
        """
        valid_indices = np.where(self.fpr <= target_fpr)[0]
        
        if len(valid_indices) == 0:
            logger.warning(f"Cannot achieve FPR of {target_fpr}. Min: {np.min(self.fpr):.4f}")
            return None, np.min(self.fpr)
        
        # Among valid thresholds, choose one with highest TPR
        best_idx = valid_indices[-1]
        threshold = self.roc_thresholds[best_idx]
        actual_fpr = self.fpr[best_idx]
        tpr = self.tpr[best_idx]
        
        logger.info(f"Threshold for FPR <= {target_fpr}: {threshold:.4f}")
        logger.info(f"Actual FPR: {actual_fpr:.4f}, TPR: {tpr:.4f}")
        
        return threshold, actual_fpr
    
    def find_optimal_threshold_security(
        self,
        target_detection_rate: float = 0.95,
        target_fpr: float = 0.02,
        detection_weight: float = 0.7,
        specificity_weight: float = 0.3
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find threshold optimized for security requirements
        
        Balances:
        - Detection rate (catch attacks)
        - False positive rate (minimize false alarms)
        
        Args:
            target_detection_rate: Target detection rate (e.g., 0.95)
            target_fpr: Target false positive rate (e.g., 0.02)
            detection_weight: Weight for detection rate
            specificity_weight: Weight for specificity
            
        Returns:
            optimal_threshold: Best threshold
            metrics: Dictionary of metrics at optimal threshold
        """
        best_threshold = 0.5
        best_score = 0.0
        best_metrics = {}
        
        # Evaluate all thresholds
        for threshold in np.linspace(0, 1, 101):
            y_pred = (self.y_proba >= threshold).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
            
            detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Security score: weighted combination
            security_score = (
                detection_weight * detection_rate +
                specificity_weight * specificity
            )
            
            # Penalty for not meeting targets
            if detection_rate < target_detection_rate:
                security_score *= 0.5
            if fpr > target_fpr:
                security_score *= 0.5
            
            if security_score > best_score:
                best_score = security_score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'security_score': security_score,
                    'detection_rate': detection_rate,
                    'specificity': specificity,
                    'fpr': fpr,
                    'precision': precision,
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'meets_detection_target': detection_rate >= target_detection_rate,
                    'meets_fpr_target': fpr <= target_fpr
                }
        
        logger.info(f"Security-optimal threshold: {best_threshold:.4f}")
        logger.info(f"Metrics: {best_metrics}")
        
        return best_threshold, best_metrics
    
    def find_optimal_threshold_custom(
        self,
        objective_fn
    ) -> Tuple[float, float]:
        """
        Find threshold using custom objective function
        
        Args:
            objective_fn: Function(y_true, y_pred) -> score
            
        Returns:
            optimal_threshold: Best threshold
            optimal_score: Score at optimal threshold
        """
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in np.linspace(0, 1, 101):
            y_pred = (self.y_proba >= threshold).astype(int)
            score = objective_fn(self.y_true, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Custom-optimal threshold: {best_threshold:.4f}, Score: {best_score:.4f}")
        
        return best_threshold, best_score
    
    def get_metrics_at_threshold(self, threshold: float) -> Dict[str, float]:
        """
        Get all metrics at a specific threshold
        
        Args:
            threshold: Decision threshold
            
        Returns:
            metrics: Dictionary of metrics
        """
        y_pred = (self.y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        
        metrics = {
            'threshold': threshold,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1': f1_score(self.y_true, y_pred),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'mcc': self._matthews_corrcoef(tp, tn, fp, fn)
        }
        
        return metrics
    
    def plot_threshold_analysis(self, save_path: Optional[str] = None):
        """
        Plot precision-recall and ROC curves with optimal thresholds
        
        Args:
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Precision-Recall curve
            axes[0].plot(self.recalls, self.precisions, 'b-', linewidth=2)
            axes[0].set_xlabel('Recall', fontsize=12)
            axes[0].set_ylabel('Precision', fontsize=12)
            axes[0].set_title('Precision-Recall Curve', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            
            # ROC curve
            axes[1].plot(self.fpr, self.tpr, 'b-', linewidth=2, label=f'ROC (AUC={self.roc_auc:.3f})')
            axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
            axes[1].set_xlabel('False Positive Rate', fontsize=12)
            axes[1].set_ylabel('True Positive Rate', fontsize=12)
            axes[1].set_title('ROC Curve', fontsize=14)
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    @staticmethod
    def _matthews_corrcoef(tp: int, tn: int, fp: int, fn: int) -> float:
        """Calculate Matthews Correlation Coefficient"""
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    objective: str = 'security',
    **kwargs
) -> Tuple[float, Dict[str, float]]:
    """
    Convenience function to find best threshold
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        objective: 'f1', 'precision', 'recall', 'fpr', or 'security'
        **kwargs: Additional arguments for specific objectives
        
    Returns:
        threshold: Optimal threshold
        metrics: Metrics at optimal threshold
    """
    tuner = ImprovedThresholdTuner(y_true, y_proba)
    
    if objective == 'f1':
        threshold, score = tuner.find_optimal_threshold_f1()
        metrics = tuner.get_metrics_at_threshold(threshold)
    elif objective == 'precision':
        target = kwargs.get('target_precision', 0.95)
        threshold, _ = tuner.find_optimal_threshold_for_precision(target)
        metrics = tuner.get_metrics_at_threshold(threshold)
    elif objective == 'recall':
        target = kwargs.get('target_recall', 0.95)
        threshold, _ = tuner.find_optimal_threshold_for_recall(target)
        metrics = tuner.get_metrics_at_threshold(threshold)
    elif objective == 'fpr':
        target = kwargs.get('target_fpr', 0.02)
        threshold, _ = tuner.find_optimal_threshold_for_fpr(target)
        metrics = tuner.get_metrics_at_threshold(threshold)
    elif objective == 'security':
        threshold, metrics = tuner.find_optimal_threshold_security(**kwargs)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    return threshold, metrics
