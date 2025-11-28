"""
Threshold Tuning Module
Tools for optimizing the decision threshold for binary classifiers to balance
precision and recall, or to meet specific performance targets (e.g., a minimum recall).
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ThresholdTuner:
    """
    Optimizes the decision threshold for a binary classifier.
    
    Instead of using a default 0.5 threshold, this class finds an optimal
    threshold on a validation set to balance trade-offs like precision vs. recall.
    """
    
    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Args:
            y_true (np.ndarray): True binary labels.
            y_proba (np.ndarray): Predicted probabilities for the positive class.
        """
        if y_proba.ndim == 2:
            # Assuming y_proba is (n_samples, n_classes), take the positive class probability
            y_proba = y_proba[:, 1]
            
        self.y_true = y_true
        self.y_proba = y_proba
        self.precisions, self.recalls, self.thresholds = precision_recall_curve(y_true, y_proba)
        
        # The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold
        # We append a value to make array lengths consistent
        if len(self.thresholds) == len(self.precisions) - 1:
            self.thresholds = np.append(self.thresholds, 1.0)

    def find_best_threshold_for_f1(self) -> Tuple[float, float]:
        """
        Find the threshold that maximizes the F1-score.
        
        Returns:
            Tuple[float, float]: The best threshold and the corresponding F1-score.
        """
        f1_scores = (2 * self.precisions * self.recalls) / (self.precisions + self.recalls)
        f1_scores = np.nan_to_num(f1_scores) # Handle division by zero
        
        best_idx = np.argmax(f1_scores)
        best_threshold = self.thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        logger.info(f"Best threshold for F1-score: {best_threshold:.4f} (F1: {best_f1:.4f})")
        return best_threshold, best_f1

    def find_threshold_for_recall(self, target_recall: float) -> Tuple[Optional[float], float]:
        """
        Find the threshold that achieves at least the target recall.
        
        This is useful when you have a strict requirement for the detection rate,
        e.g., "must detect at least 99.5% of all attacks".
        
        Args:
            target_recall (float): The minimum desired recall (e.g., 0.995).
            
        Returns:
            Tuple[Optional[float], float]: The corresponding threshold and the actual recall achieved.
                                           Returns None if the target cannot be met.
        """
        # Find the first index where recall is >= target_recall
        valid_indices = np.where(self.recalls >= target_recall)[0]
        
        if len(valid_indices) == 0:
            logger.warning(f"Could not find a threshold to achieve recall of {target_recall}. "
                           f"Max achievable recall is {np.max(self.recalls):.4f}.")
            return None, np.max(self.recalls)
            
        # We want the threshold that gives the highest precision for the target recall
        # which corresponds to the lowest threshold (highest index in precision_recall_curve output)
        best_idx = valid_indices[0]
        threshold = self.thresholds[best_idx]
        actual_recall = self.recalls[best_idx]
        
        logger.info(f"Threshold for recall >= {target_recall}: {threshold:.4f} (Actual Recall: {actual_recall:.4f})")
        return threshold, actual_recall

    def find_threshold_for_precision(self, target_precision: float) -> Tuple[Optional[float], float]:
        """
        Find the threshold that achieves at least the target precision.
        
        This is useful for minimizing false positives, e.g., "at least 99% of alerts must be real".
        
        Args:
            target_precision (float): The minimum desired precision (e.g., 0.99).
            
        Returns:
            Tuple[Optional[float], float]: The corresponding threshold and the actual precision achieved.
                                           Returns None if the target cannot be met.
        """
        valid_indices = np.where(self.precisions >= target_precision)[0]
        
        if len(valid_indices) == 0:
            logger.warning(f"Could not find a threshold to achieve precision of {target_precision}. "
                           f"Max achievable precision is {np.max(self.precisions):.4f}.")
            return None, np.max(self.precisions)
            
        # We want the threshold that gives the highest recall for that precision
        best_idx = valid_indices[-1]
        threshold = self.thresholds[best_idx]
        actual_precision = self.precisions[best_idx]
        
        logger.info(f"Threshold for precision >= {target_precision}: {threshold:.4f} (Actual Precision: {actual_precision:.4f})")
        return threshold, actual_precision

    @staticmethod
    def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply a custom threshold to probability scores to get predictions.
        
        Args:
            y_proba (np.ndarray): Predicted probabilities for the positive class.
            threshold (float): The decision threshold.
            
        Returns:
            np.ndarray: The final binary predictions.
        """
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
            
        return (y_proba >= threshold).astype(int)
