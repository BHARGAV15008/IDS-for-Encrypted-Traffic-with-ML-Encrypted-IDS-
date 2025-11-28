"""
Improved Ensemble Classifier with:
- Class weighting for imbalanced data
- Soft voting mechanism
- Confidence-based prediction filtering
- Threshold optimization
- Calibration support
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, f1_score
import logging

logger = logging.getLogger(__name__)


class ImprovedEnsembleClassifier:
    """
    Improved ensemble classifier with multiple enhancements:
    1. Class weighting to handle imbalance
    2. Soft voting for better probability estimates
    3. Confidence thresholding for uncertain predictions
    4. Probability calibration
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        max_features: str = 'sqrt',
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str = 'balanced',
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        calibrate: bool = True
    ):
        """
        Initialize improved ensemble classifier
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            max_features: Features to consider at each split
            n_jobs: Number of parallel jobs
            random_state: Random seed
            class_weight: 'balanced' or 'balanced_subsample'
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            calibrate: Whether to calibrate probabilities
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.class_weight = class_weight
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.calibrate = calibrate
        
        # Initialize base models
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            oob_score=True
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=n_estimators // 2,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=random_state,
            subsample=0.8,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        
        self.calibrator = None
        self.threshold = 0.5
        self.class_weights = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'ImprovedEnsembleClassifier':
        """
        Fit ensemble models
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Validation features for calibration
            y_val: Validation labels for calibration
            
        Returns:
            self
        """
        logger.info("Fitting ensemble models...")
        
        # Calculate class weights for analysis
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self.class_weights = len(y) / (len(unique_classes) * class_counts)
        
        logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        logger.info(f"Class weights: {dict(zip(unique_classes, self.class_weights))}")
        
        # Fit Random Forest
        logger.info("Fitting Random Forest...")
        self.rf_model.fit(X, y)
        logger.info(f"Random Forest OOB Score: {self.rf_model.oob_score_:.4f}")
        
        # Fit Gradient Boosting
        logger.info("Fitting Gradient Boosting...")
        self.gb_model.fit(X, y)
        
        # Calibrate if validation data provided
        if self.calibrate and X_val is not None and y_val is not None:
            logger.info("Calibrating probabilities...")
            self.calibrator = CalibratedClassifierCV(
                self.rf_model,
                method='sigmoid',
                cv=5
            )
            self.calibrator.fit(X_val, y_val)
        
        self.is_fitted = True
        logger.info("Ensemble training complete")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            probabilities: (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get probabilities from both models
        rf_proba = self.rf_model.predict_proba(X)
        gb_proba = self.gb_model.predict_proba(X)
        
        # Soft voting: average probabilities
        ensemble_proba = (rf_proba + gb_proba) / 2
        
        # Calibrate if available
        if self.calibrator is not None:
            ensemble_proba = self.calibrator.predict_proba(X)
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features (n_samples, n_features)
            threshold: Decision threshold (for binary classification)
            
        Returns:
            predictions: (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        proba = self.predict_proba(X)
        
        if threshold is not None and proba.shape[1] == 2:
            # Binary classification with custom threshold
            predictions = (proba[:, 1] >= threshold).astype(int)
        else:
            # Multi-class: argmax
            predictions = np.argmax(proba, axis=1)
        
        return predictions
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence scores and uncertainty handling
        
        Args:
            X: Features (n_samples, n_features)
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            predictions: (n_samples,) - class predictions (-1 for uncertain)
            confidences: (n_samples,) - confidence scores
            probabilities: (n_samples, n_classes) - class probabilities
        """
        proba = self.predict_proba(X)
        
        # Get predictions and confidences
        predictions = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        
        # Mark uncertain predictions
        uncertain_mask = confidences < confidence_threshold
        predictions[uncertain_mask] = -1
        
        return predictions, confidences, proba
    
    def optimize_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal decision threshold for binary classification
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: 'f1', 'precision', 'recall', or 'security'
            
        Returns:
            optimal_threshold: Best threshold value
            metrics: Dictionary of metrics at optimal threshold
        """
        if y_val.shape[0] == 0:
            logger.warning("Empty validation set, using default threshold 0.5")
            return 0.5, {}
        
        proba = self.predict_proba(X_val)
        
        if proba.shape[1] != 2:
            logger.warning("Threshold optimization only for binary classification")
            return 0.5, {}
        
        # Get probabilities for positive class
        y_proba = proba[:, 1]
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        
        # Evaluate different metrics
        best_threshold = 0.5
        best_score = 0.0
        best_metrics = {}
        
        for threshold in np.linspace(0, 1, 101):
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_val, y_pred, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_val, y_pred, zero_division=0)
            elif metric == 'security':
                # Prioritize recall (catch attacks) over precision
                from sklearn.metrics import confusion_matrix
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                # 70% weight on recall, 30% on specificity
                score = 0.7 * recall + 0.3 * specificity
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
                # Calculate metrics at this threshold
                from sklearn.metrics import confusion_matrix, precision_score, recall_score
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                
                best_metrics = {
                    'threshold': threshold,
                    'score': score,
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1': f1_score(y_val, y_pred, zero_division=0),
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
                    'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
                }
        
        self.threshold = best_threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.4f}")
        logger.info(f"Metrics at optimal threshold: {best_metrics}")
        
        return best_threshold, best_metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from Random Forest
        
        Returns:
            importance: Dictionary of feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importances = self.rf_model.feature_importances_
        
        return {f'feature_{i}': imp for i, imp in enumerate(importances)}
    
    def get_model_info(self) -> Dict:
        """Get information about ensemble models"""
        return {
            'rf_n_estimators': self.rf_model.n_estimators,
            'rf_max_depth': self.rf_model.max_depth,
            'rf_oob_score': self.rf_model.oob_score_ if hasattr(self.rf_model, 'oob_score_') else None,
            'gb_n_estimators': self.gb_model.n_estimators,
            'gb_learning_rate': self.gb_model.learning_rate,
            'threshold': self.threshold,
            'calibrated': self.calibrator is not None,
            'class_weights': self.class_weights.tolist() if self.class_weights is not None else None
        }


class EnsembleEvaluator:
    """Evaluate ensemble classifier performance"""
    
    def __init__(self, ensemble: ImprovedEnsembleClassifier):
        self.ensemble = ensemble
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Decision threshold
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, roc_auc_score, classification_report
        )
        
        # Get predictions
        y_pred = self.ensemble.predict(X_test, threshold=threshold)
        y_proba = self.ensemble.predict_proba(X_test)
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Binary classification metrics
        if y_proba.shape[1] == 2:
            metrics['auc_roc'] = roc_auc_score(y_test, y_proba[:, 1])
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics['tp'] = int(tp)
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
