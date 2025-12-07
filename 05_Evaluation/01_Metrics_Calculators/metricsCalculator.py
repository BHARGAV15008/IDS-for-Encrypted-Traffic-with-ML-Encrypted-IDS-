"""
Metrics Calculator Module
Comprehensive evaluation metrics for IDS performance
Includes detection rate, FPR, precision, recall, F1-score, AUC-ROC, and custom IDS metrics
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics for IDS"""
    
    def __init__(self, numClasses: int = 2, classNames: Optional[List[str]] = None):
        """
        Args:
            numClasses: Number of classes
            classNames: Names of classes
        """
        self.numClasses = numClasses
        self.classNames = classNames if classNames else [f'Class_{i}' for i in range(numClasses)]
        
    def calculateAll(
        self,
        yTrue: np.ndarray,
        yPred: np.ndarray,
        yProba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            yTrue: True labels
            yPred: Predicted labels
            yProba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self.calculateBasicMetrics(yTrue, yPred))
        
        # Confusion matrix metrics
        metrics.update(self.calculateConfusionMatrixMetrics(yTrue, yPred))
        
        # IDS-specific metrics
        metrics.update(self.calculateIDSMetrics(yTrue, yPred))
        
        # ROC and AUC metrics (if probabilities provided)
        if yProba is not None:
            metrics.update(self.calculateROCMetrics(yTrue, yProba))
            
        logger.info(f"Calculated {len(metrics)} metrics")
        
        return metrics
        
    def calculateBasicMetrics(
        self,
        yTrue: np.ndarray,
        yPred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(yTrue, yPred),
            'precision_macro': precision_score(yTrue, yPred, average='macro', zero_division=0),
            'precision_weighted': precision_score(yTrue, yPred, average='weighted', zero_division=0),
            'recall_macro': recall_score(yTrue, yPred, average='macro', zero_division=0),
            'recall_weighted': recall_score(yTrue, yPred, average='weighted', zero_division=0),
            'f1_macro': f1_score(yTrue, yPred, average='macro', zero_division=0),
            'f1_weighted': f1_score(yTrue, yPred, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        precisions = precision_score(yTrue, yPred, average=None, zero_division=0)
        recalls = recall_score(yTrue, yPred, average=None, zero_division=0)
        f1Scores = f1_score(yTrue, yPred, average=None, zero_division=0)
        
        for i, className in enumerate(self.classNames):
            if i < len(precisions):
                metrics[f'precision_{className}'] = precisions[i]
                metrics[f'recall_{className}'] = recalls[i]
                metrics[f'f1_{className}'] = f1Scores[i]
                
        return metrics
        
    def calculateConfusionMatrixMetrics(
        self,
        yTrue: np.ndarray,
        yPred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics from confusion matrix"""
        cm = confusion_matrix(yTrue, yPred)
        
        # For binary classification
        if self.numClasses == 2:
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
            }
        else:
            # Multi-class metrics
            metrics = {}
            for i in range(min(self.numClasses, cm.shape[0])):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                
                className = self.classNames[i]
                metrics[f'tp_{className}'] = int(tp)
                metrics[f'fp_{className}'] = int(fp)
                metrics[f'fn_{className}'] = int(fn)
                metrics[f'tn_{className}'] = int(tn)
                
        return metrics
        
    def calculateIDSMetrics(
        self,
        yTrue: np.ndarray,
        yPred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate IDS-specific metrics"""
        # Assuming binary: 0 = benign, 1 = attack
        if self.numClasses == 2:
            cm = confusion_matrix(yTrue, yPred)
            tn, fp, fn, tp = cm.ravel()
            
            # Detection Rate (True Positive Rate / Recall for attack class)
            detectionRate = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False Alarm Rate (False Positive Rate)
            falseAlarmRate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Miss Rate (False Negative Rate)
            missRate = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Precision for attack detection
            attackPrecision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Specificity (True Negative Rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Matthews Correlation Coefficient
            mcc = self._calculateMCC(tp, tn, fp, fn)
            
            # G-Mean (geometric mean of sensitivity and specificity)
            sensitivity = detectionRate
            gMean = np.sqrt(sensitivity * specificity)
            
            metrics = {
                'detection_rate': detectionRate,
                'false_alarm_rate': falseAlarmRate,
                'miss_rate': missRate,
                'attack_precision': attackPrecision,
                'specificity': specificity,
                'mcc': mcc,
                'g_mean': gMean
            }
        else:
            # Multi-class IDS metrics
            metrics = {}
            cm = confusion_matrix(yTrue, yPred)
            
            for i in range(min(self.numClasses, cm.shape[0])):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                
                className = self.classNames[i]
                detectionRate = tp / (tp + fn) if (tp + fn) > 0 else 0
                falseAlarmRate = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                metrics[f'detection_rate_{className}'] = detectionRate
                metrics[f'false_alarm_rate_{className}'] = falseAlarmRate
                
        return metrics
        
    def calculateROCMetrics(
        self,
        yTrue: np.ndarray,
        yProba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate ROC and AUC metrics"""
        metrics = {}
        
        try:
            if self.numClasses == 2:
                # Binary classification
                if yProba.ndim == 2:
                    yProba = yProba[:, 1]  # Probability of positive class
                    
                # ROC AUC
                aucScore = roc_auc_score(yTrue, yProba)
                metrics['auc_roc'] = aucScore
                
                # Average Precision (PR AUC)
                apScore = average_precision_score(yTrue, yProba)
                metrics['average_precision'] = apScore
                
            else:
                # Multi-class
                aucScores = roc_auc_score(
                    yTrue, yProba,
                    multi_class='ovr',
                    average=None
                )
                
                for i, className in enumerate(self.classNames):
                    if i < len(aucScores):
                        metrics[f'auc_roc_{className}'] = aucScores[i]
                        
                # Macro and weighted averages
                metrics['auc_roc_macro'] = roc_auc_score(yTrue, yProba, multi_class='ovr', average='macro')
                metrics['auc_roc_weighted'] = roc_auc_score(yTrue, yProba, multi_class='ovr', average='weighted')
                
        except Exception as e:
            logger.warning(f"Could not calculate ROC metrics: {str(e)}")
            
        return metrics
        
    def _calculateMCC(
        self,
        tp: int,
        tn: int,
        fp: int,
        fn: int
    ) -> float:
        """Calculate Matthews Correlation Coefficient"""
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        return numerator / denominator
        
    def getClassificationReport(
        self,
        yTrue: np.ndarray,
        yPred: np.ndarray
    ) -> str:
        """Get detailed classification report"""
        return classification_report(
            yTrue, yPred,
            target_names=self.classNames,
            zero_division=0
        )
        
    def getConfusionMatrix(
        self,
        yTrue: np.ndarray,
        yPred: np.ndarray
    ) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(yTrue, yPred)


class AdversarialMetricsCalculator:
    """Calculate metrics for adversarial robustness evaluation"""
    
    def __init__(self):
        self.metricsCalc = MetricsCalculator()
        
    def calculateRobustnessMetrics(
        self,
        yTrue: np.ndarray,
        yPredClean: np.ndarray,
        yPredAdv: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate adversarial robustness metrics
        
        Args:
            yTrue: True labels
            yPredClean: Predictions on clean samples
            yPredAdv: Predictions on adversarial samples
            
        Returns:
            Robustness metrics
        """
        # Clean accuracy
        cleanAcc = accuracy_score(yTrue, yPredClean)
        
        # Adversarial accuracy
        advAcc = accuracy_score(yTrue, yPredAdv)
        
        # Robustness score (ratio of adversarial to clean accuracy)
        robustnessScore = advAcc / cleanAcc if cleanAcc > 0 else 0
        
        # Attack success rate (percentage of clean correct that become incorrect)
        cleanCorrect = (yTrue == yPredClean)
        advIncorrect = (yTrue != yPredAdv)
        attackSuccess = np.logical_and(cleanCorrect, advIncorrect).sum() / cleanCorrect.sum() if cleanCorrect.sum() > 0 else 0
        
        # Robustness degradation
        robustnessDegradation = cleanAcc - advAcc
        
        metrics = {
            'clean_accuracy': cleanAcc,
            'adversarial_accuracy': advAcc,
            'robustness_score': robustnessScore,
            'attack_success_rate': attackSuccess,
            'robustness_degradation': robustnessDegradation
        }
        
        return metrics
        
    def calculateCertifiedRobustness(
        self,
        model: torch.nn.Module,
        xTest: torch.Tensor,
        yTest: torch.Tensor,
        epsilon: float = 0.01,
        numSamples: int = 100
    ) -> float:
        """
        Calculate certified robustness using randomized smoothing
        
        Args:
            model: Model to evaluate
            xTest: Test samples
            yTest: Test labels
            epsilon: Perturbation radius
            numSamples: Number of samples for smoothing
            
        Returns:
            Certified accuracy
        """
        model.eval()
        certifiedCorrect = 0
        total = len(xTest)
        
        with torch.no_grad():
            for i in range(total):
                x = xTest[i:i+1]
                y = yTest[i]
                
                # Sample predictions
                predictions = []
                for _ in range(numSamples):
                    noise = torch.randn_like(x) * epsilon
                    xNoisy = x + noise
                    pred = model(xNoisy).argmax(dim=1)
                    predictions.append(pred.item())
                    
                # Majority vote
                predCounts = np.bincount(predictions)
                majorityPred = np.argmax(predCounts)
                
                if majorityPred == y.item():
                    certifiedCorrect += 1
                    
        certifiedAcc = certifiedCorrect / total
        return certifiedAcc


class CrossDatasetMetricsCalculator:
    """Calculate metrics for cross-dataset generalization"""
    
    def __init__(self):
        self.metricsCalc = MetricsCalculator()
        
    def calculateGeneralizationMetrics(
        self,
        sourceMetrics: Dict[str, float],
        targetMetrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate generalization metrics across datasets
        
        Args:
            sourceMetrics: Metrics on source dataset
            targetMetrics: Metrics on target dataset
            
        Returns:
            Generalization metrics
        """
        # Performance drop
        accDrop = sourceMetrics['accuracy'] - targetMetrics['accuracy']
        f1Drop = sourceMetrics['f1_macro'] - targetMetrics['f1_macro']
        
        # Generalization score (ratio of target to source performance)
        genScore = targetMetrics['accuracy'] / sourceMetrics['accuracy'] if sourceMetrics['accuracy'] > 0 else 0
        
        # Consistency score (how consistent are the metrics)
        consistency = 1 - abs(accDrop) / sourceMetrics['accuracy'] if sourceMetrics['accuracy'] > 0 else 0
        
        metrics = {
            'accuracy_drop': accDrop,
            'f1_drop': f1Drop,
            'generalization_score': genScore,
            'consistency_score': consistency,
            'source_accuracy': sourceMetrics['accuracy'],
            'target_accuracy': targetMetrics['accuracy']
        }
        
        return metrics


class RealTimeMetricsCalculator:
    """Calculate real-time performance metrics"""
    
    def __init__(self):
        self.processingTimes = []
        self.throughputs = []
        
    def recordProcessingTime(self, processingTime: float):
        """Record processing time for a batch"""
        self.processingTimes.append(processingTime)
        
    def recordThroughput(self, numSamples: int, timeElapsed: float):
        """Record throughput"""
        throughput = numSamples / timeElapsed if timeElapsed > 0 else 0
        self.throughputs.append(throughput)
        
    def calculateLatencyMetrics(self) -> Dict[str, float]:
        """Calculate latency metrics"""
        if not self.processingTimes:
            return {}
            
        metrics = {
            'avg_latency_ms': np.mean(self.processingTimes) * 1000,
            'median_latency_ms': np.median(self.processingTimes) * 1000,
            'p95_latency_ms': np.percentile(self.processingTimes, 95) * 1000,
            'p99_latency_ms': np.percentile(self.processingTimes, 99) * 1000,
            'max_latency_ms': np.max(self.processingTimes) * 1000,
            'min_latency_ms': np.min(self.processingTimes) * 1000
        }
        
        return metrics
        
    def calculateThroughputMetrics(self) -> Dict[str, float]:
        """Calculate throughput metrics"""
        if not self.throughputs:
            return {}
            
        metrics = {
            'avg_throughput': np.mean(self.throughputs),
            'median_throughput': np.median(self.throughputs),
            'max_throughput': np.max(self.throughputs),
            'min_throughput': np.min(self.throughputs)
        }
        
        return metrics
