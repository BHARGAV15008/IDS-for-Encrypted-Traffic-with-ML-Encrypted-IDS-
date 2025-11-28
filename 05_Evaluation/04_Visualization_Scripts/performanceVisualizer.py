"""
Performance Visualization Module
Creates comprehensive visualizations for model performance:
- Training/validation curves
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Performance comparison charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceVisualizer:
    """Comprehensive performance visualization for IDS"""
    
    def __init__(self, outputDir: str = 'visualizations', figsize: Tuple[int, int] = (12, 8)):
        """
        Args:
            outputDir: Directory to save visualizations
            figsize: Default figure size
        """
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        
        logger.info(f"Performance Visualizer initialized: output_dir={outputDir}")
        
    def plotTrainingCurves(
        self,
        history: Dict[str, List[float]],
        savePath: Optional[str] = None
    ):
        """
        Plot training and validation curves (loss and accuracy)
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            savePath: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if savePath is None:
            savePath = self.outputDir / 'training_curves.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to: {savePath}")
        plt.close()
        
    def plotConfusionMatrix(
        self,
        yTrue: np.ndarray,
        yPred: np.ndarray,
        classNames: Optional[List[str]] = None,
        normalize: bool = True,
        savePath: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            yTrue: True labels
            yPred: Predicted labels
            classNames: Names of classes
            normalize: Whether to normalize
            savePath: Path to save figure
        """
        cm = confusion_matrix(yTrue, yPred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            
        plt.figure(figsize=self.figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=classNames if classNames else 'auto',
            yticklabels=classNames if classNames else 'auto',
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if savePath is None:
            savePath = self.outputDir / 'confusion_matrix.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {savePath}")
        plt.close()
        
    def plotROCCurve(
        self,
        yTrue: np.ndarray,
        yProba: np.ndarray,
        classNames: Optional[List[str]] = None,
        savePath: Optional[str] = None
    ):
        """
        Plot ROC curve
        
        Args:
            yTrue: True labels
            yProba: Prediction probabilities
            classNames: Names of classes
            savePath: Path to save figure
        """
        plt.figure(figsize=self.figsize)
        
        numClasses = yProba.shape[1] if yProba.ndim > 1 else 2
        
        if numClasses == 2:
            # Binary classification
            if yProba.ndim == 2:
                yProba = yProba[:, 1]
                
            fpr, tpr, _ = roc_curve(yTrue, yProba)
            rocAuc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {rocAuc:.3f})')
            
        else:
            # Multi-class
            from sklearn.preprocessing import label_binarize
            yTrueBin = label_binarize(yTrue, classes=range(numClasses))
            
            for i in range(numClasses):
                fpr, tpr, _ = roc_curve(yTrueBin[:, i], yProba[:, i])
                rocAuc = auc(fpr, tpr)
                className = classNames[i] if classNames else f'Class {i}'
                plt.plot(fpr, tpr, linewidth=2, label=f'{className} (AUC = {rocAuc:.3f})')
                
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if savePath is None:
            savePath = self.outputDir / 'roc_curve.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to: {savePath}")
        plt.close()
        
    def plotPrecisionRecallCurve(
        self,
        yTrue: np.ndarray,
        yProba: np.ndarray,
        classNames: Optional[List[str]] = None,
        savePath: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve
        
        Args:
            yTrue: True labels
            yProba: Prediction probabilities
            classNames: Names of classes
            savePath: Path to save figure
        """
        plt.figure(figsize=self.figsize)
        
        numClasses = yProba.shape[1] if yProba.ndim > 1 else 2
        
        if numClasses == 2:
            # Binary classification
            if yProba.ndim == 2:
                yProba = yProba[:, 1]
                
            precision, recall, _ = precision_recall_curve(yTrue, yProba)
            avgPrecision = average_precision_score(yTrue, yProba)
            
            plt.plot(recall, precision, linewidth=2, label=f'PR curve (AP = {avgPrecision:.3f})')
            
        else:
            # Multi-class
            from sklearn.preprocessing import label_binarize
            yTrueBin = label_binarize(yTrue, classes=range(numClasses))
            
            for i in range(numClasses):
                precision, recall, _ = precision_recall_curve(yTrueBin[:, i], yProba[:, i])
                avgPrecision = average_precision_score(yTrueBin[:, i], yProba[:, i])
                className = classNames[i] if classNames else f'Class {i}'
                plt.plot(recall, precision, linewidth=2, label=f'{className} (AP = {avgPrecision:.3f})')
                
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if savePath is None:
            savePath = self.outputDir / 'precision_recall_curve.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to: {savePath}")
        plt.close()
        
    def plotMetricsComparison(
        self,
        metrics: Dict[str, float],
        savePath: Optional[str] = None
    ):
        """
        Plot bar chart comparing different metrics
        
        Args:
            metrics: Dictionary of metric names and values
            savePath: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        metricNames = list(metrics.keys())
        metricValues = list(metrics.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(metricNames)))
        bars = ax.bar(metricNames, metricValues, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
            
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if savePath is None:
            savePath = self.outputDir / 'metrics_comparison.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison saved to: {savePath}")
        plt.close()
        
    def plotFeatureImportance(
        self,
        featureNames: List[str],
        importances: np.ndarray,
        topN: int = 20,
        savePath: Optional[str] = None
    ):
        """
        Plot feature importance
        
        Args:
            featureNames: Names of features
            importances: Importance scores
            topN: Number of top features to show
            savePath: Path to save figure
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:topN]
        topFeatures = [featureNames[i] for i in indices]
        topImportances = importances[indices]
        
        plt.figure(figsize=(12, max(8, topN * 0.4)))
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(topFeatures)))
        bars = plt.barh(range(len(topFeatures)), topImportances, color=colors, edgecolor='black')
        
        plt.yticks(range(len(topFeatures)), topFeatures)
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title(f'Top {topN} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if savePath is None:
            savePath = self.outputDir / 'feature_importance.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance saved to: {savePath}")
        plt.close()
        
    def plotLearningRate(
        self,
        lrHistory: List[float],
        savePath: Optional[str] = None
    ):
        """
        Plot learning rate schedule
        
        Args:
            lrHistory: List of learning rates per epoch
            savePath: Path to save figure
        """
        plt.figure(figsize=self.figsize)
        
        epochs = range(1, len(lrHistory) + 1)
        plt.plot(epochs, lrHistory, 'b-', linewidth=2)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Learning Rate', fontsize=12, fontweight='bold')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if savePath is None:
            savePath = self.outputDir / 'learning_rate.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"Learning rate plot saved to: {savePath}")
        plt.close()
        
    def plotClassDistribution(
        self,
        yTrain: np.ndarray,
        yTest: np.ndarray,
        classNames: Optional[List[str]] = None,
        savePath: Optional[str] = None
    ):
        """
        Plot class distribution in train and test sets
        
        Args:
            yTrain: Training labels
            yTest: Test labels
            classNames: Names of classes
            savePath: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        uniqueClasses = np.unique(np.concatenate([yTrain, yTest]))
        if classNames is None:
            classNames = [f'Class {i}' for i in uniqueClasses]
            
        # Training set
        trainCounts = [np.sum(yTrain == c) for c in uniqueClasses]
        axes[0].bar(classNames, trainCounts, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add counts on bars
        for i, count in enumerate(trainCounts):
            axes[0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
            
        # Test set
        testCounts = [np.sum(yTest == c) for c in uniqueClasses]
        axes[1].bar(classNames, testCounts, color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[1].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add counts on bars
        for i, count in enumerate(testCounts):
            axes[1].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        
        if savePath is None:
            savePath = self.outputDir / 'class_distribution.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution saved to: {savePath}")
        plt.close()
        
    def plotAdversarialRobustness(
        self,
        epsilons: List[float],
        accuracies: List[float],
        savePath: Optional[str] = None
    ):
        """
        Plot adversarial robustness vs epsilon
        
        Args:
            epsilons: List of epsilon values
            accuracies: Corresponding accuracies
            savePath: Path to save figure
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(epsilons, accuracies, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Epsilon (Perturbation Magnitude)', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Adversarial Robustness Evaluation', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        # Add value labels
        for eps, acc in zip(epsilons, accuracies):
            plt.text(eps, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)
            
        plt.tight_layout()
        
        if savePath is None:
            savePath = self.outputDir / 'adversarial_robustness.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"Adversarial robustness plot saved to: {savePath}")
        plt.close()
        
    def createComprehensiveDashboard(
        self,
        history: Dict[str, List[float]],
        yTrue: np.ndarray,
        yPred: np.ndarray,
        yProba: Optional[np.ndarray] = None,
        classNames: Optional[List[str]] = None,
        savePath: Optional[str] = None
    ):
        """
        Create comprehensive dashboard with all visualizations
        
        Args:
            history: Training history
            yTrue: True labels
            yPred: Predicted labels
            yProba: Prediction probabilities
            classNames: Names of classes
            savePath: Path to save figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Training Loss
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.set_title('Training Accuracy', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        ax3 = fig.add_subplot(gs[0, 2])
        cm = confusion_matrix(yTrue, yPred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax3,
                    xticklabels=classNames if classNames else 'auto',
                    yticklabels=classNames if classNames else 'auto')
        ax3.set_xlabel('Predicted', fontweight='bold')
        ax3.set_ylabel('True', fontweight='bold')
        ax3.set_title('Confusion Matrix', fontweight='bold')
        
        # 4. ROC Curve
        if yProba is not None:
            ax4 = fig.add_subplot(gs[1, 0])
            if yProba.ndim == 2 and yProba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(yTrue, yProba[:, 1])
                rocAuc = auc(fpr, tpr)
                ax4.plot(fpr, tpr, linewidth=2, label=f'AUC = {rocAuc:.3f}')
            ax4.plot([0, 1], [0, 1], 'k--', linewidth=2)
            ax4.set_xlabel('False Positive Rate', fontweight='bold')
            ax4.set_ylabel('True Positive Rate', fontweight='bold')
            ax4.set_title('ROC Curve', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
        # 5. Metrics Bar Chart
        ax5 = fig.add_subplot(gs[1, 1])
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            'Accuracy': accuracy_score(yTrue, yPred),
            'Precision': precision_score(yTrue, yPred, average='weighted', zero_division=0),
            'Recall': recall_score(yTrue, yPred, average='weighted', zero_division=0),
            'F1-Score': f1_score(yTrue, yPred, average='weighted', zero_division=0)
        }
        bars = ax5.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax5.set_ylabel('Score', fontweight='bold')
        ax5.set_title('Performance Metrics', fontweight='bold')
        ax5.set_ylim([0, 1.1])
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Class Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        unique, counts = np.unique(yTrue, return_counts=True)
        ax6.bar(unique, counts, color='skyblue', edgecolor='black')
        ax6.set_xlabel('Class', fontweight='bold')
        ax6.set_ylabel('Count', fontweight='bold')
        ax6.set_title('Class Distribution', fontweight='bold')
        
        # 7-9. Additional metrics or text summary
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Summary text
        summaryText = f"""
        Model Performance Summary
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Total Samples: {len(yTrue)}  |  Classes: {len(np.unique(yTrue))}  |  Epochs: {len(history['train_loss'])}
        
        Final Training Accuracy: {history['train_acc'][-1]:.4f}  |  Final Validation Accuracy: {history['val_acc'][-1]:.4f}
        Best Validation Accuracy: {max(history['val_acc']):.4f}  |  Test Accuracy: {accuracy_score(yTrue, yPred):.4f}
        
        Detection Rate: {metrics.get('Recall', 0):.4f}  |  Precision: {metrics.get('Precision', 0):.4f}  |  F1-Score: {metrics.get('F1-Score', 0):.4f}
        """
        
        ax7.text(0.5, 0.5, summaryText, ha='center', va='center', fontsize=11,
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Comprehensive Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        if savePath is None:
            savePath = self.outputDir / 'comprehensive_dashboard.png'
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive dashboard saved to: {savePath}")
        plt.close()


class RealTimeVisualizer:
    """Real-time visualization during training"""
    
    def __init__(self):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 5))
        plt.ion()
        
    def update(self, history: Dict[str, List[float]]):
        """Update plots in real-time"""
        for ax in self.axes:
            ax.clear()
            
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        self.axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        self.axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].set_title('Loss')
        self.axes[0].legend()
        self.axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        self.axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        self.axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('Accuracy')
        self.axes[1].set_title('Accuracy')
        self.axes[1].legend()
        self.axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
        
    def close(self):
        """Close the plot"""
        plt.ioff()
        plt.close()
