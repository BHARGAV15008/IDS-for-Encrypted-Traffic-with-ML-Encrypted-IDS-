"""
P22 KPI Calculator - Comprehensive Performance Metrics

This module implements the specific KPIs required for P22 project evaluation,
including detection rate, false positive rate, adversarial robustness, and
cross-dataset generalization metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, matthews_corrcoef
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging
import time
from pathlib import Path


@dataclass
class P22KPIResults:
    """Data class for P22 KPI results."""
    
    # Core Detection Metrics
    detection_rate: float  # Target: ≥95%
    false_positive_rate: float  # Target: ≤2%
    f1_score: float  # Target: ≥90%
    
    # Adversarial Robustness
    adversarial_detection_rate: float  # Target: ≥90%
    robustness_gap: float  # Clean accuracy - Adversarial accuracy
    
    # Cross-Dataset Generalization
    cross_dataset_f1: float  # Target: ≥85%
    generalization_score: float
    
    # Real-Time Performance
    average_latency_ms: float  # Target: ≤100ms
    throughput_fps: float  # Flows per second
    
    # Overall Assessment
    overall_score: float
    kpi_compliance: Dict[str, bool]
    
    # Additional Metrics
    precision: float
    recall: float
    auc_roc: float
    matthews_correlation: float


class P22KPICalculator:
    """
    Comprehensive KPI calculator for P22 project evaluation.
    
    This calculator implements all required metrics for the P22 encrypted
    traffic IDS project, focusing on the specific KPIs outlined in the
    project requirements.
    """
    
    def __init__(self, target_kpis: Optional[Dict[str, float]] = None):
        """
        Initialize P22 KPI calculator.
        
        Args:
            target_kpis: Dictionary of target KPI values
        """
        self.logger = logging.getLogger(__name__)
        
        # Default P22 target KPIs
        self.target_kpis = target_kpis or {
            'detection_rate': 0.95,  # ≥95%
            'false_positive_rate': 0.02,  # ≤2%
            'f1_score': 0.90,  # ≥90%
            'adversarial_detection_rate': 0.90,  # ≥90%
            'cross_dataset_f1': 0.85,  # ≥85%
            'average_latency_ms': 100.0,  # ≤100ms
            'throughput_fps': 1000.0  # ≥1000 flows/second
        }
        
        # KPI weights for overall score calculation
        self.kpi_weights = {
            'detection_rate': 0.25,
            'false_positive_rate': 0.20,
            'adversarial_detection_rate': 0.20,
            'cross_dataset_f1': 0.15,
            'f1_score': 0.10,
            'latency_performance': 0.10
        }
    
    def calculate_comprehensive_kpis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        adversarial_results: Optional[Dict] = None,
        cross_dataset_results: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None
    ) -> P22KPIResults:
        """
        Calculate comprehensive P22 KPIs.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            adversarial_results: Results from adversarial evaluation
            cross_dataset_results: Results from cross-dataset evaluation
            performance_metrics: Real-time performance metrics
            
        Returns:
            P22KPIResults object with all calculated metrics
        """
        # Core detection metrics
        core_metrics = self._calculate_core_metrics(y_true, y_pred, y_prob)
        
        # Adversarial robustness metrics
        adv_metrics = self._calculate_adversarial_metrics(adversarial_results)
        
        # Cross-dataset generalization metrics
        cross_metrics = self._calculate_cross_dataset_metrics(cross_dataset_results)
        
        # Performance metrics
        perf_metrics = self._calculate_performance_metrics(performance_metrics)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            core_metrics, adv_metrics, cross_metrics, perf_metrics
        )
        
        # Check KPI compliance
        kpi_compliance = self._check_kpi_compliance(
            core_metrics, adv_metrics, cross_metrics, perf_metrics
        )
        
        return P22KPIResults(
            # Core metrics
            detection_rate=core_metrics['detection_rate'],
            false_positive_rate=core_metrics['false_positive_rate'],
            f1_score=core_metrics['f1_score'],
            precision=core_metrics['precision'],
            recall=core_metrics['recall'],
            auc_roc=core_metrics['auc_roc'],
            matthews_correlation=core_metrics['matthews_correlation'],
            
            # Adversarial metrics
            adversarial_detection_rate=adv_metrics['adversarial_detection_rate'],
            robustness_gap=adv_metrics['robustness_gap'],
            
            # Cross-dataset metrics
            cross_dataset_f1=cross_metrics['cross_dataset_f1'],
            generalization_score=cross_metrics['generalization_score'],
            
            # Performance metrics
            average_latency_ms=perf_metrics['average_latency_ms'],
            throughput_fps=perf_metrics['throughput_fps'],
            
            # Overall assessment
            overall_score=overall_score,
            kpi_compliance=kpi_compliance
        )
    
    def _calculate_core_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate core detection metrics."""
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Detection rate (recall for attack classes)
        # Assuming class 0 is normal, others are attacks
        attack_mask = y_true != 0
        if np.sum(attack_mask) > 0:
            detection_rate = recall_score(
                y_true[attack_mask] != 0,
                y_pred[attack_mask] != 0,
                zero_division=0
            )
        else:
            detection_rate = 0.0
        
        # False positive rate
        normal_mask = y_true == 0
        if np.sum(normal_mask) > 0:
            false_positives = np.sum((y_true[normal_mask] == 0) & (y_pred[normal_mask] != 0))
            total_normal = np.sum(normal_mask)
            false_positive_rate = false_positives / total_normal
        else:
            false_positive_rate = 0.0
        
        # AUC-ROC (if probabilities available)
        auc_roc = 0.0
        if y_prob is not None:
            try:
                # Convert to binary classification for AUC calculation
                y_binary = (y_true != 0).astype(int)
                if y_prob.ndim > 1:
                    # Multi-class probabilities - use probability of attack classes
                    y_prob_binary = 1 - y_prob[:, 0]  # 1 - P(normal)
                else:
                    y_prob_binary = y_prob
                
                auc_roc = roc_auc_score(y_binary, y_prob_binary)
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC-ROC: {e}")
        
        # Matthews Correlation Coefficient
        try:
            matthews_correlation = matthews_corrcoef(y_true, y_pred)
        except Exception:
            matthews_correlation = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'auc_roc': auc_roc,
            'matthews_correlation': matthews_correlation
        }
    
    def _calculate_adversarial_metrics(
        self,
        adversarial_results: Optional[Dict]
    ) -> Dict[str, float]:
        """Calculate adversarial robustness metrics."""
        
        if adversarial_results is None:
            return {
                'adversarial_detection_rate': 0.0,
                'robustness_gap': 1.0,
                'attack_success_rate': 1.0
            }
        
        # Extract adversarial accuracy
        adv_accuracy = adversarial_results.get('adversarial_accuracy', 0.0)
        clean_accuracy = adversarial_results.get('clean_accuracy', 0.0)
        
        # Calculate robustness gap
        robustness_gap = clean_accuracy - adv_accuracy
        
        # Attack success rate (1 - adversarial accuracy)
        attack_success_rate = 1.0 - adv_accuracy
        
        return {
            'adversarial_detection_rate': adv_accuracy,
            'robustness_gap': robustness_gap,
            'attack_success_rate': attack_success_rate
        }
    
    def _calculate_cross_dataset_metrics(
        self,
        cross_dataset_results: Optional[Dict]
    ) -> Dict[str, float]:
        """Calculate cross-dataset generalization metrics."""
        
        if cross_dataset_results is None:
            return {
                'cross_dataset_f1': 0.0,
                'generalization_score': 0.0,
                'dataset_variance': 1.0
            }
        
        # Extract F1 scores across different datasets
        f1_scores = cross_dataset_results.get('f1_scores', [])
        
        if len(f1_scores) == 0:
            return {
                'cross_dataset_f1': 0.0,
                'generalization_score': 0.0,
                'dataset_variance': 1.0
            }
        
        # Average F1 score across datasets
        cross_dataset_f1 = np.mean(f1_scores)
        
        # Generalization score (1 - coefficient of variation)
        if len(f1_scores) > 1:
            cv = np.std(f1_scores) / (np.mean(f1_scores) + 1e-10)
            generalization_score = 1.0 / (1.0 + cv)
        else:
            generalization_score = cross_dataset_f1
        
        # Dataset variance
        dataset_variance = np.var(f1_scores) if len(f1_scores) > 1 else 0.0
        
        return {
            'cross_dataset_f1': cross_dataset_f1,
            'generalization_score': generalization_score,
            'dataset_variance': dataset_variance
        }
    
    def _calculate_performance_metrics(
        self,
        performance_metrics: Optional[Dict]
    ) -> Dict[str, float]:
        """Calculate real-time performance metrics."""
        
        if performance_metrics is None:
            return {
                'average_latency_ms': 1000.0,  # Default high latency
                'throughput_fps': 1.0,  # Default low throughput
                'memory_usage_mb': 1000.0,
                'cpu_utilization': 100.0
            }
        
        return {
            'average_latency_ms': performance_metrics.get('average_latency_ms', 1000.0),
            'throughput_fps': performance_metrics.get('throughput_fps', 1.0),
            'memory_usage_mb': performance_metrics.get('memory_usage_mb', 1000.0),
            'cpu_utilization': performance_metrics.get('cpu_utilization', 100.0)
        }
    
    def _calculate_overall_score(
        self,
        core_metrics: Dict,
        adv_metrics: Dict,
        cross_metrics: Dict,
        perf_metrics: Dict
    ) -> float:
        """Calculate weighted overall score."""
        
        # Normalize metrics to 0-1 scale where 1 is best
        normalized_scores = {}
        
        # Detection rate (higher is better)
        normalized_scores['detection_rate'] = min(
            core_metrics['detection_rate'] / self.target_kpis['detection_rate'], 1.0
        )
        
        # False positive rate (lower is better)
        if core_metrics['false_positive_rate'] <= self.target_kpis['false_positive_rate']:
            normalized_scores['false_positive_rate'] = 1.0
        else:
            normalized_scores['false_positive_rate'] = max(
                0.0, 1.0 - (core_metrics['false_positive_rate'] - self.target_kpis['false_positive_rate']) / 0.1
            )
        
        # Adversarial detection rate (higher is better)
        normalized_scores['adversarial_detection_rate'] = min(
            adv_metrics['adversarial_detection_rate'] / self.target_kpis['adversarial_detection_rate'], 1.0
        )
        
        # Cross-dataset F1 (higher is better)
        normalized_scores['cross_dataset_f1'] = min(
            cross_metrics['cross_dataset_f1'] / self.target_kpis['cross_dataset_f1'], 1.0
        )
        
        # F1 score (higher is better)
        normalized_scores['f1_score'] = min(
            core_metrics['f1_score'] / self.target_kpis['f1_score'], 1.0
        )
        
        # Latency performance (lower is better)
        if perf_metrics['average_latency_ms'] <= self.target_kpis['average_latency_ms']:
            normalized_scores['latency_performance'] = 1.0
        else:
            normalized_scores['latency_performance'] = max(
                0.0, 1.0 - (perf_metrics['average_latency_ms'] - self.target_kpis['average_latency_ms']) / 100.0
            )
        
        # Calculate weighted overall score
        overall_score = sum(
            normalized_scores[metric] * weight
            for metric, weight in self.kpi_weights.items()
            if metric in normalized_scores
        )
        
        return overall_score
    
    def _check_kpi_compliance(
        self,
        core_metrics: Dict,
        adv_metrics: Dict,
        cross_metrics: Dict,
        perf_metrics: Dict
    ) -> Dict[str, bool]:
        """Check compliance with P22 KPI targets."""
        
        compliance = {}
        
        # Core metrics compliance
        compliance['detection_rate'] = core_metrics['detection_rate'] >= self.target_kpis['detection_rate']
        compliance['false_positive_rate'] = core_metrics['false_positive_rate'] <= self.target_kpis['false_positive_rate']
        compliance['f1_score'] = core_metrics['f1_score'] >= self.target_kpis['f1_score']
        
        # Adversarial compliance
        compliance['adversarial_detection_rate'] = adv_metrics['adversarial_detection_rate'] >= self.target_kpis['adversarial_detection_rate']
        
        # Cross-dataset compliance
        compliance['cross_dataset_f1'] = cross_metrics['cross_dataset_f1'] >= self.target_kpis['cross_dataset_f1']
        
        # Performance compliance
        compliance['average_latency_ms'] = perf_metrics['average_latency_ms'] <= self.target_kpis['average_latency_ms']
        
        # Overall compliance (all KPIs must pass)
        compliance['overall'] = all(compliance.values())
        
        return compliance
    
    def generate_kpi_report(self, results: P22KPIResults) -> str:
        """Generate comprehensive KPI report."""
        
        report = "=" * 80 + "\n"
        report += "P22 ENCRYPTED TRAFFIC IDS - KPI EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Executive Summary
        report += "EXECUTIVE SUMMARY\n"
        report += "-" * 40 + "\n"
        report += f"Overall Score: {results.overall_score:.3f} / 1.000\n"
        report += f"KPI Compliance: {'✓ PASS' if results.kpi_compliance['overall'] else '✗ FAIL'}\n\n"
        
        # Core Detection Metrics
        report += "CORE DETECTION METRICS\n"
        report += "-" * 40 + "\n"
        report += f"Detection Rate: {results.detection_rate:.3f} "
        report += f"({'✓' if results.kpi_compliance['detection_rate'] else '✗'} Target: ≥{self.target_kpis['detection_rate']:.3f})\n"
        
        report += f"False Positive Rate: {results.false_positive_rate:.3f} "
        report += f"({'✓' if results.kpi_compliance['false_positive_rate'] else '✗'} Target: ≤{self.target_kpis['false_positive_rate']:.3f})\n"
        
        report += f"F1-Score: {results.f1_score:.3f} "
        report += f"({'✓' if results.kpi_compliance['f1_score'] else '✗'} Target: ≥{self.target_kpis['f1_score']:.3f})\n"
        
        report += f"Precision: {results.precision:.3f}\n"
        report += f"Recall: {results.recall:.3f}\n"
        report += f"AUC-ROC: {results.auc_roc:.3f}\n\n"
        
        # Adversarial Robustness
        report += "ADVERSARIAL ROBUSTNESS\n"
        report += "-" * 40 + "\n"
        report += f"Adversarial Detection Rate: {results.adversarial_detection_rate:.3f} "
        report += f"({'✓' if results.kpi_compliance['adversarial_detection_rate'] else '✗'} Target: ≥{self.target_kpis['adversarial_detection_rate']:.3f})\n"
        
        report += f"Robustness Gap: {results.robustness_gap:.3f}\n\n"
        
        # Cross-Dataset Generalization
        report += "CROSS-DATASET GENERALIZATION\n"
        report += "-" * 40 + "\n"
        report += f"Cross-Dataset F1: {results.cross_dataset_f1:.3f} "
        report += f"({'✓' if results.kpi_compliance['cross_dataset_f1'] else '✗'} Target: ≥{self.target_kpis['cross_dataset_f1']:.3f})\n"
        
        report += f"Generalization Score: {results.generalization_score:.3f}\n\n"
        
        # Real-Time Performance
        report += "REAL-TIME PERFORMANCE\n"
        report += "-" * 40 + "\n"
        report += f"Average Latency: {results.average_latency_ms:.1f} ms "
        report += f"({'✓' if results.kpi_compliance['average_latency_ms'] else '✗'} Target: ≤{self.target_kpis['average_latency_ms']:.1f} ms)\n"
        
        report += f"Throughput: {results.throughput_fps:.1f} flows/second\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS\n"
        report += "-" * 40 + "\n"
        
        if not results.kpi_compliance['detection_rate']:
            report += "• Improve detection rate through better feature engineering or model architecture\n"
        
        if not results.kpi_compliance['false_positive_rate']:
            report += "• Reduce false positives through threshold tuning or ensemble methods\n"
        
        if not results.kpi_compliance['adversarial_detection_rate']:
            report += "• Enhance adversarial robustness through adversarial training\n"
        
        if not results.kpi_compliance['cross_dataset_f1']:
            report += "• Improve generalization through domain adaptation techniques\n"
        
        if not results.kpi_compliance['average_latency_ms']:
            report += "• Optimize inference speed through model quantization or pruning\n"
        
        if results.kpi_compliance['overall']:
            report += "• All KPIs met - ready for production deployment\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    def plot_kpi_dashboard(self, results: P22KPIResults, save_path: Optional[Path] = None):
        """Create KPI dashboard visualization."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('P22 IDS - KPI Dashboard', fontsize=16, fontweight='bold')
        
        # KPI Compliance Overview
        ax = axes[0, 0]
        kpi_names = ['Detection\nRate', 'False Positive\nRate', 'Adversarial\nRobustness', 'Cross-Dataset\nF1', 'Latency']
        kpi_status = [
            results.kpi_compliance['detection_rate'],
            results.kpi_compliance['false_positive_rate'],
            results.kpi_compliance['adversarial_detection_rate'],
            results.kpi_compliance['cross_dataset_f1'],
            results.kpi_compliance['average_latency_ms']
        ]
        
        colors = ['green' if status else 'red' for status in kpi_status]
        bars = ax.bar(kpi_names, [1 if status else 0 for status in kpi_status], color=colors, alpha=0.7)
        ax.set_title('KPI Compliance Status')
        ax.set_ylabel('Compliance (1=Pass, 0=Fail)')
        ax.set_ylim(0, 1.2)
        
        # Add compliance text
        for bar, status in zip(bars, kpi_status):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   '✓' if status else '✗', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Overall Score Gauge
        ax = axes[0, 1]
        score = results.overall_score
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Create semicircle gauge
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, 'k-', linewidth=2)
        
        # Score indicator
        score_angle = np.pi * (1 - score)
        score_x = r * np.cos(score_angle)
        score_y = r * np.sin(score_angle)
        ax.arrow(0, 0, score_x, score_y, head_width=0.05, head_length=0.05, fc='red', ec='red')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(f'Overall Score: {score:.3f}')
        ax.axis('off')
        
        # Performance Metrics
        ax = axes[0, 2]
        metrics = ['Detection\nRate', 'Precision', 'Recall', 'F1-Score']
        values = [results.detection_rate, results.precision, results.recall, results.f1_score]
        
        bars = ax.bar(metrics, values, color='skyblue', alpha=0.7)
        ax.set_title('Core Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Adversarial Robustness
        ax = axes[1, 0]
        categories = ['Clean\nAccuracy', 'Adversarial\nAccuracy']
        clean_acc = results.adversarial_detection_rate + results.robustness_gap
        adv_acc = results.adversarial_detection_rate
        
        bars = ax.bar(categories, [clean_acc, adv_acc], color=['green', 'orange'], alpha=0.7)
        ax.set_title('Adversarial Robustness')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        # Add robustness gap annotation
        ax.annotate('', xy=(0, clean_acc), xytext=(1, adv_acc),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(0.5, (clean_acc + adv_acc) / 2, f'Gap: {results.robustness_gap:.3f}',
               ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Cross-Dataset Performance
        ax = axes[1, 1]
        # Placeholder data - in real implementation, this would show per-dataset results
        datasets = ['NSL-KDD', 'UNSW-NB15', 'CICIDS', 'Custom']
        f1_scores = [0.92, 0.88, 0.85, 0.83]  # Example data
        
        bars = ax.bar(datasets, f1_scores, color='lightcoral', alpha=0.7)
        ax.axhline(y=self.target_kpis['cross_dataset_f1'], color='red', linestyle='--', 
                  label=f'Target: {self.target_kpis["cross_dataset_f1"]:.2f}')
        ax.set_title('Cross-Dataset F1 Scores')
        ax.set_ylabel('F1-Score')
        ax.set_ylim(0, 1)
        ax.legend()
        
        # Real-Time Performance
        ax = axes[1, 2]
        perf_metrics = ['Latency\n(ms)', 'Throughput\n(fps)']
        perf_values = [results.average_latency_ms, results.throughput_fps / 10]  # Scale throughput for visualization
        targets = [self.target_kpis['average_latency_ms'], self.target_kpis['throughput_fps'] / 10]
        
        x = np.arange(len(perf_metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, perf_values, width, label='Actual', color='lightblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, targets, width, label='Target', color='lightgreen', alpha=0.7)
        
        ax.set_title('Real-Time Performance')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(perf_metrics)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"KPI dashboard saved to {save_path}")
        
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic test data
    y_true = np.random.randint(0, 5, n_samples)  # 5 classes (0=normal, 1-4=attacks)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y_pred[error_indices] = np.random.randint(0, 5, len(error_indices))
    
    # Generate probabilities
    y_prob = np.random.dirichlet(np.ones(5), n_samples)
    
    # Example adversarial results
    adversarial_results = {
        'clean_accuracy': 0.92,
        'adversarial_accuracy': 0.87
    }
    
    # Example cross-dataset results
    cross_dataset_results = {
        'f1_scores': [0.91, 0.88, 0.86, 0.84]
    }
    
    # Example performance metrics
    performance_metrics = {
        'average_latency_ms': 85.0,
        'throughput_fps': 1200.0,
        'memory_usage_mb': 512.0,
        'cpu_utilization': 45.0
    }
    
    # Initialize calculator
    calculator = P22KPICalculator()
    
    # Calculate KPIs
    results = calculator.calculate_comprehensive_kpis(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        adversarial_results=adversarial_results,
        cross_dataset_results=cross_dataset_results,
        performance_metrics=performance_metrics
    )
    
    # Generate report
    report = calculator.generate_kpi_report(results)
    print(report)
    
    # Plot dashboard
    calculator.plot_kpi_dashboard(results)
