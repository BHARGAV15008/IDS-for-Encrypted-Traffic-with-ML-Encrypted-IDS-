"""
Visualization Example Script
Demonstrates how to create all types of performance visualizations
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
projectRoot = Path(__file__).parent.parent.parent
sys.path.insert(0, str(projectRoot))

from performanceVisualizer import PerformanceVisualizer


def generateSampleData():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Training history
    epochs = 50
    history = {
        'train_loss': [0.8 * np.exp(-0.05 * i) + 0.1 + np.random.normal(0, 0.02) for i in range(epochs)],
        'val_loss': [0.85 * np.exp(-0.045 * i) + 0.15 + np.random.normal(0, 0.03) for i in range(epochs)],
        'train_acc': [1 - 0.7 * np.exp(-0.05 * i) + np.random.normal(0, 0.01) for i in range(epochs)],
        'val_acc': [1 - 0.75 * np.exp(-0.045 * i) + np.random.normal(0, 0.015) for i in range(epochs)]
    }
    
    # Test predictions (binary classification)
    numSamples = 1000
    yTrue = np.random.randint(0, 2, numSamples)
    
    # Generate predictions with some errors
    yProba = np.zeros((numSamples, 2))
    for i in range(numSamples):
        if yTrue[i] == 1:
            yProba[i, 1] = np.random.beta(8, 2)  # High probability for class 1
        else:
            yProba[i, 1] = np.random.beta(2, 8)  # Low probability for class 1
        yProba[i, 0] = 1 - yProba[i, 1]
    
    yPred = yProba.argmax(axis=1)
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        'Accuracy': accuracy_score(yTrue, yPred),
        'Precision': precision_score(yTrue, yPred, zero_division=0),
        'Recall': recall_score(yTrue, yPred, zero_division=0),
        'F1-Score': f1_score(yTrue, yPred, zero_division=0),
        'Detection Rate': recall_score(yTrue, yPred, zero_division=0),
        'False Alarm Rate': 1 - precision_score(yTrue, yPred, zero_division=0)
    }
    
    # Feature importance
    featureNames = [f'Feature_{i}' for i in range(20)]
    importances = np.random.exponential(0.1, 20)
    importances = importances / importances.sum()
    
    # Adversarial robustness
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]
    accuracies = [0.95, 0.93, 0.89, 0.78, 0.65]
    
    return history, yTrue, yPred, yProba, metrics, featureNames, importances, epsilons, accuracies


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("IDS PERFORMANCE VISUALIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Generate sample data
    print("\n[1/9] Generating sample data...")
    history, yTrue, yPred, yProba, metrics, featureNames, importances, epsilons, accuracies = generateSampleData()
    
    # Initialize visualizer
    print("[2/9] Initializing visualizer...")
    visualizer = PerformanceVisualizer(outputDir='example_visualizations')
    
    # 1. Training curves
    print("[3/9] Creating training curves...")
    visualizer.plotTrainingCurves(history)
    print("  ✓ Saved: example_visualizations/training_curves.png")
    
    # 2. Confusion matrix
    print("[4/9] Creating confusion matrix...")
    visualizer.plotConfusionMatrix(
        yTrue, yPred,
        classNames=['Benign', 'Attack'],
        normalize=True
    )
    print("  ✓ Saved: example_visualizations/confusion_matrix.png")
    
    # 3. ROC curve
    print("[5/9] Creating ROC curve...")
    visualizer.plotROCCurve(
        yTrue, yProba,
        classNames=['Benign', 'Attack']
    )
    print("  ✓ Saved: example_visualizations/roc_curve.png")
    
    # 4. Precision-Recall curve
    print("[6/9] Creating Precision-Recall curve...")
    visualizer.plotPrecisionRecallCurve(
        yTrue, yProba,
        classNames=['Benign', 'Attack']
    )
    print("  ✓ Saved: example_visualizations/precision_recall_curve.png")
    
    # 5. Metrics comparison
    print("[7/9] Creating metrics comparison...")
    visualizer.plotMetricsComparison(metrics)
    print("  ✓ Saved: example_visualizations/metrics_comparison.png")
    
    # 6. Feature importance
    print("[8/9] Creating feature importance plot...")
    visualizer.plotFeatureImportance(featureNames, importances, topN=15)
    print("  ✓ Saved: example_visualizations/feature_importance.png")
    
    # 7. Adversarial robustness
    print("[9/9] Creating adversarial robustness plot...")
    visualizer.plotAdversarialRobustness(epsilons, accuracies)
    print("  ✓ Saved: example_visualizations/adversarial_robustness.png")
    
    # 8. Comprehensive dashboard
    print("[BONUS] Creating comprehensive dashboard...")
    visualizer.createComprehensiveDashboard(
        history=history,
        yTrue=yTrue,
        yPred=yPred,
        yProba=yProba,
        classNames=['Benign', 'Attack']
    )
    print("  ✓ Saved: example_visualizations/comprehensive_dashboard.png")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION DEMONSTRATION COMPLETED!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: example_visualizations/")
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    print("  - metrics_comparison.png")
    print("  - feature_importance.png")
    print("  - adversarial_robustness.png")
    print("  - comprehensive_dashboard.png")
    print("\n" + "=" * 80)
    
    # Display sample metrics
    print("\nSample Performance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    print("-" * 40)


if __name__ == '__main__':
    main()
