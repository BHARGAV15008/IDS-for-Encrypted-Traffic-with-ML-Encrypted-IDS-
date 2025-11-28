"""
Test Script: Execute Improved Workflow on Multiple Datasets
Tests all 4 phases on MachineLearningCVE and TrafficLabelling datasets
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / '04_Source_Code'))

from improvedCompleteWorkflow import ImprovedCompleteWorkflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dataset(csv_path: str, dataset_name: str, output_dir: str) -> dict:
    """Test improved workflow on a single dataset"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {dataset_name}")
    logger.info(f"File: {csv_path}")
    logger.info(f"{'='*80}")
    
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(csv_path, nrows=10000)  # Limit to 10k rows for faster testing
        
        logger.info(f"  - Rows: {len(df)}")
        logger.info(f"  - Columns: {len(df.columns)}")
        logger.info(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Separate features and labels
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        logger.info(f"  - Features: {X.shape[1]}")
        logger.info(f"  - Label distribution: {y.value_counts().to_dict()}")
        
        # Run workflow
        logger.info("\nRunning improved workflow...")
        workflow = ImprovedCompleteWorkflow(output_dir=output_dir, device='cpu')
        results = workflow.run(X, y, num_epochs=10, batch_size=128, learning_rate=0.001)
        
        # Extract key metrics
        metrics = results.get('phase4', {}).get('metrics', {})
        
        test_result = {
            'dataset': dataset_name,
            'file': csv_path,
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'data_stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'features': X.shape[1]
            },
            'phase1': results.get('phase1', {}),
            'phase2': results.get('phase2', {}),
            'phase3': results.get('phase3', {}),
            'phase4_metrics': metrics,
            'optimal_threshold': results.get('phase4', {}).get('optimal_threshold', None)
        }
        
        logger.info(f"\n✓ Test completed successfully!")
        logger.info(f"  - Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  - Detection Rate: {metrics.get('detection_rate', 0):.4f}")
        logger.info(f"  - FPR: {metrics.get('fpr', 0):.4f}")
        logger.info(f"  - F1-Score: {metrics.get('f1', 0):.4f}")
        
        return test_result
        
    except Exception as e:
        logger.error(f"✗ Test failed: {str(e)}")
        return {
            'dataset': dataset_name,
            'file': csv_path,
            'status': 'FAILED',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main test function"""
    
    logger.info("\n" + "="*80)
    logger.info("IMPROVED WORKFLOW - COMPREHENSIVE TESTING")
    logger.info("="*80)
    
    # Test datasets
    test_datasets = [
        # MachineLearningCVE
        ('01_Data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv', 'MachineLearningCVE - Monday'),
        ('01_Data/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv', 'MachineLearningCVE - Friday Morning'),
        ('01_Data/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'MachineLearningCVE - Friday DDos'),
        
        # TrafficLabelling
        ('01_Data/TrafficLabelling_/Monday-WorkingHours.pcap_ISCX.csv', 'TrafficLabelling - Monday'),
        ('01_Data/TrafficLabelling_/Friday-WorkingHours-Morning.pcap_ISCX.csv', 'TrafficLabelling - Friday Morning'),
        ('01_Data/TrafficLabelling_/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'TrafficLabelling - Friday DDos'),
    ]
    
    # Run tests
    results = []
    for csv_path, dataset_name in test_datasets:
        output_dir = f'outputs/test_{dataset_name.replace(" ", "_").replace("-", "_")}'
        result = test_dataset(csv_path, dataset_name, output_dir)
        results.append(result)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    
    logger.info(f"\nTotal Tests: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    # Performance summary
    logger.info("\n" + "-"*80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("-"*80)
    
    for result in results:
        if result['status'] == 'SUCCESS':
            metrics = result.get('phase4_metrics', {})
            logger.info(f"\n{result['dataset']}:")
            logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"  Detection Rate: {metrics.get('detection_rate', 0):.4f}")
            logger.info(f"  FPR: {metrics.get('fpr', 0):.4f}")
            logger.info(f"  F1-Score: {metrics.get('f1', 0):.4f}")
            logger.info(f"  Optimal Threshold: {result.get('optimal_threshold', 'N/A')}")
    
    # Save results
    results_file = 'outputs/test_results_summary.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info(f"\n✓ Results saved to: {results_file}")
    
    logger.info("\n" + "="*80)
    logger.info("TESTING COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
