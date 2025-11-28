#!/usr/bin/env python3
"""
Example Usage Scripts for P22 Encrypted Traffic IDS

Demonstrates various ways to use the microservices architecture.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import ServiceOrchestrator
from services.lstmModelService import LSTMModelService
from services.cnnModelService import CNNModelService
from services.dataIngestionService import DataIngestionService
from services.outputManagementService import OutputManagementService
from services.configurationService import ConfigurationService


def example1_basic_usage():
    """Example 1: Basic end-to-end pipeline."""
    print("=" * 60)
    print("Example 1: Basic End-to-End Pipeline")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = ServiceOrchestrator()
    
    if orchestrator.initialize():
        print("✓ Services initialized\n")
        
        # Run prediction on a file
        # Note: Replace with actual file path
        result = orchestrator.runEndToEndPipeline(
            filePath='sample_data.csv',
            modelType='both',
            aggregate=True
        )
        
        print(f"Prediction Result: {result.get('predictions')}")
        print(f"Confidence: {result.get('confidence')}")
        
        # Shutdown
        orchestrator.shutdown()
        print("\n✓ Services shut down")
    else:
        print("✗ Failed to initialize services")


def example2_individual_services():
    """Example 2: Using individual services."""
    print("\n" + "=" * 60)
    print("Example 2: Using Individual Services")
    print("=" * 60)
    
    # Initialize configuration service
    configService = ConfigurationService()
    configService.start()
    config = configService.getConfiguration()
    
    # Initialize LSTM service
    lstmService = LSTMModelService(config.get('lstmModel'))
    lstmService.start()
    print(f"✓ LSTM Service started")
    print(f"  Model Info: {lstmService.getModelInfo()}")
    
    # Initialize CNN service
    cnnService = CNNModelService(config.get('cnnModel'))
    cnnService.start()
    print(f"✓ CNN Service started")
    print(f"  Model Info: {cnnService.getModelInfo()}")
    
    # Create dummy data for demonstration
    import numpy as np
    dummyFeatures = np.random.randn(1, 20, 256)  # (batch, seq, features)
    
    # Run LSTM prediction
    lstmResult = lstmService.process({
        'features': dummyFeatures,
        'metadata': {'source': 'example'}
    })
    print(f"\nLSTM Prediction: {lstmResult['predictions']}")
    
    # Run CNN prediction
    cnnFeatures = np.random.randn(1, 1, 1500)  # (batch, channels, seq)
    cnnResult = cnnService.process({
        'features': cnnFeatures,
        'metadata': {'source': 'example'}
    })
    print(f"CNN Prediction: {cnnResult['predictions']}")
    
    # Cleanup
    lstmService.stop()
    cnnService.stop()
    configService.stop()
    print("\n✓ Services stopped")


def example3_batch_processing():
    """Example 3: Batch processing multiple files."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    orchestrator = ServiceOrchestrator()
    
    if orchestrator.initialize():
        # List of files to process
        # Note: Replace with actual file paths
        fileList = [
            'data1.csv',
            'data2.csv',
            'capture1.pcap'
        ]
        
        print(f"Processing {len(fileList)} files...\n")
        
        results = orchestrator.batchProcessing(
            fileList=fileList,
            modelType='both',
            aggregate=True
        )
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"File {i}: {result['file']}")
            print(f"  Status: {result['status']}")
            if result['status'] == 'success':
                print(f"  Prediction: {result['result'].get('predictions')}")
            else:
                print(f"  Error: {result.get('error')}")
        
        orchestrator.shutdown()
        print("\n✓ Batch processing complete")
    else:
        print("✗ Failed to initialize services")


def example4_configuration_management():
    """Example 4: Configuration management."""
    print("\n" + "=" * 60)
    print("Example 4: Configuration Management")
    print("=" * 60)
    
    configService = ConfigurationService()
    configService.start()
    
    # Get configuration
    lstmConfig = configService.getConfiguration('lstmModel')
    print(f"LSTM Hidden Size: {lstmConfig['hiddenSize']}")
    
    # Update configuration
    configService.setConfiguration('lstmModel.hiddenSize', 256)
    print(f"Updated LSTM Hidden Size: {configService.getConfiguration('lstmModel.hiddenSize')}")
    
    # Validate configuration
    validation = configService.validateConfiguration()
    print(f"Configuration valid: {validation['valid']}")
    
    # Save configuration
    configService.saveConfiguration('my_config.yaml', format='yaml')
    print("✓ Configuration saved to my_config.yaml")
    
    # View history
    history = configService.getHistory(limit=5)
    print(f"\nConfiguration changes: {len(history)}")
    
    configService.stop()


def example5_output_management():
    """Example 5: Output management and aggregation."""
    print("\n" + "=" * 60)
    print("Example 5: Output Management")
    print("=" * 60)
    
    outputService = OutputManagementService({
        'baseOutputDir': './outputs',
        'maxBufferSize': 10
    })
    outputService.start()
    
    # Simulate LSTM results
    lstmResults = {
        'modelType': 'LSTM',
        'predictions': [0, 1, 0],
        'probabilities': [[0.9, 0.1], [0.3, 0.7], [0.8, 0.2]],
        'confidences': [0.9, 0.7, 0.8]
    }
    
    # Simulate CNN results
    cnnResults = {
        'modelType': 'CNN',
        'predictions': [0, 1, 0],
        'probabilities': [[0.85, 0.15], [0.2, 0.8], [0.75, 0.25]],
        'confidences': [0.85, 0.8, 0.75]
    }
    
    # Store individual results
    outputService.process(lstmResults)
    outputService.process(cnnResults)
    print("✓ Individual results stored")
    
    # Aggregate results
    aggregated = outputService.aggregateResults(
        lstmResults,
        cnnResults,
        aggregationMethod='voting'
    )
    
    print(f"\nAggregated Predictions: {aggregated['predictions']}")
    print(f"Confidence: {aggregated['confidence']:.4f}")
    print(f"Method: {aggregated['aggregationMethod']}")
    
    # Generate report
    report = outputService.generateReport()
    print(f"\nOutput counts:")
    for outputType, count in report['outputCounts'].items():
        print(f"  {outputType}: {count}")
    
    outputService.stop()


def example6_system_status():
    """Example 6: Monitoring system status."""
    print("\n" + "=" * 60)
    print("Example 6: System Status Monitoring")
    print("=" * 60)
    
    orchestrator = ServiceOrchestrator()
    
    if orchestrator.initialize():
        # Get system status
        status = orchestrator.getSystemStatus()
        
        print(f"System Initialized: {status['initialized']}\n")
        print("Service Status:")
        for serviceName, serviceInfo in status['services'].items():
            print(f"  {serviceName}:")
            print(f"    Status: {serviceInfo.get('status')}")
            if 'uptime' in serviceInfo and serviceInfo['uptime']:
                print(f"    Uptime: {serviceInfo['uptime']:.2f}s")
        
        orchestrator.shutdown()
    else:
        print("✗ Failed to initialize services")


def example7_custom_configuration():
    """Example 7: Using custom configuration."""
    print("\n" + "=" * 60)
    print("Example 7: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    customConfig = {
        'lstmModel': {
            'inputSize': 128,
            'hiddenSize': 64,
            'numLayers': 1,
            'numClasses': 5,
            'dropoutRate': 0.2,
            'useAttention': False,
            'outputDir': './custom_outputs/lstm'
        }
    }
    
    # Initialize service with custom config
    lstmService = LSTMModelService(customConfig['lstmModel'])
    lstmService.start()
    
    print("✓ LSTM Service with custom configuration")
    info = lstmService.getModelInfo()
    print(f"  Input Size: {info['inputSize']}")
    print(f"  Hidden Size: {info['hiddenSize']}")
    print(f"  Num Layers: {info['numLayers']}")
    print(f"  Use Attention: {info['useAttention']}")
    
    lstmService.stop()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("P22 Encrypted Traffic IDS - Example Usage")
    print("=" * 60)
    print("\nNote: Some examples use dummy data for demonstration.")
    print("Replace with actual data files for real usage.\n")
    
    try:
        # Run examples
        # Note: Comment out examples that require actual data files
        
        # example1_basic_usage()  # Requires actual data file
        example2_individual_services()
        # example3_batch_processing()  # Requires actual data files
        example4_configuration_management()
        example5_output_management()
        example6_system_status()
        example7_custom_configuration()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
