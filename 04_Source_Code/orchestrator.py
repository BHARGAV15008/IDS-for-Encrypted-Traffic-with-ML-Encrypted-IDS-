"""
Service Orchestrator

Orchestrates the workflow between all microservices.
Provides high-level operations and manages service dependencies.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from services.configurationService import ConfigurationService
from services.dataIngestionService import DataIngestionService
from services.lstmModelService import LSTMModelService
from services.cnnModelService import CNNModelService
from services.outputManagementService import OutputManagementService


class ServiceOrchestrator:
    """Orchestrator for managing microservices workflow."""
    
    def __init__(self, configPath: Optional[str] = None):
        """
        Initialize Service Orchestrator.
        
        Args:
            configPath: Path to configuration file
        """
        self.configPath = configPath
        self.logger = self._setupLogger()
        
        # Services
        self.configService = None
        self.dataService = None
        self.lstmService = None
        self.cnnService = None
        self.outputService = None
        
        self.isInitialized = False
    
    def _setupLogger(self) -> logging.Logger:
        """Setup orchestrator logger."""
        logger = logging.getLogger("P22.Orchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - Orchestrator - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize(self) -> bool:
        """
        Initialize all services.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing microservices...")
            
            # Initialize Configuration Service
            self.configService = ConfigurationService(self.configPath)
            if not self.configService.start():
                raise RuntimeError("Failed to start Configuration Service")
            
            # Get configurations
            config = self.configService.getConfiguration()
            
            # Initialize Data Ingestion Service
            self.dataService = DataIngestionService(config.get('dataIngestion'))
            if not self.dataService.start():
                raise RuntimeError("Failed to start Data Ingestion Service")
            
            # Initialize LSTM Model Service
            self.lstmService = LSTMModelService(config.get('lstmModel'))
            if not self.lstmService.start():
                raise RuntimeError("Failed to start LSTM Model Service")
            
            # Initialize CNN Model Service
            self.cnnService = CNNModelService(config.get('cnnModel'))
            if not self.cnnService.start():
                raise RuntimeError("Failed to start CNN Model Service")
            
            # Initialize Output Management Service
            self.outputService = OutputManagementService(config.get('outputManagement'))
            if not self.outputService.start():
                raise RuntimeError("Failed to start Output Management Service")
            
            self.isInitialized = True
            self.logger.info("All services initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self.shutdown()
            return False
    
    def shutdown(self):
        """Shutdown all services."""
        self.logger.info("Shutting down services...")
        
        services = [
            self.outputService,
            self.cnnService,
            self.lstmService,
            self.dataService,
            self.configService
        ]
        
        for service in services:
            if service and service.isRunning:
                service.stop()
        
        self.isInitialized = False
        self.logger.info("All services shut down")
    
    def processDataFile(
        self, 
        filePath: str, 
        fileType: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a data file through the ingestion service.
        
        Args:
            filePath: Path to data file
            fileType: Type of file ('csv' or 'pcap')
            
        Returns:
            Processing results
        """
        if not self.isInitialized:
            raise RuntimeError("Services not initialized")
        
        self.logger.info(f"Processing data file: {filePath}")
        
        result = self.dataService.process({
            'filePath': filePath,
            'fileType': fileType
        })
        
        return result
    
    def runInference(
        self,
        data: Dict[str, Any],
        modelType: str = 'both',
        aggregate: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on processed data.
        
        Args:
            data: Processed data dictionary
            modelType: Model to use ('lstm', 'cnn', or 'both')
            aggregate: Whether to aggregate results
            
        Returns:
            Inference results
        """
        if not self.isInitialized:
            raise RuntimeError("Services not initialized")
        
        features = data.get('features')
        metadata = {
            'sourceFile': data.get('filePath'),
            'fileType': data.get('fileType'),
            'timestamp': data.get('timestamp')
        }
        
        lstmResult = None
        cnnResult = None
        
        # Run LSTM inference
        if modelType in ['lstm', 'both']:
            self.logger.info("Running LSTM inference...")
            lstmResult = self.lstmService.process({
                'features': features,
                'metadata': metadata
            })
            self.outputService.process(lstmResult)
        
        # Run CNN inference
        if modelType in ['cnn', 'both']:
            self.logger.info("Running CNN inference...")
            cnnResult = self.cnnService.process({
                'features': features,
                'metadata': metadata
            })
            self.outputService.process(cnnResult)
        
        # Aggregate results if both models used
        if aggregate and lstmResult and cnnResult:
            self.logger.info("Aggregating results...")
            aggregatedResult = self.outputService.aggregateResults(
                lstmResult,
                cnnResult,
                aggregationMethod=self.configService.getConfiguration(
                    'outputManagement.aggregationMethod'
                )
            )
            return aggregatedResult
        
        return lstmResult or cnnResult
    
    def runEndToEndPipeline(
        self,
        filePath: str,
        modelType: str = 'both',
        aggregate: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete end-to-end pipeline.
        
        Args:
            filePath: Path to data file
            modelType: Model to use
            aggregate: Whether to aggregate results
            
        Returns:
            Pipeline results
        """
        if not self.isInitialized:
            raise RuntimeError("Services not initialized")
        
        self.logger.info(f"Starting end-to-end pipeline for: {filePath}")
        
        # Step 1: Process data
        processedData = self.processDataFile(filePath)
        
        # Step 2: Run inference
        results = self.runInference(processedData, modelType, aggregate)
        
        self.logger.info("End-to-end pipeline completed")
        
        return results
    
    def batchProcessing(
        self,
        fileList: List[str],
        modelType: str = 'both',
        aggregate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files in batch.
        
        Args:
            fileList: List of file paths
            modelType: Model to use
            aggregate: Whether to aggregate results
            
        Returns:
            List of results for each file
        """
        if not self.isInitialized:
            raise RuntimeError("Services not initialized")
        
        results = []
        
        for filePath in fileList:
            try:
                self.logger.info(f"Processing file {len(results) + 1}/{len(fileList)}: {filePath}")
                result = self.runEndToEndPipeline(filePath, modelType, aggregate)
                results.append({
                    'file': filePath,
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                self.logger.error(f"Error processing {filePath}: {str(e)}")
                results.append({
                    'file': filePath,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def getSystemStatus(self) -> Dict[str, Any]:
        """
        Get status of all services.
        
        Returns:
            System status dictionary
        """
        services = {
            'configuration': self.configService,
            'dataIngestion': self.dataService,
            'lstmModel': self.lstmService,
            'cnnModel': self.cnnService,
            'outputManagement': self.outputService
        }
        
        status = {
            'initialized': self.isInitialized,
            'services': {}
        }
        
        for name, service in services.items():
            if service:
                status['services'][name] = service.healthCheck()
            else:
                status['services'][name] = {'status': 'not_initialized'}
        
        return status
    
    def updateConfiguration(self, updates: Dict[str, Any]) -> bool:
        """
        Update system configuration.
        
        Args:
            updates: Configuration updates
            
        Returns:
            True if update successful
        """
        if not self.configService:
            return False
        
        try:
            self.configService.updateConfiguration(updates)
            
            # Propagate updates to relevant services
            if 'lstmModel' in updates and self.lstmService:
                self.lstmService.updateConfig(updates['lstmModel'])
            
            if 'cnnModel' in updates and self.cnnService:
                self.cnnService.updateConfig(updates['cnnModel'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = ServiceOrchestrator()
    
    if orchestrator.initialize():
        print("Services initialized successfully")
        
        # Display system status
        status = orchestrator.getSystemStatus()
        print(f"\nSystem Status:")
        for service, info in status['services'].items():
            print(f"  {service}: {info['status']}")
        
        # Shutdown
        orchestrator.shutdown()
    else:
        print("Failed to initialize services")
