"""
Output Management Service

Microservice for managing and aggregating outputs from:
- LSTM Model Service
- CNN Model Service
- Ensemble predictions

Provides separated storage and final aggregated results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

from .baseService import BaseService


class OutputManagementService(BaseService):
    """Output Management Service for handling prediction results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Output Management Service.
        
        Args:
            config: Service configuration
        """
        super().__init__("OutputManagementService", config)
        
        self.baseOutputDir = Path(self.getConfig('baseOutputDir', './outputs'))
        self.lstmOutputDir = self.baseOutputDir / 'lstm'
        self.cnnOutputDir = self.baseOutputDir / 'cnn'
        self.ensembleOutputDir = self.baseOutputDir / 'ensemble'
        self.finalOutputDir = self.baseOutputDir / 'final'
        
        # Create output directories
        for directory in [
            self.lstmOutputDir, 
            self.cnnOutputDir, 
            self.ensembleOutputDir,
            self.finalOutputDir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.resultsBuffer = []
        self.maxBufferSize = self.getConfig('maxBufferSize', 100)
        
    def _onStart(self) -> None:
        """Initialize output management on service start."""
        self.logger.info("Output Management Service started")
        self.resultsBuffer = []
    
    def _onStop(self) -> None:
        """Cleanup on service stop."""
        # Flush any buffered results
        if self.resultsBuffer:
            self._flushBuffer()
        self.resultsBuffer = []
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and store model outputs.
        
        Args:
            data: Dictionary containing:
                - modelType: 'LSTM', 'CNN', or 'Ensemble'
                - predictions: Model predictions
                - metadata: Additional information
                
        Returns:
            Dictionary containing storage information
        """
        if not self.isRunning:
            raise RuntimeError("Output Management Service is not running")
        
        modelType = data.get('modelType', 'Unknown')
        
        try:
            # Store output based on model type
            if modelType == 'LSTM':
                storagePath = self._storeLSTMOutput(data)
            elif modelType == 'CNN':
                storagePath = self._storeCNNOutput(data)
            elif modelType == 'Ensemble':
                storagePath = self._storeEnsembleOutput(data)
            else:
                storagePath = self._storeGenericOutput(data)
            
            # Add to buffer
            self.resultsBuffer.append(data)
            
            # Flush buffer if full
            if len(self.resultsBuffer) >= self.maxBufferSize:
                self._flushBuffer()
            
            return {
                'status': 'success',
                'storagePath': str(storagePath),
                'modelType': modelType,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error storing output: {str(e)}")
            raise
    
    def _storeLSTMOutput(self, data: Dict[str, Any]) -> Path:
        """
        Store LSTM model output.
        
        Args:
            data: LSTM output data
            
        Returns:
            Path to stored file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        outputFile = self.lstmOutputDir / f"lstm_prediction_{timestamp}.json"
        
        with open(outputFile, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.debug(f"Stored LSTM output: {outputFile}")
        return outputFile
    
    def _storeCNNOutput(self, data: Dict[str, Any]) -> Path:
        """
        Store CNN model output.
        
        Args:
            data: CNN output data
            
        Returns:
            Path to stored file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        outputFile = self.cnnOutputDir / f"cnn_prediction_{timestamp}.json"
        
        with open(outputFile, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.debug(f"Stored CNN output: {outputFile}")
        return outputFile
    
    def _storeEnsembleOutput(self, data: Dict[str, Any]) -> Path:
        """
        Store ensemble model output.
        
        Args:
            data: Ensemble output data
            
        Returns:
            Path to stored file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        outputFile = self.ensembleOutputDir / f"ensemble_prediction_{timestamp}.json"
        
        with open(outputFile, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.debug(f"Stored ensemble output: {outputFile}")
        return outputFile
    
    def _storeGenericOutput(self, data: Dict[str, Any]) -> Path:
        """
        Store generic output.
        
        Args:
            data: Output data
            
        Returns:
            Path to stored file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        modelType = data.get('modelType', 'unknown')
        outputFile = self.finalOutputDir / f"{modelType}_prediction_{timestamp}.json"
        
        with open(outputFile, 'w') as f:
            json.dump(data, f, indent=2)
        
        return outputFile
    
    def aggregateResults(
        self, 
        lstmResults: Dict[str, Any],
        cnnResults: Dict[str, Any],
        aggregationMethod: str = 'voting'
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple models.
        
        Args:
            lstmResults: LSTM model results
            cnnResults: CNN model results
            aggregationMethod: Method for aggregation ('voting', 'averaging', 'max')
            
        Returns:
            Aggregated results
        """
        try:
            if aggregationMethod == 'voting':
                finalPrediction = self._votingAggregation(lstmResults, cnnResults)
            elif aggregationMethod == 'averaging':
                finalPrediction = self._averagingAggregation(lstmResults, cnnResults)
            elif aggregationMethod == 'max':
                finalPrediction = self._maxConfidenceAggregation(lstmResults, cnnResults)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregationMethod}")
            
            # Create final result
            aggregatedResult = {
                'modelType': 'Ensemble',
                'aggregationMethod': aggregationMethod,
                'predictions': finalPrediction['predictions'],
                'confidence': finalPrediction['confidence'],
                'lstmResults': lstmResults,
                'cnnResults': cnnResults,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store final result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            finalOutputFile = self.finalOutputDir / f"final_prediction_{timestamp}.json"
            
            with open(finalOutputFile, 'w') as f:
                json.dump(aggregatedResult, f, indent=2)
            
            self.logger.info(f"Stored aggregated result: {finalOutputFile}")
            
            return aggregatedResult
            
        except Exception as e:
            self.logger.error(f"Error aggregating results: {str(e)}")
            raise
    
    def _votingAggregation(
        self, 
        lstmResults: Dict[str, Any],
        cnnResults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Majority voting aggregation."""
        lstmPred = lstmResults.get('predictions', [])
        cnnPred = cnnResults.get('predictions', [])
        
        # Simple majority voting
        if lstmPred == cnnPred:
            finalPred = lstmPred
            confidence = (
                np.mean(lstmResults.get('confidences', [0.5])) +
                np.mean(cnnResults.get('confidences', [0.5]))
            ) / 2
        else:
            # Use prediction with higher confidence
            lstmConf = np.mean(lstmResults.get('confidences', [0]))
            cnnConf = np.mean(cnnResults.get('confidences', [0]))
            
            if lstmConf > cnnConf:
                finalPred = lstmPred
                confidence = lstmConf
            else:
                finalPred = cnnPred
                confidence = cnnConf
        
        return {
            'predictions': finalPred,
            'confidence': float(confidence)
        }
    
    def _averagingAggregation(
        self,
        lstmResults: Dict[str, Any],
        cnnResults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Probability averaging aggregation."""
        lstmProbs = np.array(lstmResults.get('probabilities', [[0]]))
        cnnProbs = np.array(cnnResults.get('probabilities', [[0]]))
        
        # Average probabilities
        avgProbs = (lstmProbs + cnnProbs) / 2
        finalPred = [int(np.argmax(p)) for p in avgProbs]
        confidence = float(np.max(avgProbs))
        
        return {
            'predictions': finalPred,
            'confidence': confidence
        }
    
    def _maxConfidenceAggregation(
        self,
        lstmResults: Dict[str, Any],
        cnnResults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select prediction with maximum confidence."""
        lstmConf = np.mean(lstmResults.get('confidences', [0]))
        cnnConf = np.mean(cnnResults.get('confidences', [0]))
        
        if lstmConf > cnnConf:
            return {
                'predictions': lstmResults.get('predictions', []),
                'confidence': float(lstmConf)
            }
        else:
            return {
                'predictions': cnnResults.get('predictions', []),
                'confidence': float(cnnConf)
            }
    
    def _flushBuffer(self) -> None:
        """Flush results buffer to disk."""
        if not self.resultsBuffer:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bufferFile = self.finalOutputDir / f"results_batch_{timestamp}.json"
            
            with open(bufferFile, 'w') as f:
                json.dump(self.resultsBuffer, f, indent=2)
            
            self.logger.info(f"Flushed {len(self.resultsBuffer)} results to {bufferFile}")
            self.resultsBuffer = []
            
        except Exception as e:
            self.logger.error(f"Error flushing buffer: {str(e)}")
    
    def generateReport(
        self, 
        startTime: Optional[datetime] = None,
        endTime: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate summary report for outputs.
        
        Args:
            startTime: Start time for report period
            endTime: End time for report period
            
        Returns:
            Summary report dictionary
        """
        try:
            # Count outputs by type
            lstmCount = len(list(self.lstmOutputDir.glob('*.json')))
            cnnCount = len(list(self.cnnOutputDir.glob('*.json')))
            ensembleCount = len(list(self.ensembleOutputDir.glob('*.json')))
            finalCount = len(list(self.finalOutputDir.glob('*.json')))
            
            report = {
                'generatedAt': datetime.now().isoformat(),
                'outputCounts': {
                    'lstm': lstmCount,
                    'cnn': cnnCount,
                    'ensemble': ensembleCount,
                    'final': finalCount,
                    'total': lstmCount + cnnCount + ensembleCount + finalCount
                },
                'outputDirectories': {
                    'lstm': str(self.lstmOutputDir),
                    'cnn': str(self.cnnOutputDir),
                    'ensemble': str(self.ensembleOutputDir),
                    'final': str(self.finalOutputDir)
                }
            }
            
            # Save report
            reportFile = self.finalOutputDir / f"output_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(reportFile, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Generated output report: {reportFile}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
