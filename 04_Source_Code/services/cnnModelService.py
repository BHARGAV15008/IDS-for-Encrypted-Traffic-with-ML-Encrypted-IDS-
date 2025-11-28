"""
CNN Model Service

Microservice for 1D CNN spatial pattern detection.
Handles spatial feature extraction from packet-level data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from .baseService import BaseService


class CNNModel(nn.Module):
    """1D CNN model for spatial feature extraction."""
    
    def __init__(
        self,
        inputChannels: int,
        sequenceLength: int,
        numClasses: int,
        convLayers: List[int] = [64, 128, 256],
        kernelSizes: List[int] = [3, 5, 7],
        dropoutRate: float = 0.3
    ):
        """
        Initialize CNN model.
        
        Args:
            inputChannels: Number of input channels
            sequenceLength: Length of input sequence
            numClasses: Number of output classes
            convLayers: List of filter sizes for conv layers
            kernelSizes: List of kernel sizes for multi-scale extraction
            dropoutRate: Dropout probability
        """
        super(CNNModel, self).__init__()
        
        self.inputChannels = inputChannels
        self.sequenceLength = sequenceLength
        self.numClasses = numClasses
        self.convLayers = convLayers
        self.kernelSizes = kernelSizes
        
        # Multi-scale convolutional branches
        self.convBranches = nn.ModuleList()
        
        for kernelSize in kernelSizes:
            branch = nn.ModuleList()
            inChannels = inputChannels
            
            for outChannels in convLayers:
                branch.append(nn.Sequential(
                    nn.Conv1d(
                        inChannels, 
                        outChannels, 
                        kernel_size=kernelSize,
                        padding=kernelSize // 2
                    ),
                    nn.BatchNorm1d(outChannels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropoutRate)
                ))
                inChannels = outChannels
            
            self.convBranches.append(branch)
        
        # Calculate feature dimension after pooling
        self.featureSize = sum(convLayers) * len(kernelSizes)
        
        # Feature fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(self.featureSize, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropoutRate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropoutRate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropoutRate),
            nn.Linear(128, numClasses)
        )
        
        self._initializeWeights()
    
    def _initializeWeights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN model.
        
        Args:
            x: Input tensor (batch_size, channels, sequence_length)
            
        Returns:
            Classification logits
        """
        branchOutputs = []
        
        # Process through each multi-scale branch
        for branch in self.convBranches:
            branchX = x
            for convLayer in branch:
                branchX = convLayer(branchX)
            
            # Global average pooling
            branchFeatures = F.adaptive_avg_pool1d(branchX, 1).squeeze(-1)
            branchOutputs.append(branchFeatures)
        
        # Concatenate features from all branches
        combinedFeatures = torch.cat(branchOutputs, dim=1)
        
        # Classification
        logits = self.classifier(combinedFeatures)
        
        return logits


class CNNModelService(BaseService):
    """CNN Model Service for spatial pattern detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CNN Model Service.
        
        Args:
            config: Service configuration
        """
        super().__init__("CNNModelService", config)
        
        self.model = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.modelPath = None
        self.outputDir = Path(self.getConfig('outputDir', './outputs/cnn'))
        self.outputDir.mkdir(parents=True, exist_ok=True)
        
    def _onStart(self) -> None:
        """Initialize CNN model on service start."""
        # Load model configuration
        inputChannels = self.getConfig('inputChannels', 1)
        sequenceLength = self.getConfig('sequenceLength', 1500)
        numClasses = self.getConfig('numClasses', 10)
        convLayers = self.getConfig('convLayers', [64, 128, 256])
        kernelSizes = self.getConfig('kernelSizes', [3, 5, 7])
        dropoutRate = self.getConfig('dropoutRate', 0.3)
        
        # Initialize model
        self.model = CNNModel(
            inputChannels=inputChannels,
            sequenceLength=sequenceLength,
            numClasses=numClasses,
            convLayers=convLayers,
            kernelSizes=kernelSizes,
            dropoutRate=dropoutRate
        ).to(self.device)
        
        # Load pretrained weights if available
        modelPath = self.getConfig('modelPath', None)
        if modelPath and Path(modelPath).exists():
            self.loadModel(modelPath)
        
        self.model.eval()
        self.logger.info(f"CNN Model initialized on device: {self.device}")
    
    def _onStop(self) -> None:
        """Cleanup on service stop."""
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through CNN model.
        
        Args:
            data: Dictionary containing:
                - features: Input tensor or numpy array
                - metadata: Optional metadata
                
        Returns:
            Dictionary containing predictions and confidence scores
        """
        if not self.isRunning or self.model is None:
            raise RuntimeError("CNN Model Service is not running")
        
        try:
            # Extract features
            features = data.get('features')
            if features is None:
                raise ValueError("No features provided in input data")
            
            # Convert to tensor
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            
            # Ensure proper shape (batch_size, channels, sequence_length)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            if features.dim() == 1:
                features = features.unsqueeze(0).unsqueeze(0)
            
            features = features.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(features)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            
            # Prepare output
            result = {
                'modelType': 'CNN',
                'predictions': predictions.cpu().numpy().tolist(),
                'probabilities': probabilities.cpu().numpy().tolist(),
                'confidences': confidences.cpu().numpy().tolist(),
                'metadata': data.get('metadata', {})
            }
            
            # Save output
            self._saveOutput(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise
    
    def loadModel(self, modelPath: str) -> None:
        """
        Load pretrained model weights.
        
        Args:
            modelPath: Path to model checkpoint
        """
        try:
            checkpoint = torch.load(modelPath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.modelPath = modelPath
            self.logger.info(f"Loaded model from {modelPath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def saveModel(self, savePath: str, metadata: Optional[Dict] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            savePath: Path to save checkpoint
            metadata: Optional metadata to save with model
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'metadata': metadata or {}
            }
            torch.save(checkpoint, savePath)
            self.logger.info(f"Saved model to {savePath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def _saveOutput(self, result: Dict[str, Any]) -> None:
        """
        Save prediction output to file.
        
        Args:
            result: Prediction results
        """
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            outputFile = self.outputDir / f"cnn_output_{timestamp}.json"
            
            with open(outputFile, 'w') as f:
                json.dump(result, f, indent=2)
                
            self.logger.debug(f"Saved output to {outputFile}")
        except Exception as e:
            self.logger.warning(f"Failed to save output: {str(e)}")
    
    def getModelInfo(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model details
        """
        if self.model is None:
            return {'status': 'not_initialized'}
        
        return {
            'modelType': '1D-CNN',
            'inputChannels': self.model.inputChannels,
            'sequenceLength': self.model.sequenceLength,
            'numClasses': self.model.numClasses,
            'convLayers': self.model.convLayers,
            'kernelSizes': self.model.kernelSizes,
            'device': str(self.device),
            'modelPath': self.modelPath,
            'parameterCount': sum(p.numel() for p in self.model.parameters())
        }
