"""
LSTM Model Service

Microservice for Bi-directional LSTM temporal pattern detection.
Handles temporal dependencies and long-range correlations in flow data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from .baseService import BaseService


class BiLSTMModel(nn.Module):
    """Bi-directional LSTM model for temporal pattern recognition."""
    
    def __init__(
        self,
        inputSize: int,
        hiddenSize: int,
        numLayers: int,
        numClasses: int,
        dropoutRate: float = 0.3,
        useAttention: bool = True
    ):
        """
        Initialize BiLSTM model.
        
        Args:
            inputSize: Size of input features
            hiddenSize: LSTM hidden state size
            numLayers: Number of LSTM layers
            numClasses: Number of output classes
            dropoutRate: Dropout probability
            useAttention: Whether to use attention mechanism
        """
        super(BiLSTMModel, self).__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numClasses = numClasses
        self.useAttention = useAttention
        
        # Bi-directional LSTM layers
        self.lstm = nn.LSTM(
            input_size=inputSize,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            bidirectional=True,
            dropout=dropoutRate if numLayers > 1 else 0,
            batch_first=True
        )
        
        lstmOutputSize = hiddenSize * 2  # Bidirectional
        
        # Attention mechanism
        if useAttention:
            self.attentionWeights = nn.Linear(lstmOutputSize, 1, bias=False)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropoutRate),
            nn.Linear(lstmOutputSize, 256),
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
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BiLSTM model.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Classification logits
        """
        # LSTM forward pass
        lstmOut, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        if self.useAttention:
            # Calculate attention scores
            attentionScores = self.attentionWeights(lstmOut).squeeze(-1)
            attentionWeights = torch.softmax(attentionScores, dim=1)
            
            # Weighted sum
            contextVector = torch.sum(
                lstmOut * attentionWeights.unsqueeze(-1), dim=1
            )
        else:
            # Use final hidden state
            # Concatenate forward and backward final hidden states
            contextVector = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Classification
        logits = self.classifier(contextVector)
        
        return logits


class LSTMModelService(BaseService):
    """LSTM Model Service for temporal pattern detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LSTM Model Service.
        
        Args:
            config: Service configuration
        """
        super().__init__("LSTMModelService", config)
        
        self.model = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.modelPath = None
        self.outputDir = Path(self.getConfig('outputDir', './outputs/lstm'))
        self.outputDir.mkdir(parents=True, exist_ok=True)
        
    def _onStart(self) -> None:
        """Initialize LSTM model on service start."""
        # Load model configuration
        inputSize = self.getConfig('inputSize', 256)
        hiddenSize = self.getConfig('hiddenSize', 128)
        numLayers = self.getConfig('numLayers', 2)
        numClasses = self.getConfig('numClasses', 10)
        dropoutRate = self.getConfig('dropoutRate', 0.3)
        useAttention = self.getConfig('useAttention', True)
        
        # Initialize model
        self.model = BiLSTMModel(
            inputSize=inputSize,
            hiddenSize=hiddenSize,
            numLayers=numLayers,
            numClasses=numClasses,
            dropoutRate=dropoutRate,
            useAttention=useAttention
        ).to(self.device)
        
        # Load pretrained weights if available
        modelPath = self.getConfig('modelPath', None)
        if modelPath and Path(modelPath).exists():
            self.loadModel(modelPath)
        
        self.model.eval()
        self.logger.info(f"LSTM Model initialized on device: {self.device}")
    
    def _onStop(self) -> None:
        """Cleanup on service stop."""
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through LSTM model.
        
        Args:
            data: Dictionary containing:
                - features: Input tensor or numpy array
                - metadata: Optional metadata
                
        Returns:
            Dictionary containing predictions and confidence scores
        """
        if not self.isRunning or self.model is None:
            raise RuntimeError("LSTM Model Service is not running")
        
        try:
            # Extract features
            features = data.get('features')
            if features is None:
                raise ValueError("No features provided in input data")
            
            # Convert to tensor
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            
            # Ensure proper shape (batch_size, seq_len, input_size)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            
            features = features.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(features)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            
            # Prepare output
            result = {
                'modelType': 'LSTM',
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
            outputFile = self.outputDir / f"lstm_output_{timestamp}.json"
            
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
            'modelType': 'BiLSTM',
            'inputSize': self.model.inputSize,
            'hiddenSize': self.model.hiddenSize,
            'numLayers': self.model.numLayers,
            'numClasses': self.model.numClasses,
            'useAttention': self.model.useAttention,
            'device': str(self.device),
            'modelPath': self.modelPath,
            'parameterCount': sum(p.numel() for p in self.model.parameters())
        }
