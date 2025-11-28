"""
Hybrid CNN-BiLSTM-Attention Model
Combines spatial feature extraction (CNN), temporal modeling (BiLSTM), 
and automated feature discovery (Attention) for encrypted traffic IDS
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import logging
import importlib.util
import sys

# Dynamically load CNNFeatureExtractor and CNNModelFactory
try:
    spec_cnn_module = importlib.util.spec_from_file_location(
        "cnnModule",
        "E:/IDS for Encrypted Traffic with ML (Encrypted IDS)/03_Models/01_Architectures/cnnModule.py"
    )
    cnnModule_module = importlib.util.module_from_spec(spec_cnn_module)
    spec_cnn_module.loader.exec_module(cnnModule_module)
    CNNFeatureExtractor = cnnModule_module.CNNFeatureExtractor
    CNNModelFactory = cnnModule_module.CNNModelFactory
except Exception as e:
    print(f"Error loading CNNModule: {e}")
    sys.exit(1)

# Dynamically load LSTMFeatureExtractor and LSTMModelFactory
try:
    spec_lstm_module = importlib.util.spec_from_file_location(
        "lstmModule",
        "E:/IDS for Encrypted Traffic with ML (Encrypted IDS)/03_Models/01_Architectures/lstmModule.py"
    )
    lstmModule_module = importlib.util.module_from_spec(spec_lstm_module)
    spec_lstm_module.loader.exec_module(lstmModule_module)
    LSTMFeatureExtractor = lstmModule_module.LSTMFeatureExtractor
    LSTMModelFactory = lstmModule_module.LSTMModelFactory
except Exception as e:
    print(f"Error loading LSTMModule: {e}")
    sys.exit(1)

# Dynamically load AttentionBlock, TemporalAttention, FeatureAttention
try:
    spec_attention_module = importlib.util.spec_from_file_location(
        "attentionModule",
        "E:/IDS for Encrypted Traffic with ML (Encrypted IDS)/03_Models/01_Architectures/attentionModule.py"
    )
    attentionModule_module = importlib.util.module_from_spec(spec_attention_module)
    spec_attention_module.loader.exec_module(attentionModule_module)
    AttentionBlock = attentionModule_module.AttentionBlock
    TemporalAttention = attentionModule_module.TemporalAttention
    FeatureAttention = attentionModule_module.FeatureAttention
except Exception as e:
    print(f"Error loading AttentionModule: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


class HybridCnnBiLstmAttention(nn.Module):
    """
    Advanced Hybrid Model for Encrypted Traffic IDS
    Architecture: CNN -> BiLSTM -> Attention -> Classification
    """
    
    def __init__(
        self,
        inputFeatures: int,
        numClasses: int,
        cnnChannels: list = [64, 128, 256],
        lstmHiddenSize: int = 128,
        lstmNumLayers: int = 2,
        attentionHeads: int = 8,
        dropout: float = 0.3,
        useCnnResidual: bool = True,
        useFeatureAttention: bool = True,
        useTemporalAttention: bool = True
    ):
        """
        Args:
            inputFeatures: Number of input features
            numClasses: Number of output classes
            cnnChannels: List of CNN channel dimensions
            lstmHiddenSize: LSTM hidden size
            lstmNumLayers: Number of LSTM layers
            attentionHeads: Number of attention heads
            dropout: Dropout rate
            useCnnResidual: Use residual connections in CNN
            useFeatureAttention: Use feature-level attention
            useTemporalAttention: Use temporal attention
        """
        super().__init__()
        
        self.inputFeatures = inputFeatures
        self.numClasses = numClasses
        self.useFeatureAttention = useFeatureAttention
        self.useTemporalAttention = useTemporalAttention
        
        # Feature attention (optional)
        if useFeatureAttention:
            self.featureAttention = FeatureAttention(
                numFeatures=inputFeatures,
                attentionDim=64
            )
        
        # CNN for spatial feature extraction
        self.cnn = CNNFeatureExtractor(
            inputFeatures=inputFeatures,
            hiddenChannels=cnnChannels,
            kernelSize=3,
            architecture='residual' if useCnnResidual else 'standard',
            useResidual=useCnnResidual,
            poolingType='max',
            dropoutRate=dropout,
            outputDim=cnnChannels[-1]
        )
        
        # BiLSTM for temporal modeling
        self.biLstm = LSTMFeatureExtractor(
            inputSize=cnnChannels[-1],
            hiddenSize=lstmHiddenSize,
            numLayers=lstmNumLayers,
            dropout=dropout,
            bidirectional=True,
            architecture='standard',
            outputDim=lstmHiddenSize * 2
        )
        
        # Self-attention for feature weighting
        self.selfAttention = AttentionBlock(
            embedDim=lstmHiddenSize * 2,
            numHeads=attentionHeads,
            ffnDim=lstmHiddenSize * 4,
            dropout=dropout
        )
        
        # Temporal attention for sequence aggregation (optional)
        if useTemporalAttention:
            self.temporalAttention = TemporalAttention(
                hiddenDim=lstmHiddenSize * 2
            )
        
        # Classification head
        classifierInputDim = lstmHiddenSize * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifierInputDim, lstmHiddenSize),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstmHiddenSize, lstmHiddenSize // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstmHiddenSize // 2, numClasses)
        )
        
        logger.info(f"Hybrid CNN-BiLSTM-Attention initialized: "
                   f"input_features={inputFeatures}, num_classes={numClasses}, "
                   f"cnn_channels={cnnChannels}, lstm_hidden={lstmHiddenSize}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_features)
            
        Returns:
            logits: (batch, num_classes)
        """
        # Feature attention (optional)
        if self.useFeatureAttention:
            x, _ = self.featureAttention(x)
        
        # CNN spatial feature extraction
        cnnOut = self.cnn(x)  # (batch, seq_len', cnn_channels[-1])
        
        # BiLSTM temporal modeling
        lstmOut = self.biLstm(cnnOut)  # (batch, seq_len', lstm_hidden * 2)
        
        # Self-attention for feature weighting
        attnOut = self.selfAttention(lstmOut)  # (batch, seq_len', lstm_hidden * 2)
        
        # Temporal attention for aggregation
        if self.useTemporalAttention:
            context, _ = self.temporalAttention(attnOut)  # (batch, lstm_hidden * 2)
        else:
            # Use mean pooling
            context = attnOut.mean(dim=1)  # (batch, lstm_hidden * 2)
        
        # Classification
        logits = self.classifier(context)  # (batch, num_classes)
        
        return logits
    
    def extractFeatures(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract deep features without classification
        
        Args:
            x: Input tensor (batch, seq_len, input_features)
            
        Returns:
            features: (batch, lstm_hidden * 2)
        """
        if self.useFeatureAttention:
            x, _ = self.featureAttention(x)
        
        cnnOut = self.cnn(x)
        lstmOut = self.biLstm(cnnOut)
        attnOut = self.selfAttention(lstmOut)
        
        if self.useTemporalAttention:
            features, _ = self.temporalAttention(attnOut)
        else:
            features = attnOut.mean(dim=1)
        
        return features


class MultiScaleHybridModel(nn.Module):
    """
    Multi-scale hybrid model with parallel CNN paths
    Captures patterns at different temporal scales
    """
    
    def __init__(
        self,
        inputFeatures: int,
        numClasses: int,
        scales: list = [3, 5, 7],
        lstmHiddenSize: int = 128,
        dropout: float = 0.3
    ):
        """
        Args:
            inputFeatures: Number of input features
            numClasses: Number of output classes
            scales: List of kernel sizes for multi-scale CNN
            lstmHiddenSize: LSTM hidden size
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-scale CNN branches
        self.cnnBranches = nn.ModuleList([
            CNNFeatureExtractor(
                inputFeatures=inputFeatures,
                hiddenChannels=[64, 128],
                kernelSize=scale,
                poolingType='max',
                dropoutRate=dropout
            )
            for scale in scales
        ])
        
        # Fusion layer
        fusionInputDim = 128 * len(scales)
        self.fusion = nn.Linear(fusionInputDim, 256)
        
        # BiLSTM
        self.biLstm = LSTMFeatureExtractor(
            inputSize=256,
            hiddenSize=lstmHiddenSize,
            numLayers=2,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention
        self.attention = AttentionBlock(
            embedDim=lstmHiddenSize * 2,
            numHeads=8,
            dropout=dropout
        )
        
        # Temporal pooling
        self.temporalAttention = TemporalAttention(lstmHiddenSize * 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstmHiddenSize * 2, lstmHiddenSize),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstmHiddenSize, numClasses)
        )
        
        logger.info(f"Multi-scale Hybrid Model initialized: scales={scales}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_features)
            
        Returns:
            logits: (batch, num_classes)
        """
        # Process through multi-scale CNN branches
        branchOutputs = []
        for cnnBranch in self.cnnBranches:
            branchOut = cnnBranch(x)
            branchOutputs.append(branchOut)
        
        # Concatenate branch outputs
        multiScale = torch.cat(branchOutputs, dim=-1)
        
        # Fusion
        fused = self.fusion(multiScale)
        
        # BiLSTM
        lstmOut = self.biLstm(fused)
        
        # Attention
        attnOut = self.attention(lstmOut)
        
        # Temporal aggregation
        context, _ = self.temporalAttention(attnOut)
        
        # Classification
        logits = self.classifier(context)
        
        return logits


class DeepHybridModel(nn.Module):
    """
    Deep hybrid model with multiple attention layers
    For complex pattern recognition in encrypted traffic
    """
    
    def __init__(
        self,
        inputFeatures: int,
        numClasses: int,
        cnnChannels: list = [64, 128, 256, 512],
        lstmHiddenSize: int = 256,
        numAttentionLayers: int = 3,
        dropout: float = 0.3
    ):
        """
        Args:
            inputFeatures: Number of input features
            numClasses: Number of output classes
            cnnChannels: CNN channel dimensions
            lstmHiddenSize: LSTM hidden size
            numAttentionLayers: Number of attention layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Deep CNN
        self.cnn = CNNFeatureExtractor(
            inputFeatures=inputFeatures,
            hiddenChannels=cnnChannels,
            architecture='residual',
            useResidual=True,
            dropoutRate=dropout
        )
        
        # Stacked BiLSTM
        self.biLstm = LSTMFeatureExtractor(
            inputSize=cnnChannels[-1],
            hiddenSize=lstmHiddenSize,
            numLayers=3,
            dropout=dropout,
            bidirectional=True,
            architecture='stacked'
        )
        
        # Multiple attention layers
        self.attentionLayers = nn.ModuleList([
            AttentionBlock(
                embedDim=lstmHiddenSize * 2,
                numHeads=8,
                ffnDim=lstmHiddenSize * 4,
                dropout=dropout
            )
            for _ in range(numAttentionLayers)
        ])
        
        # Global attention pooling
        self.globalAttention = TemporalAttention(lstmHiddenSize * 2)
        
        # Deep classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstmHiddenSize * 2, lstmHiddenSize),
            nn.ReLU(),
            nn.BatchNorm1d(lstmHiddenSize),
            nn.Dropout(dropout),
            nn.Linear(lstmHiddenSize, lstmHiddenSize // 2),
            nn.ReLU(),
            nn.BatchNorm1d(lstmHiddenSize // 2),
            nn.Dropout(dropout),
            nn.Linear(lstmHiddenSize // 2, numClasses)
        )
        
        logger.info(f"Deep Hybrid Model initialized: "
                   f"cnn_depth={len(cnnChannels)}, attention_layers={numAttentionLayers}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # CNN
        x = self.cnn(x)
        
        # BiLSTM
        x = self.biLstm(x)
        
        # Multiple attention layers
        for attnLayer in self.attentionLayers:
            x = attnLayer(x)
        
        # Global pooling
        x, _ = self.globalAttention(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class LightweightHybridModel(nn.Module):
    """
    Lightweight hybrid model for resource-constrained deployment
    Optimized for speed and efficiency
    """
    
    def __init__(
        self,
        inputFeatures: int,
        numClasses: int,
        hiddenDim: int = 64,
        dropout: float = 0.2
    ):
        """
        Args:
            inputFeatures: Number of input features
            numClasses: Number of output classes
            hiddenDim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Lightweight CNN
        self.cnn = CNNFeatureExtractor(
            inputFeatures=inputFeatures,
            hiddenChannels=[hiddenDim, hiddenDim * 2],
            kernelSize=3,
            poolingType='max',
            dropoutRate=dropout
        )
        
        # Single BiLSTM layer
        self.biLstm = LSTMFeatureExtractor(
            inputSize=hiddenDim * 2,
            hiddenSize=hiddenDim,
            numLayers=1,
            dropout=dropout,
            bidirectional=True
        )
        
        # Simple attention
        self.attention = TemporalAttention(hiddenDim * 2)
        
        # Compact classifier
        self.classifier = nn.Sequential(
            nn.Linear(hiddenDim * 2, hiddenDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenDim, numClasses)
        )
        
        logger.info(f"Lightweight Hybrid Model initialized: hidden_dim={hiddenDim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.cnn(x)
        x = self.biLstm(x)
        x, _ = self.attention(x)
        logits = self.classifier(x)
        return logits


class HybridModelFactory:
    """Factory for creating different hybrid model architectures"""
    
    @staticmethod
    def create(
        modelType: str,
        inputFeatures: int,
        numClasses: int,
        **kwargs
    ) -> nn.Module:
        """
        Create hybrid model
        
        Args:
            modelType: Type of model ('standard', 'multi_scale', 'deep', 'lightweight')
            inputFeatures: Number of input features
            numClasses: Number of output classes
            **kwargs: Additional model parameters
            
        Returns:
            Hybrid model
        """
        if modelType == 'standard':
            return HybridCnnBiLstmAttention(
                inputFeatures=inputFeatures,
                numClasses=numClasses,
                **kwargs
            )
        elif modelType == 'multi_scale':
            return MultiScaleHybridModel(
                inputFeatures=inputFeatures,
                numClasses=numClasses,
                **kwargs
            )
        elif modelType == 'deep':
            return DeepHybridModel(
                inputFeatures=inputFeatures,
                numClasses=numClasses,
                **kwargs
            )
        elif modelType == 'lightweight':
            return LightweightHybridModel(
                inputFeatures=inputFeatures,
                numClasses=numClasses,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {modelType}")
