"""
CNN Module for Spatial Feature Extraction
Implements 1D Convolutional Neural Network for network traffic analysis
Extracts local spatial patterns from packet sequences
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class Conv1DBlock(nn.Module):
    """Single 1D Convolutional Block with BatchNorm and Activation"""
    
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        useBatchNorm: bool = True,
        useDropout: bool = True,
        dropoutRate: float = 0.3,
        activation: str = 'relu'
    ):
        super().__init__()
        
        layers = []
        
        # Convolutional layer
        layers.append(nn.Conv1d(
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=kernelSize,
            stride=stride,
            padding=padding,
            dilation=dilation
        ))
        
        # Batch normalization
        if useBatchNorm:
            layers.append(nn.BatchNorm1d(outChannels))
            
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'elu':
            layers.append(nn.ELU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        else:
            raise ValueError(f"Unsupported activation '{activation}'; supported: relu, leaky_relu, elu, gelu")
            
        # Dropout
        if useDropout:
            layers.append(nn.Dropout(dropoutRate))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConv1DBlock(nn.Module):
    """Residual 1D Convolutional Block with skip connection"""
    
    def __init__(
        self,
        channels: int,
        kernelSize: int = 3,
        padding: int = 1,
        useBatchNorm: bool = True,
        dropoutRate: float = 0.3
    ):
        super().__init__()
        
        self.conv1 = Conv1DBlock(
            inChannels=channels,
            outChannels=channels,
            kernelSize=kernelSize,
            padding=padding,
            useBatchNorm=useBatchNorm,
            dropoutRate=dropoutRate
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernelSize, padding=padding),
            nn.BatchNorm1d(channels) if useBatchNorm else nn.Identity()
        )
        
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual  # Skip connection
        out = self.activation(out)
        return out


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN with parallel convolutional paths"""
    
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSizes: List[int] = [3, 5, 7]
    ):
        super().__init__()
        
        self.kernelSizes = kernelSizes
        self.total_branches = len(kernelSizes) + 1  # +1 for 1x1 conv
        
        # Calculate branch channels to ensure total output is exactly outChannels
        base_channels = outChannels // self.total_branches
        remainder = outChannels % self.total_branches
        
        # Distribute remainder channels among branches
        branch_channels = [base_channels + 1 if i < remainder else base_channels 
                         for i in range(self.total_branches)]
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(inChannels, branch_channels[i], kernelSize=k, padding=k//2),
                nn.BatchNorm1d(branch_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i, k in enumerate(kernelSizes)
        ])
        
        # 1x1 conv for dimension matching (last branch)
        self.conv1x1 = nn.Conv1d(inChannels, branch_channels[-1], kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through parallel branches
        branchOutputs = [branch(x) for branch in self.branches]
        
        # Add 1x1 conv output
        branchOutputs.append(self.conv1x1(x))
        
        # Concatenate along channel dimension
        out = torch.cat(branchOutputs, dim=1)
        
        return out


class CNNFeatureExtractor(nn.Module):
    """
    Advanced CNN for extracting spatial features from network traffic
    Supports multiple architectures: standard, residual, multi-scale
    """
    
    def __init__(
        self,
        inputFeatures: int,
        hiddenChannels: List[int] = [64, 128, 256],
        kernelSize: int = 3,
        architecture: str = 'standard',
        useResidual: bool = False,
        useMultiScale: bool = False,
        poolingType: str = 'max',
        dropoutRate: float = 0.3,
        outputDim: Optional[int] = None
    ):
        """
        Args:
            inputFeatures: Number of input features (channels)
            hiddenChannels: List of hidden channel dimensions
            kernelSize: Kernel size for convolutions
            architecture: Architecture type ('standard', 'residual', 'multi_scale')
            useResidual: Whether to use residual connections
            useMultiScale: Whether to use multi-scale convolutions
            poolingType: Type of pooling ('max', 'avg', 'adaptive')
            dropoutRate: Dropout rate
            outputDim: Output dimension (if None, use last hidden channel)
        """
        super().__init__()
        
        self.inputFeatures = inputFeatures
        self.architecture = architecture
        self.outputDim = outputDim if outputDim else hiddenChannels[-1]
        
        layers = []
        currentChannels = inputFeatures
        
        # Build convolutional layers
        for i, hiddenChannel in enumerate(hiddenChannels):
            if architecture == 'multi_scale' or useMultiScale:
                layers.append(MultiScaleCNN(
                    inChannels=currentChannels,
                    outChannels=hiddenChannel
                ))
            elif architecture == 'residual' or (useResidual and i > 0):
                # First layer is standard, rest are residual
                if i == 0:
                    layers.append(Conv1DBlock(
                        inChannels=currentChannels,
                        outChannels=hiddenChannel,
                        kernelSize=kernelSize,
                        dropoutRate=dropoutRate
                    ))
                else:
                    # If input channels don't match desired output channels for residual block,
                    # add a 1x1 conv to project to the correct number of channels.
                    if currentChannels != hiddenChannel:
                        layers.append(nn.Conv1d(currentChannels, hiddenChannel, kernel_size=1))
                        currentChannels = hiddenChannel # Update currentChannels after projection

                    layers.append(ResidualConv1DBlock(
                        channels=hiddenChannel, # Now the residual block operates on the correct number of channels
                        kernelSize=kernelSize,
                        dropoutRate=dropoutRate
                    ))
            else:
                # Standard convolutional block
                layers.append(Conv1DBlock(
                    inChannels=currentChannels,
                    outChannels=hiddenChannel,
                    kernelSize=kernelSize,
                    dropoutRate=dropoutRate
                ))
                
            # Add pooling
            if poolingType == 'max':
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            elif poolingType == 'avg':
                layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
                
            currentChannels = hiddenChannel
            
        self.convLayers = nn.Sequential(*layers)
        
        # Adaptive pooling for fixed output size
        if poolingType == 'adaptive':
            self.adaptivePool = nn.AdaptiveAvgPool1d(1)
        else:
            self.adaptivePool = None
            
        # Output projection
        if self.outputDim != hiddenChannels[-1]:
            self.outputProjection = nn.Linear(hiddenChannels[-1], self.outputDim)
        else:
            self.outputProjection = None
            
        logger.info(f"CNN Feature Extractor initialized: {architecture} architecture, "
                   f"channels={hiddenChannels}, output_dim={self.outputDim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) or (batch, features, seq_len)
            
        Returns:
            Output tensor of shape (batch, seq_len', output_dim) or (batch, output_dim)
        """
        # Ensure input is (batch, features, seq_len)
        if x.dim() == 3 and x.size(2) == self.inputFeatures:
            x = x.transpose(1, 2)  # (batch, seq_len, features) -> (batch, features, seq_len)
            
        # Apply convolutional layers
        x = self.convLayers(x)  # (batch, channels, seq_len')
        
        # Apply adaptive pooling if specified
        if self.adaptivePool is not None:
            x = self.adaptivePool(x)  # (batch, channels, 1)
            x = x.squeeze(-1)  # (batch, channels)
        else:
            x = x.transpose(1, 2)  # (batch, seq_len', channels)
            
        # Apply output projection if specified
        if self.outputProjection is not None:
            if x.dim() == 2:
                x = self.outputProjection(x)  # (batch, output_dim)
            else:
                x = self.outputProjection(x)  # (batch, seq_len', output_dim)
                
        return x


class DilatedCNN(nn.Module):
    """Dilated CNN for capturing multi-scale temporal patterns"""
    
    def __init__(
        self,
        inputFeatures: int,
        hiddenChannels: int = 128,
        numLayers: int = 4,
        kernelSize: int = 3,
        dropoutRate: float = 0.3
    ):
        """
        Args:
            inputFeatures: Number of input features
            hiddenChannels: Number of hidden channels
            numLayers: Number of dilated conv layers
            kernelSize: Kernel size
            dropoutRate: Dropout rate
        """
        super().__init__()
        
        layers = []
        currentChannels = inputFeatures
        
        for i in range(numLayers):
            dilation = 2 ** i
            padding = (kernelSize - 1) * dilation // 2
            
            layers.append(Conv1DBlock(
                inChannels=currentChannels,
                outChannels=hiddenChannels,
                kernelSize=kernelSize,
                padding=padding,
                dilation=dilation,
                dropoutRate=dropoutRate
            ))
            
            currentChannels = hiddenChannels
            
        self.dilatedLayers = nn.Sequential(*layers)
        
        logger.info(f"Dilated CNN initialized: {numLayers} layers, dilation=[1, 2, 4, ...]")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, features, seq_len) or (batch, seq_len, features)
            
        Returns:
            Output tensor (batch, hidden_channels, seq_len) or (batch, seq_len, hidden_channels)
        """
        # Ensure input is (batch, features, seq_len)
        needsTranspose = False
        if x.dim() == 3 and x.size(1) > x.size(2):
            x = x.transpose(1, 2)
            needsTranspose = True
            
        x = self.dilatedLayers(x)
        
        # Transpose back if needed
        if needsTranspose:
            x = x.transpose(1, 2)
            
        return x


class DepthwiseSeparableCNN(nn.Module):
    """Depthwise Separable Convolution for efficient feature extraction"""
    
    def __init__(
        self,
        inputFeatures: int,
        outputChannels: int,
        kernelSize: int = 3,
        padding: int = 1
    ):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            inputFeatures,
            inputFeatures,
            kernel_size=kernelSize,
            padding=padding,
            groups=inputFeatures
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(
            inputFeatures,
            outputChannels,
            kernel_size=1
        )
        
        self.bn = nn.BatchNorm1d(outputChannels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class TemporalConvolutionalNetwork(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling
    Uses dilated causal convolutions
    """
    
    def __init__(
        self,
        inputFeatures: int,
        hiddenChannels: int = 128,
        numLayers: int = 4,
        kernelSize: int = 3,
        dropoutRate: float = 0.3
    ):
        super().__init__()
        
        self.tcnLayers = nn.ModuleList()
        self.residuals = nn.ModuleList()
        currentChannels = inputFeatures
        
        for i in range(numLayers):
            dilation = 2 ** i
            padding = (kernelSize - 1) * dilation
            
            # Causal convolution layer
            layer = nn.Sequential(
                nn.Conv1d(
                    currentChannels,
                    hiddenChannels,
                    kernel_size=kernelSize,
                    padding=padding,
                    dilation=dilation
                ),
                nn.BatchNorm1d(hiddenChannels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropoutRate)
            )
            self.tcnLayers.append(layer)
            
            # Residual connection for this layer
            if currentChannels != hiddenChannels:
                self.residuals.append(nn.Conv1d(currentChannels, hiddenChannels, kernel_size=1))
            else:
                self.residuals.append(None)
                
            currentChannels = hiddenChannels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal convolution
        
        Args:
            x: Input tensor (batch, features, seq_len)
            
        Returns:
            Output tensor (batch, hidden_channels, seq_len)
        """
        for i, (layer, residual) in enumerate(zip(self.tcnLayers, self.residuals)):
            identity = x
            
            # Apply layer
            out = layer(x)
            
            # Crop to original sequence length (remove future padding)
            out = out[:, :, :identity.size(2)]
            
            # Add residual connection
            if residual is not None:
                identity = residual(identity)
            out = out + identity
            
            x = out
        
        return x


class CNNModelFactory:
    """Factory for creating different CNN architectures"""
    
    @staticmethod
    def create(
        modelType: str,
        inputFeatures: int,
        **kwargs
    ) -> nn.Module:
        """
        Create CNN model
        
        Args:
            modelType: Type of CNN ('standard', 'residual', 'multi_scale', 'dilated', 'tcn', 'depthwise')
            inputFeatures: Number of input features
            **kwargs: Additional model parameters
            
        Returns:
            CNN model
        """
        if modelType == 'standard':
            return CNNFeatureExtractor(
                inputFeatures=inputFeatures,
                architecture='standard',
                **kwargs
            )
        elif modelType == 'residual':
            return CNNFeatureExtractor(
                inputFeatures=inputFeatures,
                architecture='residual',
                useResidual=True,
                **kwargs
            )
        elif modelType == 'multi_scale':
            return CNNFeatureExtractor(
                inputFeatures=inputFeatures,
                architecture='multi_scale',
                useMultiScale=True,
                **kwargs
            )
        elif modelType == 'dilated':
            return DilatedCNN(
                inputFeatures=inputFeatures,
                **kwargs
            )
        elif modelType == 'tcn':
            return TemporalConvolutionalNetwork(
                inputFeatures=inputFeatures,
                **kwargs
            )
        elif modelType == 'depthwise':
            outputChannels = kwargs.get('outputChannels', 128)
            # Remove outputChannels from kwargs to avoid duplicate argument
            kwargs.pop('outputChannels', None)
            return DepthwiseSeparableCNN(
                inputFeatures=inputFeatures,
                outputChannels=outputChannels,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown CNN model type: {modelType}")
