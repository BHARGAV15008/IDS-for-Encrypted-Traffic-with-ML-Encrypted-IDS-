"""
LSTM Module for Temporal Feature Extraction
Implements Bidirectional LSTM for capturing temporal dependencies in network traffic
Analyzes sequential patterns and long-range correlations
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class BiLSTMLayer(nn.Module):
    """Bidirectional LSTM Layer with optional attention"""
    
    def __init__(
        self,
        inputSize: int,
        hiddenSize: int,
        numLayers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional
        self.numDirections = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=inputSize,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
            dropout=dropout if numLayers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout for single layer
        self.dropout = nn.Dropout(dropout) if numLayers == 1 else None
        
    def forward(
        self,
        x: torch.Tensor,
        hiddenState: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            hiddenState: Optional initial hidden state
            
        Returns:
            output: (batch, seq_len, hidden_size * num_directions)
            (h_n, c_n): Final hidden and cell states
        """
        output, (h_n, c_n) = self.lstm(x, hiddenState)
        
        if self.dropout is not None:
            output = self.dropout(output)
            
        return output, (h_n, c_n)


class StackedBiLSTM(nn.Module):
    """Stacked Bidirectional LSTM with residual connections"""
    
    def __init__(
        self,
        inputSize: int,
        hiddenSizes: List[int] = [128, 256],
        dropout: float = 0.3,
        useResidual: bool = True
    ):
        """
        Args:
            inputSize: Input feature dimension
            hiddenSizes: List of hidden sizes for each layer
            dropout: Dropout rate
            useResidual: Whether to use residual connections
        """
        super().__init__()
        
        self.inputSize = inputSize
        self.hiddenSizes = hiddenSizes
        self.useResidual = useResidual
        
        self.layers = nn.ModuleList()
        currentSize = inputSize
        
        for i, hiddenSize in enumerate(hiddenSizes):
            self.layers.append(BiLSTMLayer(
                inputSize=currentSize,
                hiddenSize=hiddenSize,
                numLayers=1,
                dropout=dropout,
                bidirectional=True
            ))
            
            # Projection layer for residual connection
            if useResidual and currentSize != hiddenSize * 2:
                self.layers.append(nn.Linear(currentSize, hiddenSize * 2))
            else:
                self.layers.append(None)
                
            currentSize = hiddenSize * 2  # Bidirectional doubles the size
            
        logger.info(f"Stacked BiLSTM initialized: layers={len(hiddenSizes)}, hidden_sizes={hiddenSizes}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacked LSTM layers
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Output tensor (batch, seq_len, hidden_size * 2)
        """
        for i in range(0, len(self.layers), 2):
            lstm_layer = self.layers[i]
            projection = self.layers[i + 1]
            
            # Save input for residual
            residual = x
            
            # Apply LSTM
            x, _ = lstm_layer(x)
            
            # Apply residual connection
            if self.useResidual:
                if projection is not None:
                    residual = projection(residual)
                x = x + residual
                
        return x


class AttentionLSTM(nn.Module):
    """LSTM with self-attention mechanism"""
    
    def __init__(
        self,
        inputSize: int,
        hiddenSize: int,
        numLayers: int = 2,
        dropout: float = 0.3,
        attentionHeads: int = 4
    ):
        """
        Args:
            inputSize: Input feature dimension
            hiddenSize: Hidden state dimension
            numLayers: Number of LSTM layers
            dropout: Dropout rate
            attentionHeads: Number of attention heads
        """
        super().__init__()
        
        self.lstm = BiLSTMLayer(
            inputSize=inputSize,
            hiddenSize=hiddenSize,
            numLayers=numLayers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hiddenSize * 2,
            num_heads=attentionHeads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layerNorm = nn.LayerNorm(hiddenSize * 2)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Attention LSTM initialized: hidden={hiddenSize}, heads={attentionHeads}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Output tensor (batch, seq_len, hidden_size * 2)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection and layer norm
        out = self.layerNorm(lstm_out + self.dropout(attn_out))
        
        return out


class ConvLSTM(nn.Module):
    """Convolutional LSTM for spatial-temporal feature extraction"""
    
    def __init__(
        self,
        inputChannels: int,
        hiddenChannels: int,
        kernelSize: int = 3,
        numLayers: int = 1
    ):
        """
        Args:
            inputChannels: Number of input channels
            hiddenChannels: Number of hidden channels
            kernelSize: Convolution kernel size
            numLayers: Number of ConvLSTM layers
        """
        super().__init__()
        
        self.inputChannels = inputChannels
        self.hiddenChannels = hiddenChannels
        self.kernelSize = kernelSize
        self.numLayers = numLayers
        self.padding = kernelSize // 2
        
        # Gates: input, forget, cell, output
        self.conv = nn.Conv1d(
            in_channels=inputChannels + hiddenChannels,
            out_channels=4 * hiddenChannels,
            kernel_size=kernelSize,
            padding=self.padding
        )
        
    def forward(
        self,
        x: torch.Tensor,
        hiddenState: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, channels)
            hiddenState: Optional (h, c) tuple
            
        Returns:
            output: (batch, seq_len, hidden_channels)
            (h, c): Final hidden and cell states
        """
        batch, seqLen, _ = x.size()
        
        # Initialize hidden state if not provided
        if hiddenState is None:
            h = torch.zeros(batch, self.hiddenChannels, device=x.device)
            c = torch.zeros(batch, self.hiddenChannels, device=x.device)
        else:
            h, c = hiddenState
            
        outputs = []
        
        for t in range(seqLen):
            x_t = x[:, t, :]  # (batch, channels)
            
            # Concatenate input and hidden state
            combined = torch.cat([x_t, h], dim=1)  # (batch, input_channels + hidden_channels)
            combined = combined.unsqueeze(-1)  # (batch, channels, 1)
            
            # Convolutional gates
            gates = self.conv(combined).squeeze(-1)  # (batch, 4 * hidden_channels)
            
            # Split into gates
            i, f, g, o = gates.chunk(4, dim=1)
            
            # Apply activations
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            
            # Update cell and hidden state
            c = f * c + i * g
            h = o * torch.tanh(c)
            
            outputs.append(h)
            
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_channels)
        
        return output, (h, c)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) as alternative to LSTM"""
    
    def __init__(
        self,
        inputSize: int,
        hiddenSize: int,
        numLayers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=inputSize,
            hidden_size=hiddenSize,
            num_layers=numLayers,
            batch_first=True,
            dropout=dropout if numLayers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout) if numLayers == 1 else None
        
    def forward(
        self,
        x: torch.Tensor,
        hiddenState: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            hiddenState: Optional initial hidden state
            
        Returns:
            output: (batch, seq_len, hidden_size * num_directions)
            h_n: Final hidden state
        """
        output, h_n = self.gru(x, hiddenState)
        
        if self.dropout is not None:
            output = self.dropout(output)
            
        return output, h_n


class LSTMFeatureExtractor(nn.Module):
    """
    Advanced LSTM-based feature extractor for network traffic
    Supports multiple architectures and configurations
    """
    
    def __init__(
        self,
        inputSize: int,
        hiddenSize: int = 128,
        numLayers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        architecture: str = 'standard',
        useAttention: bool = False,
        outputDim: Optional[int] = None
    ):
        """
        Args:
            inputSize: Input feature dimension
            hiddenSize: Hidden state dimension
            numLayers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            architecture: Architecture type ('standard', 'stacked', 'attention', 'gru')
            useAttention: Whether to add attention mechanism
            outputDim: Output dimension (if None, use hidden_size * num_directions)
        """
        super().__init__()
        
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional
        self.architecture = architecture
        self.numDirections = 2 if bidirectional else 1
        self.outputDim = outputDim if outputDim else hiddenSize * self.numDirections
        
        # Build architecture
        if architecture == 'standard':
            self.rnn = BiLSTMLayer(
                inputSize=inputSize,
                hiddenSize=hiddenSize,
                numLayers=numLayers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif architecture == 'stacked':
            hiddenSizes = [hiddenSize * (i + 1) for i in range(numLayers)]
            self.rnn = StackedBiLSTM(
                inputSize=inputSize,
                hiddenSizes=hiddenSizes,
                dropout=dropout,
                useResidual=True
            )
        elif architecture == 'attention':
            self.rnn = AttentionLSTM(
                inputSize=inputSize,
                hiddenSize=hiddenSize,
                numLayers=numLayers,
                dropout=dropout,
                attentionHeads=4
            )
        elif architecture == 'gru':
            self.rnn = GRULayer(
                inputSize=inputSize,
                hiddenSize=hiddenSize,
                numLayers=numLayers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
            
        # Optional attention layer
        if useAttention and architecture != 'attention':
            self.attentionLayer = nn.MultiheadAttention(
                embed_dim=hiddenSize * self.numDirections,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.attentionLayer = None
            
        # Output projection
        rnnOutputSize = hiddenSize * self.numDirections
        if architecture == 'stacked':
            rnnOutputSize = hiddenSize * numLayers * self.numDirections
            
        if self.outputDim != rnnOutputSize:
            self.outputProjection = nn.Linear(rnnOutputSize, self.outputDim)
        else:
            self.outputProjection = None
            
        logger.info(f"LSTM Feature Extractor initialized: {architecture} architecture, "
                   f"hidden={hiddenSize}, layers={numLayers}, bidirectional={bidirectional}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Output tensor (batch, seq_len, output_dim)
        """
        # Apply RNN
        if isinstance(self.rnn, (BiLSTMLayer, GRULayer)):
            output, _ = self.rnn(x)
        else:
            output = self.rnn(x)
            
        # Apply attention if specified
        if self.attentionLayer is not None:
            attn_out, _ = self.attentionLayer(output, output, output)
            output = output + attn_out
            
        # Apply output projection if specified
        if self.outputProjection is not None:
            output = self.outputProjection(output)
            
        return output


class LSTMEncoder(nn.Module):
    """LSTM Encoder that outputs a fixed-size representation"""
    
    def __init__(
        self,
        inputSize: int,
        hiddenSize: int = 128,
        numLayers: int = 2,
        dropout: float = 0.3,
        pooling: str = 'last'
    ):
        """
        Args:
            inputSize: Input feature dimension
            hiddenSize: Hidden state dimension
            numLayers: Number of LSTM layers
            dropout: Dropout rate
            pooling: Pooling method ('last', 'mean', 'max', 'attention')
        """
        super().__init__()
        
        self.pooling = pooling
        
        self.lstm = BiLSTMLayer(
            inputSize=inputSize,
            hiddenSize=hiddenSize,
            numLayers=numLayers,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention pooling
        if pooling == 'attention':
            self.attentionWeights = nn.Linear(hiddenSize * 2, 1)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Encoded representation (batch, hidden_size * 2)
        """
        output, (h_n, c_n) = self.lstm(x)
        
        if self.pooling == 'last':
            # Use last hidden state
            # h_n shape: (num_layers * num_directions, batch, hidden_size)
            # Take last layer, concatenate both directions
            encoded = torch.cat([h_n[-2], h_n[-1]], dim=1)
            
        elif self.pooling == 'mean':
            # Mean pooling over sequence
            encoded = output.mean(dim=1)
            
        elif self.pooling == 'max':
            # Max pooling over sequence
            encoded, _ = output.max(dim=1)
            
        elif self.pooling == 'attention':
            # Attention-based pooling
            scores = self.attentionWeights(output)  # (batch, seq_len, 1)
            weights = torch.softmax(scores, dim=1)
            encoded = (output * weights).sum(dim=1)
            
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
            
        return encoded


class LSTMModelFactory:
    """Factory for creating different LSTM architectures"""
    
    @staticmethod
    def create(
        modelType: str,
        inputSize: int,
        **kwargs
    ) -> nn.Module:
        """
        Create LSTM model
        
        Args:
            modelType: Type of LSTM ('standard', 'stacked', 'attention', 'gru', 'encoder')
            inputSize: Input feature dimension
            **kwargs: Additional model parameters
            
        Returns:
            LSTM model
        """
        if modelType == 'standard':
            return LSTMFeatureExtractor(
                inputSize=inputSize,
                architecture='standard',
                **kwargs
            )
        elif modelType == 'stacked':
            return LSTMFeatureExtractor(
                inputSize=inputSize,
                architecture='stacked',
                **kwargs
            )
        elif modelType == 'attention':
            return LSTMFeatureExtractor(
                inputSize=inputSize,
                architecture='attention',
                useAttention=True,
                **kwargs
            )
        elif modelType == 'gru':
            return LSTMFeatureExtractor(
                inputSize=inputSize,
                architecture='gru',
                **kwargs
            )
        elif modelType == 'encoder':
            return LSTMEncoder(
                inputSize=inputSize,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown LSTM model type: {modelType}")
