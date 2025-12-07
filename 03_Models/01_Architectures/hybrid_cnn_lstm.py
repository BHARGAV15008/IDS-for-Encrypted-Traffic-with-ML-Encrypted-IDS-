"""
Hybrid CNN-LSTM Architecture for P22 IDS

This module implements the core hybrid detection framework combining
CNN spatial feature extraction with LSTM temporal modeling for
encrypted traffic classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional, Dict, List
import numpy as np
import logging


class SpatialFeatureExtractor(nn.Module):
    """CNN-based spatial feature extractor for packet-level analysis."""
    
    def __init__(
        self,
        input_channels: int = 1,
        conv_layers: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 5, 7],
        dropout_rate: float = 0.3
    ):
        """
        Initialize spatial feature extractor.
        
        Args:
            input_channels: Number of input channels
            conv_layers: Number of filters in each conv layer
            kernel_sizes: Kernel sizes for multi-scale feature extraction
            dropout_rate: Dropout probability
        """
        super(SpatialFeatureExtractor, self).__init__()
        
        self.conv_layers = conv_layers
        self.kernel_sizes = kernel_sizes
        
        # Multi-scale convolutional branches
        self.conv_branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            branch = nn.ModuleList()
            in_channels = input_channels
            
            for out_channels in conv_layers:
                branch.append(nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout_rate)
                ))
                in_channels = out_channels
            
            self.conv_branches.append(branch)
        
        # Feature fusion layer
        # Each branch produces conv_layers[-1] features (final output channels)
        total_features = conv_layers[-1] * len(kernel_sizes)
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            Extracted spatial features
        """
        branch_outputs = []
        
        # Process through each multi-scale branch
        for branch in self.conv_branches:
            branch_x = x
            for conv_layer in branch:
                branch_x = conv_layer(branch_x)
            
            # Global average pooling
            branch_features = F.adaptive_avg_pool1d(branch_x, 1).squeeze(-1)
            branch_outputs.append(branch_features)
        
        # Concatenate features from all branches
        combined_features = torch.cat(branch_outputs, dim=1)
        
        # Fusion layer
        fused_features = self.fusion_layer(combined_features)
        
        return fused_features


class TemporalSequenceModeler(nn.Module):
    """LSTM-based temporal sequence modeler for flow-level analysis."""
    
    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
        attention: bool = True
    ):
        """
        Initialize temporal sequence modeler.
        
        Args:
            input_size: Size of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout_rate: Dropout probability
            attention: Whether to use attention mechanism
        """
        super(TemporalSequenceModeler, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        if attention:
            self.attention = AttentionLayer(lstm_output_size)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal sequence modeler.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Sequence lengths for packing (optional)
            
        Returns:
            Temporal features
        """
        batch_size, seq_len, _ = x.size()
        
        # Pack sequences if lengths provided
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, 
                                   enforce_sorted=False)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Apply attention if enabled
        if self.use_attention:
            attended_output = self.attention(lstm_out, lengths)
        else:
            # Use final hidden state
            if self.bidirectional:
                # Concatenate forward and backward final hidden states
                attended_output = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                attended_output = hidden[-1]
        
        # Output projection
        temporal_features = self.output_projection(attended_output)
        
        return temporal_features


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence modeling."""
    
    def __init__(self, hidden_size: int):
        """
        Initialize attention layer.
        
        Args:
            hidden_size: Size of hidden states
        """
        super(AttentionLayer, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Attention parameters
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_outputs: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention mechanism.
        
        Args:
            lstm_outputs: LSTM outputs (batch_size, seq_len, hidden_size)
            lengths: Sequence lengths for masking
            
        Returns:
            Attended output
        """
        batch_size, seq_len, hidden_size = lstm_outputs.size()
        
        # Calculate attention scores
        attention_scores = self.attention_weights(lstm_outputs).squeeze(-1)
        
        # Apply length masking if provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=lstm_outputs.device).expand(
                batch_size, seq_len
            ) >= lengths.unsqueeze(1)
            attention_scores.masked_fill_(mask, float('-inf'))
        
        # Softmax normalization
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        attended_output = torch.sum(
            lstm_outputs * attention_weights.unsqueeze(-1), dim=1
        )
        
        return attended_output


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for encrypted traffic classification.
    
    This model combines:
    1. CNN spatial feature extraction from packet-level data
    2. LSTM temporal modeling of flow sequences
    3. Attention mechanisms for focus on important patterns
    4. Multi-scale feature fusion for robust detection
    """
    
    def __init__(
        self,
        packet_input_size: int = 1500,
        num_classes: int = 10,
        cnn_channels: List[int] = [64, 128, 256],
        cnn_kernels: List[int] = [3, 5, 7],
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize hybrid CNN-LSTM model.
        
        Args:
            packet_input_size: Size of packet-level input
            num_classes: Number of output classes
            cnn_channels: CNN channel dimensions
            cnn_kernels: CNN kernel sizes
            lstm_hidden_size: LSTM hidden state size
            lstm_layers: Number of LSTM layers
            dropout_rate: Dropout probability
            use_attention: Whether to use attention mechanism
            use_residual: Whether to use residual connections
        """
        super(HybridCNNLSTM, self).__init__()
        
        self.packet_input_size = packet_input_size
        self.num_classes = num_classes
        self.use_residual = use_residual
        
        # Spatial feature extractor (CNN)
        self.spatial_extractor = SpatialFeatureExtractor(
            input_channels=1,
            conv_layers=cnn_channels,
            kernel_sizes=cnn_kernels,
            dropout_rate=dropout_rate
        )
        
        # Temporal sequence modeler (LSTM)
        spatial_feature_size = 256  # Output size from spatial extractor
        self.temporal_modeler = TemporalSequenceModeler(
            input_size=spatial_feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout_rate=dropout_rate,
            attention=use_attention
        )
        
        # Feature fusion and classification
        temporal_feature_size = 128  # Output size from temporal modeler
        
        if use_residual:
            # Residual connection from spatial features
            self.residual_projection = nn.Linear(spatial_feature_size, temporal_feature_size)
            fusion_input_size = temporal_feature_size * 2
        else:
            fusion_input_size = temporal_feature_size
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, packet_size)
            lengths: Sequence lengths for variable-length sequences
            
        Returns:
            Classification logits
        """
        batch_size, seq_len, packet_size = x.size()
        
        # Reshape for CNN processing: (batch_size * seq_len, 1, packet_size)
        x_reshaped = x.view(batch_size * seq_len, 1, packet_size)
        
        # Extract spatial features from each packet
        spatial_features = self.spatial_extractor(x_reshaped)
        
        # Reshape back to sequence format: (batch_size, seq_len, feature_size)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # Extract temporal features from sequence
        temporal_features = self.temporal_modeler(spatial_features, lengths)
        
        # Feature fusion
        if self.use_residual:
            # Global pooling of spatial features for residual connection
            pooled_spatial = torch.mean(spatial_features, dim=1)
            residual_features = self.residual_projection(pooled_spatial)
            
            # Concatenate temporal and residual features
            fused_features = torch.cat([temporal_features, residual_features], dim=1)
        else:
            fused_features = temporal_features
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps at different stages
        """
        batch_size, seq_len, packet_size = x.size()
        feature_maps = {}
        
        # Reshape for CNN processing
        x_reshaped = x.view(batch_size * seq_len, 1, packet_size)
        
        # Extract spatial features
        spatial_features = self.spatial_extractor(x_reshaped)
        feature_maps['spatial_features'] = spatial_features.view(batch_size, seq_len, -1)
        
        # Extract temporal features
        temporal_features = self.temporal_modeler(spatial_features.view(batch_size, seq_len, -1))
        feature_maps['temporal_features'] = temporal_features
        
        return feature_maps


class EnsembleHybridModel(nn.Module):
    """Ensemble of hybrid CNN-LSTM models for improved robustness."""
    
    def __init__(
        self,
        num_models: int = 3,
        model_configs: Optional[List[Dict]] = None,
        **base_config
    ):
        """
        Initialize ensemble model.
        
        Args:
            num_models: Number of models in ensemble
            model_configs: List of configuration dictionaries for each model
            **base_config: Base configuration for all models
        """
        super(EnsembleHybridModel, self).__init__()
        
        self.num_models = num_models
        
        # Create ensemble models
        self.models = nn.ModuleList()
        
        if model_configs is None:
            # Create diverse models with different architectures
            model_configs = self._generate_diverse_configs(num_models, base_config)
        
        for config in model_configs:
            model = HybridCNNLSTM(**config)
            self.models.append(model)
        
        # Ensemble fusion
        num_classes = base_config.get('num_classes', 10)
        self.ensemble_fusion = nn.Sequential(
            nn.Linear(num_classes * num_models, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            lengths: Sequence lengths
            
        Returns:
            Ensemble predictions
        """
        # Get predictions from each model
        model_outputs = []
        for model in self.models:
            output = model(x, lengths)
            model_outputs.append(output)
        
        # Concatenate all outputs
        ensemble_input = torch.cat(model_outputs, dim=1)
        
        # Fusion layer
        ensemble_output = self.ensemble_fusion(ensemble_input)
        
        return ensemble_output
    
    def _generate_diverse_configs(self, num_models: int, base_config: Dict) -> List[Dict]:
        """Generate diverse model configurations."""
        configs = []
        
        # Different CNN architectures
        cnn_variants = [
            [32, 64, 128],
            [64, 128, 256],
            [128, 256, 512]
        ]
        
        # Different kernel combinations
        kernel_variants = [
            [3, 5],
            [3, 5, 7],
            [5, 7, 9]
        ]
        
        # Different LSTM configurations
        lstm_variants = [
            {'lstm_hidden_size': 64, 'lstm_layers': 1},
            {'lstm_hidden_size': 128, 'lstm_layers': 2},
            {'lstm_hidden_size': 256, 'lstm_layers': 3}
        ]
        
        for i in range(num_models):
            config = base_config.copy()
            config['cnn_channels'] = cnn_variants[i % len(cnn_variants)]
            config['cnn_kernels'] = kernel_variants[i % len(kernel_variants)]
            config.update(lstm_variants[i % len(lstm_variants)])
            configs.append(config)
        
        return configs


def create_hybrid_model(model_type: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to create hybrid models.
    
    Args:
        model_type: Type of hybrid model to create
        **kwargs: Model configuration parameters
        
    Returns:
        Initialized hybrid model
    """
    if model_type == "standard":
        return HybridCNNLSTM(**kwargs)
    elif model_type == "ensemble":
        return EnsembleHybridModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test hybrid model
    model = HybridCNNLSTM(
        packet_input_size=1500,
        num_classes=10,
        cnn_channels=[64, 128, 256],
        lstm_hidden_size=128,
        use_attention=True
    )
    
    # Test input
    batch_size = 4
    seq_len = 20
    packet_size = 1500
    
    x = torch.randn(batch_size, seq_len, packet_size)
    lengths = torch.randint(10, seq_len + 1, (batch_size,))
    
    # Forward pass
    output = model(x, lengths)
    print(f"Model output shape: {output.shape}")
    
    # Test ensemble model
    ensemble_model = EnsembleHybridModel(
        num_models=3,
        packet_input_size=1500,
        num_classes=10
    )
    
    ensemble_output = ensemble_model(x, lengths)
    print(f"Ensemble output shape: {ensemble_output.shape}")
    
    # Get feature maps
    feature_maps = model.get_feature_maps(x)
    for name, features in feature_maps.items():
        print(f"{name} shape: {features.shape}")
