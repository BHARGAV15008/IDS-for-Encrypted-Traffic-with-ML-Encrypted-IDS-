"""
Improved Hybrid CNN-BiLSTM-Attention Model with Multi-Head Attention

Enhancements:
- Multi-head attention mechanism
- Layer normalization and residual connections
- Improved gradient flow
- Better feature extraction
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with layer normalization"""
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = np.sqrt(self.head_dim)
        
        # Linear projections
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.fc_out = nn.Linear(input_dim, input_dim)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) optional mask for padding
            
        Returns:
            output: (batch, seq_len, input_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Linear transformations
        Q = self.query(x)  # (batch, seq_len, input_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (batch, num_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.input_dim)
        # (batch, seq_len, input_dim)
        
        # Final linear projection
        output = self.fc_out(context)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + x)
        
        return output, attention_weights


class ImprovedHybridCNNBiLSTMAttention(nn.Module):
    """
    Improved Hybrid CNN-BiLSTM-Attention model with:
    - Multi-head attention
    - Layer normalization
    - Residual connections
    - Dropout for regularization
    - Better feature extraction
    """
    
    def __init__(
        self,
        input_features: int,
        num_classes: int,
        num_attention_heads: int = 4,
        lstm_hidden_size: int = 64,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # CNN layers for spatial feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_features,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # BiLSTM for temporal feature extraction
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_hidden_size > 1 else 0
        )
        
        # Multi-head attention
        lstm_output_size = lstm_hidden_size * 2  # bidirectional
        self.attention = MultiHeadAttention(
            input_dim=lstm_output_size,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # Classification head
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_features)
            
        Returns:
            logits: (batch, num_classes)
        """
        # CNN feature extraction
        # x: (batch, seq_len, input_features)
        x = x.transpose(1, 2)  # (batch, input_features, seq_len)
        
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Transpose back for LSTM
        x = x.transpose(1, 2)  # (batch, seq_len, 128)
        
        # BiLSTM
        lstm_out, (h_n, c_n) = self.bilstm(x)
        # lstm_out: (batch, seq_len, lstm_hidden_size * 2)
        
        # Multi-head attention
        attn_out, attn_weights = self.attention(lstm_out)
        # attn_out: (batch, seq_len, lstm_hidden_size * 2)
        
        # Global average pooling
        context = torch.mean(attn_out, dim=1)  # (batch, lstm_hidden_size * 2)
        
        # Classification head
        x = self.fc1(context)
        x = self.bn_fc1(x)
        x = self.relu_fc(x)
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout_fc2(x)
        
        # Output logits
        logits = self.classifier(x)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability"""
        # CNN feature extraction
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = x.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        
        # Attention
        _, attn_weights = self.attention(lstm_out)
        
        return attn_weights


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for handling class imbalance
    
    Combines focal loss with class weighting to:
    1. Focus on hard examples (focal loss)
    2. Handle class imbalance (class weights)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, num_classes) logits
            targets: (batch,) class indices
            
        Returns:
            loss: scalar
        """
        # Cross entropy loss
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            reduction='none'
        )
        
        # Focal loss: (1 - p_t)^gamma * ce_loss
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_improved_model(
    input_features: int,
    num_classes: int,
    num_attention_heads: int = 4,
    lstm_hidden_size: int = 64,
    dropout_rate: float = 0.2,
    device: str = 'cpu'
) -> ImprovedHybridCNNBiLSTMAttention:
    """
    Factory function to create improved model
    
    Args:
        input_features: Number of input features
        num_classes: Number of output classes
        num_attention_heads: Number of attention heads
        lstm_hidden_size: LSTM hidden size
        dropout_rate: Dropout rate
        device: Device to place model on
        
    Returns:
        model: ImprovedHybridCNNBiLSTMAttention instance
    """
    model = ImprovedHybridCNNBiLSTMAttention(
        input_features=input_features,
        num_classes=num_classes,
        num_attention_heads=num_attention_heads,
        lstm_hidden_size=lstm_hidden_size,
        dropout_rate=dropout_rate
    )
    
    model = model.to(device)
    
    return model
