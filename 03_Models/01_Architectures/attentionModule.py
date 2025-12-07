"""
Attention Module for Feature Weighting and Discovery
Implements various attention mechanisms for automated feature importance learning
Supports self-attention, multi-head attention, and custom attention layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism"""
    
    def __init__(self, temperature: float, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, n_heads, seq_len, d_k)
            key: (batch, n_heads, seq_len, d_k)
            value: (batch, n_heads, seq_len, d_v)
            mask: Optional mask tensor
            
        Returns:
            output: (batch, n_heads, seq_len, d_v)
            attention: (batch, n_heads, seq_len, seq_len)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, value)
        
        return output, attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(
        self,
        embedDim: int,
        numHeads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Args:
            embedDim: Embedding dimension
            numHeads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        assert embedDim % numHeads == 0, "embedDim must be divisible by numHeads"
        
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim // numHeads
        
        # Linear projections
        self.qLinear = nn.Linear(embedDim, embedDim, bias=bias)
        self.kLinear = nn.Linear(embedDim, embedDim, bias=bias)
        self.vLinear = nn.Linear(embedDim, embedDim, bias=bias)
        self.outLinear = nn.Linear(embedDim, embedDim, bias=bias)
        
        # Attention
        self.attention = ScaledDotProductAttention(
            temperature=math.sqrt(self.headDim),
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(embedDim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, embed_dim)
            key: (batch, seq_len, embed_dim)
            value: (batch, seq_len, embed_dim)
            mask: Optional mask
            
        Returns:
            output: (batch, seq_len, embed_dim)
            attention: (batch, num_heads, seq_len, seq_len)
        """
        batch, seqLen, _ = query.size()
        residual = query
        
        # Linear projections and reshape
        q = self.qLinear(query).view(batch, seqLen, self.numHeads, self.headDim).transpose(1, 2)
        k = self.kLinear(key).view(batch, seqLen, self.numHeads, self.headDim).transpose(1, 2)
        v = self.vLinear(value).view(batch, seqLen, self.numHeads, self.headDim).transpose(1, 2)
        
        # Apply attention
        output, attention = self.attention(q, k, v, mask)
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch, seqLen, self.embedDim)
        output = self.outLinear(output)
        output = self.dropout(output)
        
        # Add residual and normalize
        output = self.layerNorm(output + residual)
        
        return output, attention


class SelfAttention(nn.Module):
    """Self-Attention layer for sequence modeling"""
    
    def __init__(
        self,
        inputDim: int,
        attentionDim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            inputDim: Input feature dimension
            attentionDim: Attention dimension (if None, use inputDim)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.inputDim = inputDim
        self.attentionDim = attentionDim if attentionDim else inputDim
        
        # Attention weights
        self.queryLayer = nn.Linear(inputDim, self.attentionDim)
        self.keyLayer = nn.Linear(inputDim, self.attentionDim)
        self.valueLayer = nn.Linear(inputDim, self.attentionDim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.attentionDim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional mask
            
        Returns:
            output: (batch, seq_len, attention_dim)
            weights: (batch, seq_len, seq_len)
        """
        # Compute Q, K, V
        query = self.queryLayer(x)
        key = self.keyLayer(x)
        value = self.valueLayer(x)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention to values
        output = torch.matmul(weights, value)
        
        return output, weights


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) Attention mechanism"""
    
    def __init__(self, hiddenDim: int, attentionDim: int = 128):
        """
        Args:
            hiddenDim: Hidden state dimension
            attentionDim: Attention dimension
        """
        super().__init__()
        
        self.hiddenDim = hiddenDim
        self.attentionDim = attentionDim
        
        self.queryLayer = nn.Linear(hiddenDim, attentionDim, bias=False)
        self.keyLayer = nn.Linear(hiddenDim, attentionDim, bias=False)
        self.energyLayer = nn.Linear(attentionDim, 1, bias=False)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor (batch, hidden_dim)
            keys: Key tensor (batch, seq_len, hidden_dim)
            
        Returns:
            context: (batch, hidden_dim)
            weights: (batch, seq_len)
        """
        # Expand query to match keys
        query = query.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Compute energy
        queryProj = self.queryLayer(query)  # (batch, 1, attention_dim)
        keysProj = self.keyLayer(keys)  # (batch, seq_len, attention_dim)
        
        energy = torch.tanh(queryProj + keysProj)  # (batch, seq_len, attention_dim)
        scores = self.energyLayer(energy).squeeze(-1)  # (batch, seq_len)
        
        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Compute context vector
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (batch, hidden_dim)
        
        return context, weights


class FeatureAttention(nn.Module):
    """
    Feature-level attention for automatic feature importance learning
    Assigns importance weights to different features
    """
    
    def __init__(self, numFeatures: int, attentionDim: int = 64):
        """
        Args:
            numFeatures: Number of input features
            attentionDim: Attention dimension
        """
        super().__init__()
        
        self.numFeatures = numFeatures
        self.attentionDim = attentionDim
        
        # Feature attention network
        self.attentionNet = nn.Sequential(
            nn.Linear(numFeatures, attentionDim),
            nn.Tanh(),
            nn.Linear(attentionDim, numFeatures),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, num_features) or (batch, num_features)
            
        Returns:
            weighted: Weighted features
            weights: Feature importance weights
        """
        # Compute feature attention weights
        if x.dim() == 3:
            # For sequences, compute attention per timestep
            weights = self.attentionNet(x)  # (batch, seq_len, num_features)
        else:
            # For single vectors
            weights = self.attentionNet(x)  # (batch, num_features)
            
        # Apply weights to features
        weighted = x * weights
        
        return weighted, weights


class TemporalAttention(nn.Module):
    """
    Temporal attention for sequence data
    Assigns importance weights to different time steps
    """
    
    def __init__(self, hiddenDim: int):
        """
        Args:
            hiddenDim: Hidden dimension of sequence
        """
        super().__init__()
        
        self.hiddenDim = hiddenDim
        
        # Temporal attention network
        self.attentionNet = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim),
            nn.Tanh(),
            nn.Linear(hiddenDim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            
        Returns:
            context: Context vector (batch, hidden_dim)
            weights: Temporal attention weights (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attentionNet(x).squeeze(-1)  # (batch, seq_len)
        
        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Compute weighted context
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        
        return context, weights


class DualAttention(nn.Module):
    """
    Dual attention mechanism combining feature and temporal attention
    """
    
    def __init__(self, numFeatures: int, hiddenDim: int):
        """
        Args:
            numFeatures: Number of input features
            hiddenDim: Hidden dimension
        """
        super().__init__()
        
        self.featureAttention = FeatureAttention(numFeatures)
        self.temporalAttention = TemporalAttention(hiddenDim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, num_features)
            
        Returns:
            context: Context vector (batch, hidden_dim)
            (feature_weights, temporal_weights): Attention weights
        """
        # Apply feature attention
        featureWeighted, featureWeights = self.featureAttention(x)
        
        # Apply temporal attention
        context, temporalWeights = self.temporalAttention(featureWeighted)
        
        return context, (featureWeights, temporalWeights)


class CrossAttention(nn.Module):
    """Cross-attention for attending to different modalities or sources"""
    
    def __init__(
        self,
        queryDim: int,
        keyDim: int,
        valueDim: int,
        outputDim: int,
        numHeads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            queryDim: Query dimension
            keyDim: Key dimension
            valueDim: Value dimension
            outputDim: Output dimension
            numHeads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.numHeads = numHeads
        self.headDim = outputDim // numHeads
        
        self.qProj = nn.Linear(queryDim, outputDim)
        self.kProj = nn.Linear(keyDim, outputDim)
        self.vProj = nn.Linear(valueDim, outputDim)
        self.outProj = nn.Linear(outputDim, outputDim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.headDim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, query_len, query_dim)
            key: (batch, key_len, key_dim)
            value: (batch, key_len, value_dim)
            
        Returns:
            output: (batch, query_len, output_dim)
            attention: (batch, num_heads, query_len, key_len)
        """
        batch = query.size(0)
        
        # Project and reshape
        q = self.qProj(query).view(batch, -1, self.numHeads, self.headDim).transpose(1, 2)
        k = self.kProj(key).view(batch, -1, self.numHeads, self.headDim).transpose(1, 2)
        v = self.vProj(value).view(batch, -1, self.numHeads, self.headDim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply to values
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch, -1, self.numHeads * self.headDim)
        output = self.outProj(output)
        
        return output, attention


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence aggregation"""
    
    def __init__(self, inputDim: int, attentionDim: int = 128):
        """
        Args:
            inputDim: Input feature dimension
            attentionDim: Attention dimension
        """
        super().__init__()
        
        self.attentionNet = nn.Sequential(
            nn.Linear(inputDim, attentionDim),
            nn.Tanh(),
            nn.Linear(attentionDim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            pooled: Pooled representation (batch, input_dim)
            weights: Attention weights (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attentionNet(x).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, input_dim)
        
        return pooled, weights


class AttentionBlock(nn.Module):
    """
    Complete attention block with normalization and feed-forward
    Similar to Transformer encoder layer
    """
    
    def __init__(
        self,
        embedDim: int,
        numHeads: int = 8,
        ffnDim: int = 2048,
        dropout: float = 0.1
    ):
        """
        Args:
            embedDim: Embedding dimension
            numHeads: Number of attention heads
            ffnDim: Feed-forward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            embedDim=embedDim,
            numHeads=numHeads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedDim, ffnDim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffnDim, embedDim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embedDim)
        self.norm2 = nn.LayerNorm(embedDim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        # Self-attention with residual
        attnOut, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attnOut)
        
        # Feed-forward with residual
        ffnOut = self.ffn(x)
        x = self.norm2(x + ffnOut)
        
        return x


class AttentionModelFactory:
    """Factory for creating different attention mechanisms"""
    
    @staticmethod
    def create(
        attentionType: str,
        **kwargs
    ) -> nn.Module:
        """
        Create attention module
        
        Args:
            attentionType: Type of attention ('self', 'multi_head', 'additive', 'feature', 
                          'temporal', 'dual', 'cross', 'pooling', 'block')
            **kwargs: Additional parameters
            
        Returns:
            Attention module
        """
        if attentionType == 'self':
            return SelfAttention(**kwargs)
        elif attentionType == 'multi_head':
            return MultiHeadAttention(**kwargs)
        elif attentionType == 'additive':
            return AdditiveAttention(**kwargs)
        elif attentionType == 'feature':
            return FeatureAttention(**kwargs)
        elif attentionType == 'temporal':
            return TemporalAttention(**kwargs)
        elif attentionType == 'dual':
            return DualAttention(**kwargs)
        elif attentionType == 'cross':
            return CrossAttention(**kwargs)
        elif attentionType == 'pooling':
            return AttentionPooling(**kwargs)
        elif attentionType == 'block':
            return AttentionBlock(**kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attentionType}")
