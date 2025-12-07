from typing import Tuple
import torch
import torch.nn as nn


class AttentionBlock(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # x: (batch, seq, feat)
        scores = self.attn(x)  # (batch, seq, 1)
        weights = torch.softmax(scores.squeeze(-1), dim=-1)  # (batch, seq)
        context = torch.einsum("bsd,bs->bd", x, weights)  # (batch, feat)
        return context, weights


class HybridCNNBiLSTMAttention(nn.Module):

    def __init__(self, input_features: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.attn = AttentionBlock(input_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch, seq, feat)
        x = x.transpose(1, 2)  # (batch, feat, seq)
        x = self.conv(x)       # (batch, 128, seq)
        x = x.transpose(1, 2)  # (batch, seq, 128)
        lstm_out, _ = self.bilstm(x)  # (batch, seq, 128)
        context, _ = self.attn(lstm_out)  # (batch, 128)
        logits = self.classifier(context)
        return logits


