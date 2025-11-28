"""
Custom Loss Functions for Imbalanced Datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced datasets.
    
    This loss function is an improvement over CrossEntropyLoss for classification tasks
    with a large class imbalance. It dynamically scales the cross-entropy loss,
    down-weighting the loss assigned to well-classified examples. This allows the
    model to focus on hard-to-classify examples.
    
    Attributes:
        alpha (torch.Tensor or float): Weighting factor for each class. Can be a float
            for binary classification or a tensor for multi-class.
        gamma (float): Focusing parameter. Higher values give more weight to hard examples.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha (Optional[torch.Tensor]): A manual rescaling weight given to each class.
                If given, has to be a Tensor of size C.
            gamma (float): Focusing parameter. Defaults to 2.0.
            reduction (str): 'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Focal Loss.
        
        Args:
            inputs (torch.Tensor): Model's raw output (logits) of shape (N, C).
            targets (torch.Tensor): Ground truth labels of shape (N).
            
        Returns:
            torch.Tensor: The calculated focal loss.
        """
        # Calculate Cross-Entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities of the correct class
        pt = torch.exp(-ce_loss)
        
        # Calculate the focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply class weighting (alpha)
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            
            # Gather the alpha values for each sample
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = alpha_t * focal_loss
            
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
