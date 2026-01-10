"""
Loss Functions

Custom loss functions for skin lesion classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Focuses training on hard examples by down-weighting easy examples.
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N,) or (N, 1)
            targets: Binary targets of shape (N,)
            
        Returns:
            Focal loss
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute probabilities
        p = torch.sigmoid(inputs)
        
        # Compute cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute focal weight
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for handling extreme class imbalance.
    
    Different gamma values for positive and negative samples.
    Useful when positive class is much rarer.
    
    Args:
        gamma_pos: Focusing parameter for positives (default: 0)
        gamma_neg: Focusing parameter for negatives (default: 4)
        clip: Probability margin for clipping (default: 0.05)
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N,) or (N, 1)
            targets: Binary targets of shape (N,)
            
        Returns:
            Asymmetric loss
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Probabilities
        p = torch.sigmoid(inputs)
        
        # Positive samples
        pos_mask = (targets == 1)
        p_pos = p[pos_mask]
        
        # Asymmetric clipping for negatives
        p_neg = p[~pos_mask]
        p_neg_clipped = (p_neg + self.clip).clamp(max=1)
        
        # Focal weights
        if len(p_pos) > 0:
            pos_weight = (1 - p_pos) ** self.gamma_pos
            pos_loss = -pos_weight * torch.log(p_pos.clamp(min=1e-8))
        else:
            pos_loss = torch.tensor(0.0, device=inputs.device)
        
        if len(p_neg_clipped) > 0:
            neg_weight = p_neg_clipped ** self.gamma_neg
            neg_loss = -neg_weight * torch.log((1 - p_neg).clamp(min=1e-8))
        else:
            neg_loss = torch.tensor(0.0, device=inputs.device)
        
        # Combine
        if self.reduction == 'mean':
            total_loss = (pos_loss.sum() + neg_loss.sum()) / len(targets)
        elif self.reduction == 'sum':
            total_loss = pos_loss.sum() + neg_loss.sum()
        else:
            # Reconstruct full loss tensor
            loss = torch.zeros_like(inputs)
            loss[pos_mask] = pos_loss
            loss[~pos_mask] = neg_loss
            total_loss = loss
        
        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined loss with BCE and Focal Loss.
    
    Allows balancing between standard BCE and focal loss.
    """
    
    def __init__(
        self,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float = None,
    ):
        super().__init__()
        
        self.focal_weight = focal_weight
        
        # BCE loss
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce = nn.BCEWithLogitsLoss()
        
        # Focal loss
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs.view(-1), targets.view(-1))
        focal_loss = self.focal(inputs, targets)
        
        return (1 - self.focal_weight) * bce_loss + self.focal_weight * focal_loss


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross-Entropy with Label Smoothing.
    
    Prevents overconfident predictions by smoothing labels:
        y_smooth = (1 - ε) * y + ε / 2
    
    This helps with calibration and reduces overfitting.
    
    Args:
        smoothing: Label smoothing factor (default: 0.1)
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + self.smoothing / 2
        
        return F.binary_cross_entropy_with_logits(inputs, targets_smooth)


class AUCMaxLoss(nn.Module):
    """
    AUC Maximization Loss.
    
    Directly optimizes the AUC by approximating the pairwise ranking loss.
    For each (pos, neg) pair: loss = max(0, margin - (s_pos - s_neg))
    
    This is more aligned with the pAUC evaluation metric.
    
    Args:
        margin: Margin for pairwise ranking (default: 1.0)
        gamma: Soft margin temperature (default: 2.0)
    """
    
    def __init__(self, margin: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Separate positive and negative samples
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        pos_scores = inputs[pos_mask]
        neg_scores = inputs[neg_mask]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        # Compute pairwise differences: pos - neg should be positive
        # Shape: (n_pos, n_neg)
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        
        # Soft hinge loss with temperature
        loss = torch.log(1 + torch.exp(-self.gamma * (diff - self.margin)))
        
        return loss.mean()


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss combining multiple loss functions.
    
    Combines Focal Loss + AUC Loss + Label Smoothing for
    comprehensive training signal.
    
    Args:
        focal_weight: Weight for focal loss (default: 0.5)
        auc_weight: Weight for AUC loss (default: 0.3)
        smooth_weight: Weight for label smoothing BCE (default: 0.2)
    """
    
    def __init__(
        self,
        focal_weight: float = 0.5,
        auc_weight: float = 0.3,
        smooth_weight: float = 0.2,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        smoothing: float = 0.1,
        auc_margin: float = 1.0,
    ):
        super().__init__()
        
        self.focal_weight = focal_weight
        self.auc_weight = auc_weight
        self.smooth_weight = smooth_weight
        
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.auc = AUCMaxLoss(margin=auc_margin)
        self.smooth = LabelSmoothingBCE(smoothing=smoothing)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(inputs, targets)
        auc_loss = self.auc(inputs, targets)
        smooth_loss = self.smooth(inputs, targets)
        
        total = (
            self.focal_weight * focal_loss +
            self.auc_weight * auc_loss +
            self.smooth_weight * smooth_loss
        )
        
        return total

