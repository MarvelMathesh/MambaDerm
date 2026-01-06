"""
Metrics for ISIC 2024

pAUC at 80% TPR metric used in the ISIC 2024 challenge.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from typing import Tuple


def compute_pauc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_tpr: float = 0.80,
) -> float:
    """
    Compute partial AUC at specified minimum TPR.
    
    This is the official ISIC 2024 challenge metric.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        min_tpr: Minimum true positive rate (default: 0.80)
        
    Returns:
        Partial AUC score
    """
    # Convert to numpy if needed
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return 0.0
    
    # Rescale: set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = np.abs(y_true - 1)
    v_pred = -1.0 * y_pred  # Flip predictions
    
    max_fpr = 1.0 - min_tpr
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(v_gt, v_pred)
    
    if max_fpr == 1:
        return auc(fpr, tpr)
    
    # Add point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, 'right')
    
    if stop == 0:
        return 0.0
    if stop >= len(fpr):
        stop = len(fpr) - 1
    
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    
    tpr_at_max_fpr = np.interp(max_fpr, x_interp, y_interp)
    
    fpr_clipped = np.append(fpr[:stop], max_fpr)
    tpr_clipped = np.append(tpr[:stop], tpr_at_max_fpr)
    
    partial_auc = auc(fpr_clipped, tpr_clipped)
    
    return partial_auc


def compute_pauc_scaled(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_tpr: float = 0.80,
) -> float:
    """
    Compute scaled partial AUC (used in some ISIC implementations).
    
    Maps the pAUC to [0, 1] range.
    """
    max_fpr = 1.0 - min_tpr
    
    # Compute scaled version
    v_gt = np.abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_pred])
    
    try:
        partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    except ValueError:
        return 0.0
    
    # Unscale
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc


class pAUCMetric:
    """
    pAUC metric accumulator for validation.
    
    Accumulates predictions and computes pAUC at the end of epoch.
    """
    
    def __init__(self, min_tpr: float = 0.80):
        self.min_tpr = min_tpr
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions."""
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Update with batch predictions.
        
        Args:
            y_true: Batch ground truth labels
            y_pred: Batch predicted probabilities
        """
        if hasattr(y_true, 'cpu'):
            y_true = y_true.cpu().numpy()
        if hasattr(y_pred, 'cpu'):
            y_pred = y_pred.cpu().numpy()
        
        self.y_true.extend(y_true.flatten().tolist())
        self.y_pred.extend(y_pred.flatten().tolist())
    
    def compute(self) -> float:
        """Compute pAUC from accumulated predictions."""
        if len(self.y_true) == 0:
            return 0.0
        
        return compute_pauc(
            np.array(self.y_true),
            np.array(self.y_pred),
            self.min_tpr,
        )
    
    def __repr__(self) -> str:
        return f"pAUCMetric(min_tpr={self.min_tpr}, n_samples={len(self.y_true)})"
