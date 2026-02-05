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


def compute_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
) -> float:
    """
    Compute balanced accuracy (average of per-class recall).
    
    More appropriate than accuracy for imbalanced datasets.
    
    Args:
        y_true: Ground truth class indices
        y_pred: Predicted class indices
        num_classes: Number of classes
        
    Returns:
        Balanced accuracy score
    """
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
) -> float:
    """
    Compute macro F1 score (unweighted mean of per-class F1).
    
    Args:
        y_true: Ground truth class indices
        y_pred: Predicted class indices
        num_classes: Number of classes
        
    Returns:
        Macro F1 score
    """
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
) -> np.ndarray:
    """
    Compute per-class accuracy (recall for each class).
    
    Args:
        y_true: Ground truth class indices
        y_pred: Predicted class indices
        num_classes: Number of classes
        
    Returns:
        Array of per-class accuracy scores
    """
    per_class_acc = np.zeros(num_classes)
    
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc[c] = (y_pred[mask] == c).mean()
    
    return per_class_acc


class MultiClassMetrics:
    """
    Multi-class metrics accumulator for validation.
    
    Accumulates predictions and computes balanced accuracy,
    macro F1, and per-class metrics at the end of epoch.
    """
    
    def __init__(self, num_classes: int = 7, class_names: list = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions."""
        self.y_true = []
        self.y_pred = []
        self.y_prob = []
    
    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray = None,
        y_prob: np.ndarray = None,
    ):
        """
        Update with batch predictions.
        
        Args:
            y_true: Batch ground truth labels (class indices)
            y_pred: Batch predicted classes (optional, computed from y_prob if not given)
            y_prob: Batch predicted probabilities (N, C)
        """
        if hasattr(y_true, 'cpu'):
            y_true = y_true.cpu().numpy()
        if y_prob is not None and hasattr(y_prob, 'cpu'):
            y_prob = y_prob.cpu().numpy()
        if y_pred is not None and hasattr(y_pred, 'cpu'):
            y_pred = y_pred.cpu().numpy()
        
        # Compute predictions from probabilities if not provided
        if y_pred is None and y_prob is not None:
            y_pred = y_prob.argmax(axis=1)
        
        self.y_true.extend(y_true.flatten().tolist())
        if y_pred is not None:
            self.y_pred.extend(y_pred.flatten().tolist())
        if y_prob is not None:
            self.y_prob.extend(y_prob.tolist())
    
    def compute(self) -> dict:
        """
        Compute all metrics from accumulated predictions.
        
        Returns:
            Dictionary with balanced_accuracy, macro_f1, per_class_accuracy
        """
        if len(self.y_true) == 0:
            return {
                'balanced_accuracy': 0.0,
                'macro_f1': 0.0,
                'per_class_accuracy': np.zeros(self.num_classes),
            }
        
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        balanced_acc = compute_balanced_accuracy(y_true, y_pred, self.num_classes)
        macro_f1 = compute_macro_f1(y_true, y_pred, self.num_classes)
        per_class_acc = compute_per_class_accuracy(y_true, y_pred, self.num_classes)
        
        return {
            'balanced_accuracy': balanced_acc,
            'macro_f1': macro_f1,
            'per_class_accuracy': per_class_acc,
        }
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(
            np.array(self.y_true),
            np.array(self.y_pred),
            labels=list(range(self.num_classes))
        )
    
    def __repr__(self) -> str:
        return f"MultiClassMetrics(num_classes={self.num_classes}, n_samples={len(self.y_true)})"
