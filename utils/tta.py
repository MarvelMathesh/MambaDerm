"""
Test-Time Augmentation (TTA) Inference Module

Provides TTA strategies for improved prediction robustness.
Averages predictions across augmented versions of input images.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable
import numpy as np


class TTAWrapper(nn.Module):
    """
    Test-Time Augmentation wrapper for MambaDerm.
    
    Applies multiple augmentations at inference time and averages predictions
    for more robust results.
    
    Args:
        model: The base model (MambaDerm)
        tta_transforms: List of transform functions to apply
        merge_mode: How to merge predictions ('mean', 'max', 'geometric_mean')
    """
    
    def __init__(
        self,
        model: nn.Module,
        tta_transforms: Optional[List[Callable]] = None,
        merge_mode: str = 'mean',
    ):
        super().__init__()
        self.model = model
        self.merge_mode = merge_mode
        
        # Default TTA transforms (applied on tensor level)
        if tta_transforms is None:
            self.tta_transforms = [
                lambda x: x,  # Original
                lambda x: torch.flip(x, dims=[-1]),  # Horizontal flip
                lambda x: torch.flip(x, dims=[-2]),  # Vertical flip
                lambda x: torch.rot90(x, k=1, dims=[-2, -1]),  # Rotate 90
                lambda x: torch.rot90(x, k=2, dims=[-2, -1]),  # Rotate 180
            ]
        else:
            self.tta_transforms = tta_transforms
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with TTA.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            tabular: Tabular features of shape (B, num_features)
            
        Returns:
            Averaged logits of shape (B, num_classes)
        """
        self.model.eval()
        
        all_outputs = []
        
        for transform in self.tta_transforms:
            # Apply augmentation
            x_aug = transform(x)
            
            # Get prediction
            output = self.model(x_aug, tabular)
            all_outputs.append(output)
        
        # Stack outputs: (num_transforms, B, num_classes)
        all_outputs = torch.stack(all_outputs, dim=0)
        
        # Merge predictions
        if self.merge_mode == 'mean':
            merged = all_outputs.mean(dim=0)
        elif self.merge_mode == 'max':
            merged = all_outputs.max(dim=0)[0]
        elif self.merge_mode == 'geometric_mean':
            # Apply sigmoid first, geometric mean, then inverse sigmoid
            probs = torch.sigmoid(all_outputs)
            geo_mean = torch.exp(torch.log(probs + 1e-8).mean(dim=0))
            merged = torch.log(geo_mean / (1 - geo_mean + 1e-8))  # logit
        else:
            merged = all_outputs.mean(dim=0)
        
        return merged
    
    def predict_proba(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict probabilities with TTA.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            tabular: Tabular features of shape (B, num_features)
            
        Returns:
            Probabilities of shape (B, num_classes)
        """
        logits = self.forward(x, tabular)
        return torch.sigmoid(logits)


class TTAInference:
    """
    Comprehensive TTA inference pipeline.
    
    Supports multiple TTA strategies and provides easy-to-use interface
    for inference on test data.
    
    Args:
        model: The trained MambaDerm model
        device: Device to run inference on
        tta_level: TTA level ('none', 'light', 'full')
        batch_size: Batch size for inference
    """
    
    TTA_LEVELS = {
        'none': 1,  # No TTA
        'light': 3,  # Original + HFlip + VFlip
        'full': 5,   # Original + HFlip + VFlip + Rot90 + Rot180
        'extreme': 8,  # Above + Rot270 + Transpose + Transpose+Rot90
    }
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        tta_level: str = 'light',
        merge_mode: str = 'mean',
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tta_level = tta_level
        
        # Setup transforms based on TTA level
        self.transforms = self._get_transforms_for_level(tta_level)
        
        self.tta_wrapper = TTAWrapper(
            model=model,
            tta_transforms=self.transforms,
            merge_mode=merge_mode,
        )
        self.tta_wrapper.to(self.device)
    
    def _get_transforms_for_level(self, level: str) -> List[Callable]:
        """Get transform list based on TTA level."""
        transforms = [lambda x: x]  # Original always included
        
        if level in ['light', 'full', 'extreme']:
            transforms.append(lambda x: torch.flip(x, dims=[-1]))  # HFlip
            transforms.append(lambda x: torch.flip(x, dims=[-2]))  # VFlip
        
        if level in ['full', 'extreme']:
            transforms.append(lambda x: torch.rot90(x, k=1, dims=[-2, -1]))  # Rot90
            transforms.append(lambda x: torch.rot90(x, k=2, dims=[-2, -1]))  # Rot180
        
        if level == 'extreme':
            transforms.append(lambda x: torch.rot90(x, k=3, dims=[-2, -1]))  # Rot270
            transforms.append(lambda x: x.transpose(-2, -1))  # Transpose
            transforms.append(lambda x: torch.rot90(x.transpose(-2, -1), k=1, dims=[-2, -1]))
        
        return transforms
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Run TTA inference on a batch.
        
        Args:
            images: Batch of images (B, 3, H, W)
            tabular: Batch of tabular features (B, num_features)
            
        Returns:
            Numpy array of probabilities (B,)
        """
        images = images.to(self.device)
        if tabular is not None:
            tabular = tabular.to(self.device)
        
        probs = self.tta_wrapper.predict_proba(images, tabular)
        
        return probs.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def predict_dataset(
        self,
        dataloader,
        return_ids: bool = False,
    ):
        """
        Run TTA inference on entire dataset.
        
        Args:
            dataloader: DataLoader for test dataset
            return_ids: Whether to return sample IDs (for test datasets)
            
        Returns:
            If return_ids: (predictions, ids) tuple
            Else: predictions array
        """
        all_preds = []
        all_ids = []
        
        for batch in dataloader:
            if return_ids:
                images, tabular, ids = batch
                all_ids.extend(ids)
            else:
                images, tabular, _ = batch
            
            preds = self.predict(images, tabular)
            all_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
        
        predictions = np.array(all_preds)
        
        if return_ids:
            return predictions, all_ids
        return predictions


def get_tta_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device = None,
    tta_level: str = 'light',
    merge_mode: str = 'mean',
) -> np.ndarray:
    """
    Convenience function for TTA inference.
    
    Args:
        model: Trained MambaDerm model
        dataloader: DataLoader for inference
        device: Device to use
        tta_level: TTA level ('none', 'light', 'full', 'extreme')
        merge_mode: How to merge predictions ('mean', 'max', 'geometric_mean')
        
    Returns:
        Numpy array of predictions
    """
    inference = TTAInference(
        model=model,
        device=device,
        tta_level=tta_level,
        merge_mode=merge_mode,
    )
    
    return inference.predict_dataset(dataloader)
