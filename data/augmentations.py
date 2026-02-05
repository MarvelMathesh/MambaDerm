"""
Advanced Augmentations for Dermoscopy Images

Includes:
- MixUp: Convex interpolation of images and labels
- CutMix: Region-based mixing
- RandomErasing: Randomly erase patches
- Dermoscopy-specific augmentations
"""

import numpy as np
import torch
from typing import Tuple, Optional


class MixUp:
    """
    MixUp augmentation for training.
    
    Linearly interpolates images and labels:
        x̃ = λ * x_i + (1-λ) * x_j
        ỹ = λ * y_i + (1-λ) * y_j
    
    Reference: "mixup: Beyond Empirical Risk Minimization"
    """
    
    def __init__(self, alpha: float = 0.4, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter (higher = more mixing)
            p: Probability of applying mixup
        """
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Apply MixUp to a batch.
        
        Args:
            images: (B, C, H, W) images
            labels: (B,) labels
            tabular: (B, F) optional tabular features
            
        Returns:
            mixed_images, mixed_labels, mixed_tabular, lam
        """
        if np.random.random() > self.p:
            return images, labels, tabular, torch.ones(1, device=images.device)
        
        batch_size = images.shape[0]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure lambda >= 0.5
        
        # Random permutation for mixing pairs
        indices = torch.randperm(batch_size, device=images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Mix labels
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        # Mix tabular if provided
        if tabular is not None:
            mixed_tabular = lam * tabular + (1 - lam) * tabular[indices]
        else:
            mixed_tabular = None
        
        return mixed_images, mixed_labels, mixed_tabular, torch.tensor(lam)


class CutMix:
    """
    CutMix augmentation for training.
    
    Cuts a rectangular region from one image and pastes to another,
    mixing labels proportionally to area.
    
    Reference: "CutMix: Regularization Strategy to Train Strong Classifiers"
    """
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying cutmix
        """
        self.alpha = alpha
        self.p = p
    
    def _rand_bbox(
        self, size: Tuple[int, ...], lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box."""
        W = size[3]
        H = size[2]
        
        # Ratio of box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Center of box
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bound box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Apply CutMix to a batch."""
        
        if np.random.random() > self.p:
            return images, labels, tabular, torch.ones(1, device=images.device)
        
        batch_size = images.shape[0]
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        indices = torch.randperm(batch_size, device=images.device)
        
        # Get bounding box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Mix images
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        # Mix labels
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        # Mix tabular proportionally
        if tabular is not None:
            mixed_tabular = lam * tabular + (1 - lam) * tabular[indices]
        else:
            mixed_tabular = None
        
        return mixed_images, mixed_labels, mixed_tabular, torch.tensor(lam)


class MixUpCutMix:
    """
    Combined MixUp and CutMix with configurable probabilities.
    
    Randomly selects between MixUp, CutMix, or no augmentation.
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.3,
        cutmix_prob: float = 0.3,
    ):
        """
        Args:
            mixup_alpha: MixUp beta parameter
            cutmix_alpha: CutMix beta parameter
            mixup_prob: Probability of MixUp
            cutmix_prob: Probability of CutMix
        """
        self.mixup = MixUp(alpha=mixup_alpha, p=1.0)
        self.cutmix = CutMix(alpha=cutmix_alpha, p=1.0)
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Apply MixUp or CutMix randomly."""
        
        r = np.random.random()
        
        if r < self.mixup_prob:
            return self.mixup(images, labels, tabular)
        elif r < self.mixup_prob + self.cutmix_prob:
            return self.cutmix(images, labels, tabular)
        else:
            return images, labels, tabular, torch.ones(1, device=images.device)


class RandomErasing:
    """
    Random Erasing augmentation.
    
    Randomly erases a rectangular region and replaces with random values.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return images
        
        B, C, H, W = images.shape
        area = H * W
        
        for i in range(B):
            for _ in range(10):  # Try up to 10 times
                target_area = np.random.uniform(*self.scale) * area
                aspect_ratio = np.random.uniform(*self.ratio)
                
                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))
                
                if w < W and h < H:
                    x1 = np.random.randint(0, W - w)
                    y1 = np.random.randint(0, H - h)
                    
                    if self.value == 'random':
                        images[i, :, y1:y1+h, x1:x1+w] = torch.randn(C, h, w, device=images.device)
                    else:
                        images[i, :, y1:y1+h, x1:x1+w] = self.value
                    break
        
        return images


def get_mixup_cutmix(
    mixup_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    prob: float = 0.5,
) -> MixUpCutMix:
    """Get MixUp/CutMix augmentation with equal probability."""
    return MixUpCutMix(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        mixup_prob=prob / 2,
        cutmix_prob=prob / 2,
    )
