"""
Image Transforms

Augmentation pipelines for training and validation using albumentations.
Includes dermoscopy-specific augmentations for skin lesion classification.
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


class HairArtifactSimulation(ImageOnlyTransform):
    """
    Simulate dermoscopic hair artifacts overlaid on skin images.
    
    Draws random thin dark lines to mimic hairs that commonly
    occlude lesion features in dermoscopy images.
    """
    
    def __init__(
        self,
        num_hairs_range=(3, 12),
        hair_color_range=(0, 40),
        thickness_range=(1, 3),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.num_hairs_range = num_hairs_range
        self.hair_color_range = hair_color_range
        self.thickness_range = thickness_range
    
    def apply(self, img, **params):
        h, w = img.shape[:2]
        result = img.copy()
        
        num_hairs = np.random.randint(*self.num_hairs_range)
        for _ in range(num_hairs):
            # Random start/end points
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            # Hairs are roughly linear with slight curve
            angle = np.random.uniform(0, np.pi)
            length = np.random.randint(h // 4, h)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            color_val = np.random.randint(*self.hair_color_range)
            color = (color_val, color_val, color_val)
            thickness = np.random.randint(*self.thickness_range)
            
            cv2.line(result, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        
        return result
    
    def get_transform_init_args_names(self):
        return ("num_hairs_range", "hair_color_range", "thickness_range")


class ColorConstancy(ImageOnlyTransform):
    """
    Shades of Gray color constancy algorithm.
    
    Normalizes color channels using Minkowski norm to reduce
    illumination variation across different dermoscopes/lighting.
    """
    
    def __init__(self, power=6, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.power = power
    
    def apply(self, img, **params):
        img_float = img.astype(np.float32) / 255.0
        
        # Minkowski norm per channel
        norm = np.power(img_float, self.power)
        norm = np.power(norm.mean(axis=(0, 1)), 1.0 / self.power)
        
        # Avoid division by zero
        norm = np.maximum(norm, 1e-7)
        
        # Normalize
        scale = norm.mean() / norm
        result = img_float * scale[np.newaxis, np.newaxis, :]
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        
        return result
    
    def get_transform_init_args_names(self):
        return ("power",)


class MicroscopeEffect(ImageOnlyTransform):
    """
    Simulate circular vignette from dermoscope lens.
    
    Darkens corners to mimic the circular field of view
    commonly seen in dermoscopic images.
    """
    
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
    
    def apply(self, img, **params):
        h, w = img.shape[:2]
        result = img.copy()
        
        # Create circular mask
        radius = min(h, w) // 2
        center = (w // 2, h // 2)
        
        # Random offset for center
        cx = center[0] + np.random.randint(-w // 20, w // 20)
        cy = center[1] + np.random.randint(-h // 20, h // 20)
        
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        
        # Smooth vignette falloff
        r = radius * np.random.uniform(0.85, 1.0)
        mask = np.clip(1.0 - np.maximum(0, dist - r * 0.7) / (r * 0.3), 0, 1)
        mask = mask.astype(np.float32)
        
        for c in range(3):
            result[:, :, c] = (result[:, :, c] * mask).astype(np.uint8)
        
        return result
    
    def get_transform_init_args_names(self):
        return ()


def get_train_transforms(img_size: int = 224):
    """
    Training augmentation pipeline with dermoscopy-specific augmentations.
    """
    return A.Compose([
        # Geometric transforms
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Scale and rotate
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=0,
            p=0.7,
        ),
        
        # Dermoscopy-specific augmentations
        HairArtifactSimulation(p=0.3),
        ColorConstancy(power=6, p=0.3),
        MicroscopeEffect(p=0.2),
        
        # Color transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.75,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.5,
        ),
        A.RGBShift(
            r_shift_limit=10,
            g_shift_limit=10,
            b_shift_limit=10,
            p=0.3,
        ),
        
        # CLAHE for local contrast
        A.CLAHE(clip_limit=4.0, p=0.5),
        
        # Simulate lower-resolution captures
        A.Downscale(
            scale_min=0.5,
            scale_max=0.9,
            p=0.2,
        ),
        
        # Blur and noise (one of)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
        ], p=0.5),
        
        # Distortion (one of)
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.3),
        
        # Resize
        A.Resize(img_size, img_size),
        
        # Coarse dropout (cutout)
        A.CoarseDropout(
            max_holes=4,
            max_height=int(img_size * 0.1),
            max_width=int(img_size * 0.1),
            min_holes=1,
            min_height=int(img_size * 0.05),
            min_width=int(img_size * 0.05),
            fill_value=0,
            p=0.5,
        ),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224):
    """
    Get validation/test transform pipeline.
    
    Only resize and normalize, no augmentation.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 224):
    """
    Get test-time augmentation transforms.
    
    Returns a list of transforms for TTA.
    """
    base = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    
    transforms_list = [
        # Original
        A.Compose(base),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        # Vertical flip  
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        # Rotate 90
        A.Compose([A.Rotate(limit=(90, 90), p=1.0)] + base),
        # Rotate 180
        A.Compose([A.Rotate(limit=(180, 180), p=1.0)] + base),
    ]
    
    return transforms_list
