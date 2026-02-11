"""
Image Transforms

Augmentation pipelines for training and validation using albumentations.
Based on ISIC 2024 winner strategies with domain-specific refinements.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 224):
    """
    Get training augmentation pipeline.
    
    Heavy augmentation based on ISIC 2024 winner strategies:
    - Geometric: Flip, Rotate90, ShiftScaleRotate
    - Color: RandomBrightnessContrast, HueSaturationValue, CLAHE, RGBShift
    - Resolution: Downscale (simulates lower-res clinical captures)
    - Blur/Noise: MotionBlur, MedianBlur, GaussNoise
    - Regularization: CoarseDropout
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
        
        # Distortion (one of) — no ElasticTransform (changes diagnostic morphology)
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.3),
        
        # Resize
        A.Resize(img_size, img_size),
        
        # Coarse dropout (cutout) — max 4 holes to avoid destroying lesion
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
