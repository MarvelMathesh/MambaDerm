"""
HAM10000 Dataset

Dataset loader for HAM10000 skin cancer classification dataset.
Supports 7-class multi-class classification with tabular features.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# HAM10000 class mapping (dx column)
HAM10000_CLASSES = {
    'akiec': 0,  # Actinic keratoses and intraepithelial carcinoma
    'bcc': 1,    # Basal cell carcinoma
    'bkl': 2,    # Benign keratosis-like lesions
    'df': 3,     # Dermatofibroma
    'mel': 4,    # Melanoma (most critical!)
    'nv': 5,     # Melanocytic nevi
    'vasc': 6,   # Vascular lesions
}

HAM10000_CLASS_NAMES = [
    'Actinic Keratoses',
    'Basal Cell Carcinoma', 
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular Lesions',
]

# Localization mapping
LOCALIZATION_MAPPING = {
    'scalp': 0,
    'ear': 1,
    'face': 2,
    'back': 3,
    'trunk': 4,
    'chest': 5,
    'upper extremity': 6,
    'abdomen': 7,
    'lower extremity': 8,
    'genital': 9,
    'neck': 10,
    'hand': 11,
    'foot': 12,
    'acral': 13,
    'unknown': 14,
}

SEX_MAPPING = {'male': 0, 'female': 1, 'unknown': 2}


class HAM10000Dataset(Dataset):
    """
    HAM10000 Dataset for multi-class skin cancer classification.
    
    Loads dermoscopic images from JPG files and metadata from CSV.
    Supports 7-class classification with class-balanced sampling.
    
    Args:
        data_dir: Path to HAM10000 data directory (contains HAM10000_metadata.csv
                  and HAM10000_images_part_1/, HAM10000_images_part_2/ folders)
        split: 'train' or 'val'
        fold: Cross-validation fold (0-4)
        n_folds: Total number of folds
        transform: Image transforms (albumentations)
        quick_test: If True, use subset for quick testing
        quick_test_samples: Number of samples for quick test
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        fold: int = 0,
        n_folds: int = 5,
        transform=None,
        quick_test: bool = False,
        quick_test_samples: int = 1000,
        norm_stats: dict = None,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.norm_stats = norm_stats
        self.num_classes = 7
        
        # Find image directories
        self.image_dirs = self._find_image_dirs()
        
        # Load metadata
        csv_path = self.data_dir / 'HAM10000_metadata.csv'
        self.df = pd.read_csv(csv_path)
        
        # Handle age
        self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
        # self.df['age'] = self.df['age'].fillna(self.df['age'].median()) # Age fillna moved to _prepare_tabular_features
        
        # Handle sex
        self.df['sex'] = self.df['sex'].fillna('unknown')
        
        # Handle localization
        self.df['localization'] = self.df['localization'].fillna('unknown')
        
        # Encode labels
        self.df['label'] = self.df['dx'].map(HAM10000_CLASSES)
        
        # Create cross-validation splits
        self._create_cv_split(fold, n_folds)
        
        # Quick test mode (ensure class balance)
        if quick_test:
            self._subsample_balanced(quick_test_samples)
        
        # Build image path lookup
        self._build_image_paths()
        
        # Prepare tabular features
        self._prepare_tabular_features()
        
        # Compute class weights for balanced training
        self._compute_class_weights()
        
        # Print dataset info
        print(f"HAM10000 {split} dataset: {len(self.df)} samples")
        print(f"  Class distribution: {dict(self.df['dx'].value_counts())}")
    
    def _find_image_dirs(self) -> List[Path]:
        """Find image directories (uppercase or lowercase naming)."""
        dirs = []
        for name in ['HAM10000_images_part_1', 'HAM10000_images_part_2',
                     'ham10000_images_part_1', 'ham10000_images_part_2']:
            path = self.data_dir / name
            if path.exists():
                dirs.append(path)
        
        if not dirs:
            raise FileNotFoundError(f"No image directories found in {self.data_dir}")
        
        return dirs
    
    def _create_cv_split(self, fold: int, n_folds: int):
        """Create stratified k-fold split by lesion_id."""
        from sklearn.model_selection import StratifiedGroupKFold
        
        # Use lesion_id for grouping (same lesion = same split)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        y = self.df['label'].values
        groups = self.df['lesion_id'].values
        
        for i, (train_idx, val_idx) in enumerate(sgkf.split(self.df, y, groups)):
            if i == fold:
                if self.split == 'train':
                    self.df = self.df.iloc[train_idx].reset_index(drop=True)
                else:
                    self.df = self.df.iloc[val_idx].reset_index(drop=True)
                break
    
    def _subsample_balanced(self, n_samples: int):
        """Subsample while maintaining class balance."""
        samples_per_class = max(1, n_samples // self.num_classes)
        
        balanced_dfs = []
        for label in range(self.num_classes):
            class_df = self.df[self.df['label'] == label]
            n = min(len(class_df), samples_per_class)
            balanced_dfs.append(class_df.sample(n=n, random_state=42))
        
        self.df = pd.concat(balanced_dfs, ignore_index=True)
    
    def _build_image_paths(self):
        """Build lookup from image_id to file path."""
        self.image_paths = {}
        
        for image_dir in self.image_dirs:
            for img_file in image_dir.glob('*.jpg'):
                image_id = img_file.stem  # ISIC_0024306
                self.image_paths[image_id] = img_file
        
        # Verify all images exist
        missing = []
        for image_id in self.df['image_id']:
            if image_id not in self.image_paths:
                missing.append(image_id)
        
        if missing:
            print(f"  Warning: {len(missing)} images not found, filtering...")
            self.df = self.df[~self.df['image_id'].isin(missing)].reset_index(drop=True)
    
    def _prepare_tabular_features(self):
        """Prepare tabular features: age (normalized) + sex + localization."""
        # Age: normalize to 0-1 range
        # Normalize age using training set statistics to prevent leakage
        age_series = self.df['age'] # Use the original series for stats calculation
        
        if self.norm_stats is not None:
            age_median = self.norm_stats['age_median']
            age_min = self.norm_stats['age_min']
            age_max = self.norm_stats['age_max']
        else:
            # Compute stats from current dataset (should be training set)
            age_median = age_series.median()
            age_min = age_series.min()
            age_max = age_series.max()
        
        # Fill NaNs with the determined median
        age_filled = age_series.fillna(age_median)
        
        # Normalize age
        age_range = age_max - age_min
        if age_range > 0:
            age_norm = (age_filled - age_min) / age_range
        else:
            age_norm = pd.Series(0.0, index=self.df.index) # All ages are the same, normalize to 0
        
        self.df['age_norm'] = age_norm
        
        # Store for get_norm_stats
        self._age_median = float(age_median)
        self._age_min = float(age_min)
        self._age_max = float(age_max)
        
        # Sex: one-hot encode (3 values: male, female, unknown)
        sex = self.df['sex'].map(SEX_MAPPING).fillna(2).values.astype(np.int64)
        sex_onehot = np.eye(3)[sex].astype(np.float32)
        
        # Localization: encode to index, then one-hot
        loc = self.df['localization'].map(
            lambda x: LOCALIZATION_MAPPING.get(x, LOCALIZATION_MAPPING['unknown'])
        ).values.astype(np.int64)
        loc_onehot = np.eye(len(LOCALIZATION_MAPPING))[loc].astype(np.float32)
        
        # Concatenate: age (1) + sex (3) + localization (15) = 19 features
        age_arr = np.array(self.df['age_norm'].values, dtype=np.float32)
        self.tabular_features = np.concatenate([
            age_arr.reshape(-1, 1),
            sex_onehot,
            loc_onehot,
        ], axis=1).astype(np.float32)
        
        self.num_tabular_features = self.tabular_features.shape[1]
        
        # Store labels
        self.labels = self.df['label'].values.astype(np.int64)
        self.image_ids = self.df['image_id'].values
    
    def get_norm_stats(self) -> dict:
        """Return normalization statistics for cross-split consistency.
        
        Pass the result of this method to the `norm_stats` parameter of
        validation/test dataset constructors to prevent data leakage.
        """
        return {
            'age_median': self._age_median,
            'age_min': self._age_min,
            'age_max': self._age_max,
        }
    
    def _compute_class_weights(self):
        """Compute inverse frequency class weights for balanced training."""
        class_counts = np.bincount(self.labels, minlength=self.num_classes)
        
        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum() * self.num_classes
        
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        
        # Sample weights for sampler
        self.sample_weights = torch.tensor(
            [weights[label] for label in self.labels],
            dtype=torch.float32
        )
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            image: Transformed image tensor (C, H, W)
            tabular: Tabular features tensor (19,)
            label: Class label tensor (int64 for cross-entropy)
        """
        image_id = self.image_ids[idx]
        
        # Load image
        img_path = self.image_paths[image_id]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        else:
            # Default: convert to tensor and normalize
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Tabular features
        tabular = torch.from_numpy(self.tabular_features[idx])
        
        # Label (int64 for CrossEntropyLoss)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img, tabular, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss function."""
        return self.class_weights
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for WeightedRandomSampler."""
        return self.sample_weights
    
    @staticmethod
    def get_class_names() -> List[str]:
        """Get human-readable class names."""
        return HAM10000_CLASS_NAMES


def get_ham10000_class_weights(data_dir: Union[str, Path]) -> torch.Tensor:
    """
    Compute class weights from full dataset for loss weighting.
    
    Call this before training to get weights.
    """
    csv_path = Path(data_dir) / 'HAM10000_metadata.csv'
    df = pd.read_csv(csv_path)
    
    class_counts = df['dx'].map(HAM10000_CLASSES).value_counts().sort_index()
    weights = 1.0 / class_counts.values
    weights = weights / weights.sum() * len(HAM10000_CLASSES)
    
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Quick test
    print("Testing HAM10000Dataset...")
    
    dataset = HAM10000Dataset(
        data_dir='./skin-cancer-mnist-ham10000',
        split='train',
        fold=0,
        quick_test=True,
        quick_test_samples=100,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Num tabular features: {dataset.num_tabular_features}")
    print(f"Class weights: {dataset.get_class_weights()}")
    
    # Test single item
    img, tabular, label = dataset[0]
    print(f"\nSample item:")
    print(f"  Image shape: {img.shape}")
    print(f"  Tabular shape: {tabular.shape}")
    print(f"  Label: {label} ({HAM10000_CLASS_NAMES[label]})")
    
    print("\n✓ HAM10000Dataset test passed!")
