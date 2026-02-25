"""
ISIC 2024 Dataset

Dataset loader for ISIC 2024 skin lesion images and metadata.
Supports HDF5 image storage and comprehensive tabular features.
"""

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# Numerical features from ISIC 2024 metadata
NUMERICAL_FEATURES = [
    'age_approx',
    'clin_size_long_diam_mm',
    'tbp_lv_A',
    'tbp_lv_Aext',
    'tbp_lv_B',
    'tbp_lv_Bext',
    'tbp_lv_C',
    'tbp_lv_Cext',
    'tbp_lv_H',
    'tbp_lv_Hext',
    'tbp_lv_L',
    'tbp_lv_Lext',
    'tbp_lv_areaMM2',
    'tbp_lv_area_perim_ratio',
    'tbp_lv_color_std_mean',
    'tbp_lv_deltaA',
    'tbp_lv_deltaB',
    'tbp_lv_deltaL',
    'tbp_lv_deltaLB',
    'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity',
    'tbp_lv_minorAxisMM',
    'tbp_lv_nevi_confidence',
    'tbp_lv_norm_border',
    'tbp_lv_norm_color',
    'tbp_lv_perimeterMM',
    'tbp_lv_radial_color_std_max',
    'tbp_lv_stdL',
    'tbp_lv_stdLExt',
    'tbp_lv_symm_2axis',
    'tbp_lv_symm_2axis_angle',
    'tbp_lv_x',
    'tbp_lv_y',
    'tbp_lv_z',
]

# Categorical features
CATEGORICAL_FEATURES = [
    'sex',
    'anatom_site_general',
    'tbp_tile_type',
    'tbp_lv_location',
    'tbp_lv_location_simple',
    'attribution',
]

# Mapping for categorical values
CATEGORICAL_MAPPINGS = {
    'sex': {'female': 1, 'male': 2},
    'anatom_site_general': {
        'head/neck': 1, 'upper extremity': 2, 'lower extremity': 3,
        'torso': 4, 'palms/soles': 5, 'oral/genital': 6
    },
    'tbp_tile_type': {'3D: white': 1, '3D: XP': 2, '2D: other': 3},
    'tbp_lv_location': {}, # Will be built dynamically
    'tbp_lv_location_simple': {
        'head & neck': 1, 'upper limb': 2, 'lower limb': 3,
        'torso': 4, 'other': 5
    },
    'attribution': {}, # Will be built dynamically
}


class ISICDataset(Dataset):
    """
    ISIC 2024 Dataset for training and validation.
    
    Loads images from HDF5 file and metadata from CSV.
    Supports cross-validation splits and class balancing.
    
    Args:
        data_dir: Path to data directory
        split: 'train' or 'val'
        hdf5_name: Name of HDF5 file
        csv_name: Name of metadata CSV file
        fold: Cross-validation fold (0-4)
        n_folds: Total number of folds
        transform: Image transforms
        target_col: Target column name
        norm_stats: Optional dict with pre-computed 'mean', 'std', 'median' for 
                   normalization. If None, computed from this split's data.
                   For validation/test sets, pass training set statistics to
                   prevent data leakage.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        hdf5_name: str = 'train-image.hdf5',
        csv_name: str = 'train-metadata.csv',
        fold: int = 0,
        n_folds: int = 5,
        transform = None,
        target_col: str = 'target',
        quick_test: bool = False,
        quick_test_samples: int = 1000,
        norm_stats: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_col = target_col
        self.norm_stats = norm_stats
        
        # Load metadata
        csv_path = self.data_dir / csv_name
        self.df = pd.read_csv(csv_path, low_memory=False)
        
        # Handle missing values in age
        self.df['age_approx'] = self.df['age_approx'].replace('NA', np.nan)
        self.df['age_approx'] = pd.to_numeric(self.df['age_approx'], errors='coerce')
        
        # Build categorical mappings for dynamic columns
        self._build_categorical_mappings()
        
        # Create cross-validation splits
        self._create_cv_split(fold, n_folds)
        
        # Quick test mode
        if quick_test:
            self.df = self.df.head(quick_test_samples)
            # Ensure we have some positive samples
            pos_df = self.df[self.df[target_col] == 1]
            neg_df = self.df[self.df[target_col] == 0]
            if len(pos_df) < 10:
                # Oversample positives for quick test
                all_pos = pd.read_csv(csv_path, low_memory=False)
                all_pos = all_pos[all_pos[target_col] == 1].head(50)
                self.df = pd.concat([neg_df.head(quick_test_samples - 50), all_pos])
        
        # Store ISIC IDs
        self.isic_ids = self.df['isic_id'].values
        
        # Precompute tabular features
        self._prepare_tabular_features()
        
        # HDF5 file (opened lazily)
        self.hdf5_path = self.data_dir / hdf5_name
        self._hdf5_file = None
        
        print(f"Loaded {split} dataset: {len(self.df)} samples")
        if target_col in self.df.columns:
            pos_count = self.df[target_col].sum()
            print(f"  Positive: {pos_count}, Negative: {len(self.df) - pos_count}")
    
    def _build_categorical_mappings(self):
        """Build instance-level mappings for categorical columns.
        
        Uses a copy of the global mappings to avoid mutating shared state
        across dataset instances (train/val), which would cause data leakage.
        """
        import copy
        self._categorical_mappings = copy.deepcopy(CATEGORICAL_MAPPINGS)
        for col in ['tbp_lv_location', 'attribution']:
            if col in self.df.columns:
                unique_vals = self.df[col].dropna().unique()
                self._categorical_mappings[col] = {v: i+1 for i, v in enumerate(unique_vals)}
    
    def _create_cv_split(self, fold: int, n_folds: int):
        """Create stratified group k-fold split by patient_id."""
        from sklearn.model_selection import StratifiedGroupKFold
        
        if self.target_col not in self.df.columns:
            return  # Test set, no split needed
        
        # Create fold indices
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        y = self.df[self.target_col].values
        groups = self.df['patient_id'].values
        
        for i, (train_idx, val_idx) in enumerate(sgkf.split(self.df, y, groups)):
            if i == fold:
                if self.split == 'train':
                    self.df = self.df.iloc[train_idx].reset_index(drop=True)
                else:
                    self.df = self.df.iloc[val_idx].reset_index(drop=True)
                break
    
    def _prepare_tabular_features(self):
        """Precompute numerical and categorical features."""
        # Numerical features
        num_df = self.df[NUMERICAL_FEATURES].copy()
        
        # Use provided statistics or compute from this split
        if self.norm_stats is not None:
            # Use pre-computed statistics from training set
            medians = self.norm_stats['median']
            self.num_mean = self.norm_stats['mean']
            self.num_std = self.norm_stats['std']
        else:
            # Compute statistics from this split (only for training)
            medians = {}
            for col in NUMERICAL_FEATURES:
                if col in num_df.columns:
                    medians[col] = num_df[col].median()
                else:
                    medians[col] = 0.0
        
        # Fill missing values with median
        for col in NUMERICAL_FEATURES:
            if col in num_df.columns:
                median_val = medians.get(col, 0.0)
                num_df[col] = num_df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        self.numerical_features = num_df.values.astype(np.float32)
        
        # Normalize numerical features
        if self.norm_stats is None:
            # Compute mean/std from this split
            self.num_mean = np.nanmean(self.numerical_features, axis=0)
            self.num_std = np.nanstd(self.numerical_features, axis=0) + 1e-8
            # Store medians for get_norm_stats
            self._medians = medians
        
        self.numerical_features = (self.numerical_features - self.num_mean) / self.num_std
        
        # Categorical features
        cat_features = []
        for col in CATEGORICAL_FEATURES:
            if col in self.df.columns:
                mapping = self._categorical_mappings.get(col, {})
                cat_values = self.df[col].map(mapping).fillna(0).astype(np.int64)
                cat_features.append(cat_values.values)
            else:
                cat_features.append(np.zeros(len(self.df), dtype=np.int64))
        
        self.categorical_features = np.stack(cat_features, axis=1)
        
        # Targets
        if self.target_col in self.df.columns:
            self.targets = self.df[self.target_col].values.astype(np.float32)
        else:
            self.targets = np.zeros(len(self.df), dtype=np.float32)
    
    def get_norm_stats(self) -> Dict:
        """
        Get normalization statistics for reuse in validation/test sets.
        
        Call this on training dataset and pass result to validation dataset
        via norm_stats parameter to prevent data leakage.
        
        Returns:
            Dict with 'mean', 'std', 'median' arrays
        """
        if hasattr(self, '_medians'):
            medians = self._medians
        else:
            medians = {col: 0.0 for col in NUMERICAL_FEATURES}
        
        return {
            'mean': self.num_mean,
            'std': self.num_std,
            'median': medians,
        }
    
    @property
    def hdf5_file(self):
        """Lazily open HDF5 file."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Returns:
            image: Transformed image tensor (C, H, W)
            tabular: Tabular features tensor (num_features,)
            target: Target label tensor
        """
        isic_id = self.isic_ids[idx]
        
        # Load image from HDF5
        img_bytes = self.hdf5_file[isic_id][()]
        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        else:
            # Default: convert to tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Tabular features: concatenate numerical and categorical
        numerical = self.numerical_features[idx]
        categorical = self.categorical_features[idx]
        tabular = np.concatenate([numerical, categorical.astype(np.float32)])
        tabular = torch.from_numpy(tabular)
        
        # Target
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return img, tabular, target
    
    def get_sample_weights(self) -> np.ndarray:
        """Get sample weights for balanced sampling."""
        pos_count = self.targets.sum()
        neg_count = len(self.targets) - pos_count
        
        # Weight inversely proportional to class frequency
        pos_weight = len(self.targets) / (2 * pos_count) if pos_count > 0 else 1.0
        neg_weight = len(self.targets) / (2 * neg_count) if neg_count > 0 else 1.0
        
        weights = np.where(self.targets == 1, pos_weight, neg_weight)
        return weights
    
    def __del__(self):
        """Close HDF5 file on deletion."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()


class ISICTestDataset(Dataset):
    """
    ISIC 2024 Test Dataset (without labels).
    
    For inference on test set.
    Requires norm_stats from training set to prevent distribution mismatch.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        hdf5_name: str = 'test-image.hdf5',
        csv_name: str = 'test-metadata.csv',
        transform = None,
        norm_stats: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        if norm_stats is None:
            raise ValueError(
                "norm_stats is required for ISICTestDataset to prevent "
                "train/test distribution mismatch. Pass the output of "
                "train_dataset.get_norm_stats() here."
            )
        self.norm_stats = norm_stats
        
        # Load metadata
        self.df = pd.read_csv(self.data_dir / csv_name, low_memory=False)
        self.isic_ids = self.df['isic_id'].values
        
        # Prepare tabular features (using training statistics)
        self._prepare_tabular_features()
        
        # HDF5 file
        self.hdf5_path = self.data_dir / hdf5_name
        self._hdf5_file = None
        
        print(f"Loaded test dataset: {len(self.df)} samples")
    
    def _prepare_tabular_features(self):
        """Prepare tabular features for test set using training statistics."""
        # Handle missing features
        for col in NUMERICAL_FEATURES:
            if col not in self.df.columns:
                self.df[col] = 0.0
        
        num_df = self.df[NUMERICAL_FEATURES].copy()
        
        # Use training medians for imputation (not zeros!)
        medians = self.norm_stats.get('median', {})
        for col in NUMERICAL_FEATURES:
            median_val = medians.get(col, 0.0)
            if pd.isna(median_val):
                median_val = 0.0
            num_df[col] = num_df[col].fillna(median_val)
        
        # Apply training z-score normalization
        self.numerical_features = (
            num_df.values.astype(np.float32) - self.norm_stats['mean']
        ) / self.norm_stats['std']
        
        # Categorical features
        cat_features = []
        for col in CATEGORICAL_FEATURES:
            if col in self.df.columns:
                mapping = CATEGORICAL_MAPPINGS.get(col, {})
                cat_values = self.df[col].map(mapping).fillna(0).astype(np.int64)
                cat_features.append(cat_values.values)
            else:
                cat_features.append(np.zeros(len(self.df), dtype=np.int64))
        
        self.categorical_features = np.stack(cat_features, axis=1)
    
    @property
    def hdf5_file(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        isic_id = self.isic_ids[idx]
        
        # Load image
        img_bytes = self.hdf5_file[isic_id][()]
        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Tabular
        numerical = self.numerical_features[idx]
        categorical = self.categorical_features[idx]
        tabular = np.concatenate([numerical, categorical.astype(np.float32)])
        tabular = torch.from_numpy(tabular)
        
        return img, tabular, isic_id
    
    def __del__(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()


def hdf5_worker_init_fn(worker_id: int):
    """
    Worker init function for DataLoader with HDF5 datasets.
    
    HDF5 file handles are not fork-safe. Each DataLoader worker
    must re-open its own file handle to prevent corruption.
    
    Usage:
        DataLoader(dataset, ..., worker_init_fn=hdf5_worker_init_fn)
    """
    import torch.utils.data as data
    worker_info = data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Reset HDF5 handle so it will be re-opened in this worker process
        if hasattr(dataset, '_hdf5_file'):
            dataset._hdf5_file = None
