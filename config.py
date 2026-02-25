"""
MambaDerm Configuration

Centralized configuration for training and model parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Image settings
    img_size: int = 224
    in_channels: int = 3
    
    # CNN Stem
    stem_channels: int = 96
    stem_kernel_size: int = 4
    stem_stride: int = 4
    
    # VMamba backbone
    d_model: int = 192
    n_mamba_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dropout: float = 0.1
    
    # Tabular encoder
    num_numerical_features: int = 33
    num_categorical_features: int = 6
    categorical_embedding_dim: int = 8
    tabular_hidden_dim: int = 128
    
    # Classifier
    num_classes: int = 1
    classifier_dropout: float = 0.3


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_dir: Path = Path("../isic-2024-challenge")
    train_hdf5: str = "train-image.hdf5"
    test_hdf5: str = "test-image.hdf5"
    train_csv: str = "train-metadata.csv"
    test_csv: str = "test-metadata.csv"
    
    # Cross-validation
    n_folds: int = 5
    fold: int = 0
    
    # Sampling
    neg_sampling_ratio: float = 0.01  # 1:100 pos:neg
    oversampling_ratio: float = 0.003
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    """Training configuration."""
    # Training settings
    epochs: int = 30
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_top_k: int = 3
    
    # Logging
    log_interval: int = 100
    val_interval: int = 1
    
    # Early stopping
    early_stop_patience: int = 10
    
    # Reproducibility
    seed: int = 42
    
    # Quick test mode
    quick_test: bool = False
    quick_test_samples: int = 1000


@dataclass 
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Create directories if needed."""
        self.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()
