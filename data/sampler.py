"""
Data Samplers

Balanced and patient-aware sampling strategies for ISIC training.
"""

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler
from typing import Iterator, List, Optional


class BalancedSampler(Sampler):
    """
    Balanced sampler for handling class imbalance.
    
    Uses oversampling of minority class and undersampling of majority class.
    Based on ISIC 2024 winner strategies.
    
    Args:
        dataset: Dataset with targets attribute
        neg_sampling_ratio: Ratio of negative to positive samples (e.g., 0.01 = 1:100)
        oversample_ratio: Factor to oversample positive class
        shuffle: Whether to shuffle samples
    """
    
    def __init__(
        self,
        dataset,
        neg_sampling_ratio: float = 0.01,
        oversample_ratio: float = 2.0,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.neg_sampling_ratio = neg_sampling_ratio
        self.oversample_ratio = oversample_ratio
        self.shuffle = shuffle
        
        # Get targets
        self.targets = np.array(dataset.targets)
        
        # Get indices for each class
        self.pos_indices = np.where(self.targets == 1)[0]
        self.neg_indices = np.where(self.targets == 0)[0]
        
        # Calculate number of samples
        n_pos = len(self.pos_indices)
        n_neg = len(self.neg_indices)
        
        # Oversample positives
        n_pos_sampled = int(n_pos * oversample_ratio)
        
        # Undersample negatives based on ratio
        n_neg_sampled = int(n_pos_sampled / neg_sampling_ratio)
        n_neg_sampled = min(n_neg_sampled, n_neg)  # Don't exceed available
        
        self.n_pos_sampled = n_pos_sampled
        self.n_neg_sampled = n_neg_sampled
        self.n_samples = n_pos_sampled + n_neg_sampled
        
        print(f"BalancedSampler: pos={n_pos_sampled}, neg={n_neg_sampled}, total={self.n_samples}")
    
    def __iter__(self) -> Iterator[int]:
        # Sample with replacement for positives (oversampling)
        pos_sampled = np.random.choice(
            self.pos_indices,
            size=self.n_pos_sampled,
            replace=True,
        )
        
        # Sample without replacement for negatives (undersampling)
        neg_sampled = np.random.choice(
            self.neg_indices,
            size=self.n_neg_sampled,
            replace=False,
        )
        
        # Combine
        indices = np.concatenate([pos_sampled, neg_sampled])
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        return self.n_samples


class PatientAwareSampler(Sampler):
    """
    Patient-aware sampler that groups samples by patient.
    
    Ensures samples from the same patient appear together in batches,
    which can help with patient-level normalization features.
    
    Args:
        dataset: Dataset with patient_id information
        batch_size: Batch size
        samples_per_patient: Number of samples per patient in each batch
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        samples_per_patient: int = 4,
        neg_sampling_ratio: float = 0.01,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_patient = samples_per_patient
        
        # Get patient IDs and targets
        df = dataset.df
        self.patient_ids = df['patient_id'].values
        self.targets = np.array(dataset.targets)
        
        # Group indices by patient
        self.patient_to_indices = {}
        for idx, pid in enumerate(self.patient_ids):
            if pid not in self.patient_to_indices:
                self.patient_to_indices[pid] = []
            self.patient_to_indices[pid].append(idx)
        
        # Separate patients by whether they have positive samples
        self.pos_patients = []
        self.neg_patients = []
        for pid, indices in self.patient_to_indices.items():
            if any(self.targets[i] == 1 for i in indices):
                self.pos_patients.append(pid)
            else:
                self.neg_patients.append(pid)
        
        # Calculate batches
        patients_per_batch = batch_size // samples_per_patient
        n_pos_patients = len(self.pos_patients)
        n_neg_patients = int(n_pos_patients / neg_sampling_ratio)
        n_neg_patients = min(n_neg_patients, len(self.neg_patients))
        
        self.n_batches = (n_pos_patients + n_neg_patients) // patients_per_batch
        self.n_samples = self.n_batches * batch_size
    
    def __iter__(self) -> Iterator[int]:
        # Sample patients
        pos_patients = np.random.choice(
            self.pos_patients,
            size=len(self.pos_patients),
            replace=False,
        )
        neg_patients = np.random.choice(
            self.neg_patients,
            size=min(len(self.neg_patients), len(self.pos_patients) * 10),
            replace=False,
        )
        
        all_patients = np.concatenate([pos_patients, neg_patients])
        np.random.shuffle(all_patients)
        
        # Generate indices
        indices = []
        for pid in all_patients:
            patient_indices = self.patient_to_indices[pid]
            # Sample from this patient
            if len(patient_indices) >= self.samples_per_patient:
                sampled = np.random.choice(
                    patient_indices,
                    size=self.samples_per_patient,
                    replace=False,
                )
            else:
                sampled = np.random.choice(
                    patient_indices,
                    size=self.samples_per_patient,
                    replace=True,
                )
            indices.extend(sampled.tolist())
        
        return iter(indices[:self.n_samples])
    
    def __len__(self) -> int:
        return self.n_samples


def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Get a weighted random sampler based on class frequencies.
    
    Args:
        dataset: Dataset with get_sample_weights method
        
    Returns:
        WeightedRandomSampler
    """
    weights = dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


class CurriculumSampler(Sampler):
    """
    Curriculum sampler with progressive hard-example mining.
    
    Phase 1 (warm-up): Uniform random sampling
    Phase 2 (curriculum): Gradually increases sampling weight 
                          for hard examples based on per-sample loss
    
    Usage:
        sampler = CurriculumSampler(dataset, total_epochs=30)
        # After each epoch:
        sampler.update_losses(per_sample_losses)  # (N,) tensor
        sampler.set_epoch(epoch)
    """
    
    def __init__(
        self,
        dataset,
        total_epochs: int = 30,
        warmup_fraction: float = 0.2,
        hard_fraction: float = 0.3,
        oversample_pos: float = 2.0,
    ):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_fraction)
        self.hard_fraction = hard_fraction
        self.oversample_pos = oversample_pos
        self.current_epoch = 0
        
        self.targets = np.array(dataset.targets)
        self.n_samples = len(self.targets)
        
        # Per-sample loss (updated by trainer)
        self.sample_losses = np.ones(self.n_samples, dtype=np.float32)
        
        self.pos_indices = np.where(self.targets == 1)[0]
        self.neg_indices = np.where(self.targets == 0)[0]
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    def update_losses(self, losses: np.ndarray):
        """Update per-sample losses for curriculum weighting."""
        self.sample_losses = np.array(losses, dtype=np.float32)
    
    def __iter__(self) -> Iterator[int]:
        if self.current_epoch < self.warmup_epochs:
            # Warm-up: balanced random sampling
            return self._uniform_iter()
        else:
            # Curriculum: bias toward hard examples
            return self._curriculum_iter()
    
    def _uniform_iter(self) -> Iterator[int]:
        # Oversample positives
        n_pos = int(len(self.pos_indices) * self.oversample_pos)
        pos = np.random.choice(self.pos_indices, size=n_pos, replace=True)
        neg = np.random.choice(self.neg_indices, size=len(self.neg_indices), replace=False)
        indices = np.concatenate([pos, neg])
        np.random.shuffle(indices)
        return iter(indices.tolist())
    
    def _curriculum_iter(self) -> Iterator[int]:
        # Progress from warmup to end: 0→1
        progress = (self.current_epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        progress = min(progress, 1.0)
        
        # Sampling weights: mix uniform + loss-proportional
        uniform_weight = 1.0 - progress * self.hard_fraction
        loss_weight = progress * self.hard_fraction
        
        weights = uniform_weight * np.ones_like(self.sample_losses)
        
        # Normalize losses to [0, 1] range
        loss_min = self.sample_losses.min()
        loss_max = self.sample_losses.max()
        if loss_max > loss_min:
            norm_losses = (self.sample_losses - loss_min) / (loss_max - loss_min)
        else:
            norm_losses = np.ones_like(self.sample_losses)
        
        weights += loss_weight * norm_losses
        
        # Extra boost for positive samples
        weights[self.pos_indices] *= self.oversample_pos
        
        # Normalize to probabilities
        weights /= weights.sum()
        
        indices = np.random.choice(
            self.n_samples,
            size=self.n_samples,
            replace=True,
            p=weights,
        )
        return iter(indices.tolist())
    
    def __len__(self) -> int:
        if self.current_epoch < self.warmup_epochs:
            return int(len(self.pos_indices) * self.oversample_pos) + len(self.neg_indices)
        return self.n_samples
