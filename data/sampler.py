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
