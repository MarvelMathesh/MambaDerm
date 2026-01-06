"""
MambaDerm Training Script

Production-ready training script with:
- Mixed precision (AMP) training
- Gradient checkpointing
- Cross-validation support
- pAUC@80% TPR evaluation
- Checkpointing and logging
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from models import MambaDerm
from data import ISICDataset, get_train_transforms, get_val_transforms
from data.sampler import BalancedSampler, get_weighted_sampler
from utils import compute_pauc, pAUCMetric, FocalLoss, get_cosine_schedule_with_warmup


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MambaDerm Training")
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                        default='../isic-2024-challenge',
                        help='Path to ISIC data directory')
    parser.add_argument('--fold', type=int, default=0, help='Cross-validation fold (0-4)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    
    # Model
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--d_model', type=int, default=192, help='Model dimension')
    parser.add_argument('--n_mamba_layers', type=int, default=4, help='Number of Mamba layers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clip norm')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    
    # Sampling
    parser.add_argument('--neg_sampling_ratio', type=float, default=0.01, 
                        help='Negative sampling ratio')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_top_k', type=int, default=3, help='Save top K models')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    
    return parser.parse_args()


def create_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    # Transforms
    train_transforms = get_train_transforms(args.img_size)
    val_transforms = get_val_transforms(args.img_size)
    
    # Create training dataset first to get normalization statistics
    train_dataset = ISICDataset(
        data_dir=args.data_dir,
        split='train',
        fold=args.fold,
        n_folds=args.n_folds,
        transform=train_transforms,
        quick_test=args.quick_test,
        quick_test_samples=1000,
    )
    
    # Get normalization stats from training set
    norm_stats = train_dataset.get_norm_stats()
    
    # Create validation dataset with training statistics to prevent data leakage
    val_dataset = ISICDataset(
        data_dir=args.data_dir,
        split='val',
        fold=args.fold,
        n_folds=args.n_folds,
        transform=val_transforms,
        quick_test=args.quick_test,
        quick_test_samples=500,
        norm_stats=norm_stats,  # Use training set statistics
    )
    
    # Sampler for balanced training
    train_sampler = BalancedSampler(
        train_dataset,
        neg_sampling_ratio=args.neg_sampling_ratio,
        oversample_ratio=2.0,
        shuffle=True,
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_model(args) -> nn.Module:
    """Create MambaDerm model."""
    model = MambaDerm(
        img_size=args.img_size,
        d_model=args.d_model,
        n_mamba_layers=args.n_mamba_layers,
        num_classes=1,
        dropout=0.1,
        use_tabular=True,
        use_cross_attention=True,
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    scheduler,
    device: torch.device,
    args,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    optimizer.zero_grad()
    
    for batch_idx, (images, tabular, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Mixed precision forward
        with autocast(enabled=args.use_amp):
            outputs = model(images, tabular)
            loss = criterion(outputs.squeeze(), targets)
            loss = loss / args.accumulation_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % args.accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Check for inf/nan gradients before stepping
            if torch.isfinite(grad_norm):
                # Optimizer step (scaler handles skip internally)
                scaler.step(optimizer)
                scaler.update()
                
                # Scheduler step only if optimizer actually stepped
                if scheduler is not None:
                    scheduler.step()
            else:
                # Skip this step due to bad gradients
                scaler.update()  # Still need to update scaler
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.accumulation_steps
        num_batches += 1
        
        # Update progress bar
        if batch_idx % args.log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{current_lr:.2e}",
            })
    
    metrics = {
        'train_loss': total_loss / num_batches,
        'lr': optimizer.param_groups[0]['lr'],
    }
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args,
    epoch: int,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    pauc_metric = pAUCMetric(min_tpr=0.80)
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
    
    for images, tabular, targets in pbar:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast(enabled=args.use_amp):
            outputs = model(images, tabular)
            loss = criterion(outputs.squeeze(), targets)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update pAUC metric
        probs = torch.sigmoid(outputs.squeeze())
        pauc_metric.update(targets.cpu(), probs.cpu())
        
        pbar.set_postfix({'loss': f"{total_loss / num_batches:.4f}"})
    
    pauc = pauc_metric.compute()
    
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_pauc': pauc,
    }
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"  Saved best model with pAUC={metrics['val_pauc']:.4f}")
    
    return checkpoint_path


def main():
    """Main training function."""
    args = get_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / f"fold_{args.fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(args)
    model = model.to(device)
    
    # Loss function
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * args.epochs // args.accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr / args.lr,
    )
    
    # Mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    print("\nStarting training...")
    best_pauc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print('='*60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, scheduler, device, args, epoch
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, args, epoch
        )
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        # Print metrics
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {metrics['train_loss']:.4f}")
        print(f"  Val Loss:   {metrics['val_loss']:.4f}")
        print(f"  Val pAUC:   {metrics['val_pauc']:.4f}")
        print(f"  LR:         {metrics['lr']:.2e}")
        
        # Check if best
        is_best = metrics['val_pauc'] > best_pauc
        if is_best:
            best_pauc = metrics['val_pauc']
            best_epoch = epoch + 1
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            metrics, checkpoint_dir, is_best
        )
        
        print(f"  Best pAUC: {best_pauc:.4f} (Epoch {best_epoch})")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best pAUC: {best_pauc:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
