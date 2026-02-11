"""
MambaDerm Training Script

Training infrastructure with:
- Early stopping with patience
- MixUp/CutMix augmentation with hard-target routing for AUCMaxLoss
- Multi-objective loss (Focal + AUC + Label Smoothing)
- Learning rate warmup + cosine decay with layer-wise LR
- Gradient checkpointing support
- Weights & Biases logging
- HDF5-safe multi-worker data loading
- Proper logging and checkpointing
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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config
from models import MambaDerm
from data import ISICDataset, get_train_transforms, get_val_transforms
from data.dataset import hdf5_worker_init_fn
from data.sampler import BalancedSampler
from data.augmentations import MixUpCutMix, get_mixup_cutmix
from utils import compute_pauc, pAUCMetric, get_cosine_schedule_with_warmup
from utils.losses import MultiObjectiveLoss, FocalLoss


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'max' for metrics like pAUC, 'min' for loss
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = 'max',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def get_args():
    parser = argparse.ArgumentParser(description="MambaDerm Training")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='../isic-2024-challenge')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_folds', type=int, default=5)
    
    # Model architecture
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6])
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--drop_path_rate', type=float, default=0.2)
    parser.add_argument('--use_multi_scale', action='store_true', default=True)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.8,
                       help='Layer-wise learning rate decay factor')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=10)
    
    # Augmentation
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--mixup_prob', type=float, default=0.5)
    
    # Loss
    parser.add_argument('--loss_type', type=str, default='multi',
                       choices=['focal', 'multi'])
    
    # Sampling
    parser.add_argument('--neg_sampling_ratio', type=float, default=0.01)
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mambaderm')
    
    return parser.parse_args()


def create_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_transforms = get_train_transforms(args.img_size)
    val_transforms = get_val_transforms(args.img_size)
    
    train_dataset = ISICDataset(
        data_dir=args.data_dir,
        split='train',
        fold=args.fold,
        n_folds=args.n_folds,
        transform=train_transforms,
        quick_test=args.quick_test,
        quick_test_samples=1000,
    )
    
    norm_stats = train_dataset.get_norm_stats()
    
    val_dataset = ISICDataset(
        data_dir=args.data_dir,
        split='val',
        fold=args.fold,
        n_folds=args.n_folds,
        transform=val_transforms,
        quick_test=args.quick_test,
        quick_test_samples=500,
        norm_stats=norm_stats,
    )
    
    train_sampler = BalancedSampler(
        train_dataset,
        neg_sampling_ratio=args.neg_sampling_ratio,
        oversample_ratio=2.0,
        shuffle=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=hdf5_worker_init_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=hdf5_worker_init_fn,
    )
    
    return train_loader, val_loader


def create_model(args) -> nn.Module:
    """Create MambaDerm model."""
    model = MambaDerm(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        depths=args.depths,
        d_state=args.d_state,
        num_classes=1,
        dropout=0.15,
        drop_path_rate=args.drop_path_rate,
        use_tabular=True,
        use_multi_scale=args.use_multi_scale,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MambaDerm parameters: {n_params / 1e6:.2f}M")
    
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
    mixup_fn: Optional[MixUpCutMix] = None,
) -> Dict[str, float]:
    """Train for one epoch with MixUp/CutMix and hard-target routing."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    optimizer.zero_grad()
    
    for batch_idx, (images, tabular, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Preserve hard targets for AUCMaxLoss before MixUp
        hard_targets = targets.clone()
        
        # Apply MixUp/CutMix (produces soft targets)
        if mixup_fn is not None and model.training:
            images, targets, tabular, _ = mixup_fn(images, targets, tabular)
        
        # Mixed precision forward
        with autocast('cuda', enabled=args.use_amp):
            outputs = model(images, tabular)
            # Pass both soft and hard targets to loss
            if hasattr(criterion, 'auc'):
                loss = criterion(outputs.squeeze(), targets, hard_targets=hard_targets)
            else:
                loss = criterion(outputs.squeeze(), targets)
            loss = loss / args.accumulation_steps
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            else:
                scaler.update()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.accumulation_steps
        num_batches += 1
        
        if batch_idx % args.log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{current_lr:.2e}",
            })
    
    return {
        'train_loss': total_loss / num_batches,
        'lr': optimizer.param_groups[0]['lr'],
    }


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
        
        with autocast('cuda', enabled=args.use_amp):
            outputs = model(images, tabular)
            loss = criterion(outputs.squeeze(), targets)
        
        total_loss += loss.item()
        num_batches += 1
        
        probs = torch.sigmoid(outputs.squeeze())
        pauc_metric.update(targets.cpu(), probs.cpu())
        
        pbar.set_postfix({'loss': f"{total_loss / num_batches:.4f}"})
    
    pauc = pauc_metric.compute()
    
    return {
        'val_loss': total_loss / num_batches,
        'val_pauc': pauc,
    }


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
    
    # Save latest
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save best
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"  ✓ Saved best model with pAUC={metrics['val_pauc']:.4f}")


def get_layer_wise_lr_groups(model, base_lr: float, decay: float = 0.8):
    """
    Create parameter groups with layer-wise learning rate decay.
    
    Deeper (later) stages get higher LR; earlier stages get decayed LR.
    Tabular encoder and classifier heads use full base_lr.
    
    Args:
        model: MambaDerm model
        base_lr: Base learning rate for deepest layers
        decay: Multiplicative decay factor per stage
        
    Returns:
        List of parameter groups for optimizer
    """
    groups = []
    seen_params = set()
    
    # Backbone stages: deeper stages get higher LR
    num_stages = len(model.backbone.stages)
    for i, stage in enumerate(reversed(model.backbone.stages)):
        lr = base_lr * (decay ** i)
        params = [p for p in stage.parameters() if p.requires_grad]
        seen_params.update(id(p) for p in params)
        if params:
            groups.append({'params': params, 'lr': lr, 'name': f'stage_{num_stages - 1 - i}'})
    
    # Patch embed: lowest LR
    patch_params = [p for p in model.backbone.patch_embed.parameters() if p.requires_grad]
    seen_params.update(id(p) for p in patch_params)
    if patch_params:
        groups.append({'params': patch_params, 'lr': base_lr * (decay ** num_stages), 'name': 'patch_embed'})
    
    # Position embeddings and backbone norm
    backbone_other = [p for n, p in model.backbone.named_parameters() 
                      if p.requires_grad and id(p) not in seen_params]
    seen_params.update(id(p) for p in backbone_other)
    if backbone_other:
        groups.append({'params': backbone_other, 'lr': base_lr * decay, 'name': 'backbone_other'})
    
    # All non-backbone params (tabular encoder, fusion, classifier) at base_lr
    other_params = [p for p in model.parameters() 
                    if p.requires_grad and id(p) not in seen_params]
    if other_params:
        groups.append({'params': other_params, 'lr': base_lr, 'name': 'head'})
    
    return groups


def main():
    args = get_args()
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"MambaDerm - Research-Grade Training")
    print("=" * 60)
    
    checkpoint_dir = Path(args.checkpoint_dir) / f"fold_{args.fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    print("\nCreating MambaDerm model...")
    model = create_model(args)
    model = model.to(device)
    
    # Loss
    if args.loss_type == 'multi':
        criterion = MultiObjectiveLoss(
            focal_weight=0.5,
            auc_weight=0.3,
            smooth_weight=0.2,
        )
        print("Using MultiObjectiveLoss (Focal + AUC + Label Smoothing)")
    else:
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        print("Using FocalLoss")
    
    # Optimizer with layer-wise learning rate decay
    param_groups = get_layer_wise_lr_groups(model, base_lr=args.lr, decay=args.lr_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    for g in param_groups:
        print(f"  LR group '{g.get('name', '?')}': lr={g['lr']:.2e}, params={len(g['params'])}")
    
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
    scaler = GradScaler('cuda', enabled=args.use_amp)
    
    # MixUp/CutMix
    mixup_fn = None
    if args.use_mixup:
        mixup_fn = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
        )
        print(f"Using MixUp/CutMix (α={args.mixup_alpha}/{args.cutmix_alpha}, p={args.mixup_prob})")
    
    # Early stopping
    early_stopper = EarlyStopping(patience=args.patience, mode='max')
    
    # Weights & Biases logging
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"fold{args.fold}_seed{args.seed}",
                tags=[f"fold{args.fold}", f"embed{args.embed_dim}", 
                      f"depths{'_'.join(map(str, args.depths))}"],
            )
            print(f"W&B run: {wandb_run.url}")
        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")
            args.use_wandb = False
    
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
            scaler, scheduler, device, args, epoch, mixup_fn
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, args, epoch
        )
        
        metrics = {**train_metrics, **val_metrics}
        
        # Print
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {metrics['train_loss']:.4f}")
        print(f"  Val Loss:   {metrics['val_loss']:.4f}")
        print(f"  Val pAUC:   {metrics['val_pauc']:.4f}")
        print(f"  LR:         {metrics['lr']:.2e}")
        
        # Check best
        is_best = metrics['val_pauc'] > best_pauc
        if is_best:
            best_pauc = metrics['val_pauc']
            best_epoch = epoch + 1
        
        # Save
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            metrics, checkpoint_dir, is_best
        )
        
        print(f"  Best pAUC: {best_pauc:.4f} (Epoch {best_epoch})")
        
        # W&B logging
        if args.use_wandb and wandb_run is not None:
            import wandb
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': metrics['train_loss'],
                'train/lr': metrics['lr'],
                'val/loss': metrics['val_loss'],
                'val/pAUC': metrics['val_pauc'],
                'val/best_pAUC': best_pauc,
            })
        
        # Early stopping
        if args.early_stopping and early_stopper(metrics['val_pauc']):
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"  No improvement for {args.patience} epochs")
            break
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best pAUC: {best_pauc:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print('='*60)
    
    # Finish W&B
    if args.use_wandb and wandb_run is not None:
        import wandb
        wandb.log({'best_pAUC': best_pauc, 'best_epoch': best_epoch})
        wandb.finish()
    
    return best_pauc


if __name__ == "__main__":
    main()
