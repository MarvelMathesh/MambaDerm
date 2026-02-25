"""
HAM10000 Training Script (Golden Pipeline)

7-class skin cancer classification with:
- Multi-class Focal Loss with class weighting
- CutMix augmentation (clinically appropriate)
- Progressive oversampling for minority classes
- Stochastic depth (drop_path) + gated bidirectional fusion
- Layer-wise learning rate decay
- Early stopping with patience
- Balanced accuracy & Macro F1 evaluation
- Optional W&B logging
"""

import argparse
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models import MambaDerm
from data import HAM10000Dataset, get_train_transforms, get_val_transforms
from data.ham10000_dataset import HAM10000_CLASS_NAMES
from data.augmentations import CutMix
from utils.losses import MultiClassFocalLoss, MultiClassMultiObjectiveLoss
from utils.metrics import MultiClassMetrics
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.lr_groups import get_layer_wise_lr_groups


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping based on balanced accuracy."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class CutMixMultiClass:
    """CutMix for multi-class with soft labels."""
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 7):
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.
        
        Args:
            images: Batch of images (N, C, H, W)
            labels: Class indices (N,)
            
        Returns:
            mixed_images: Augmented images
            mixed_labels: Soft label distribution (N, num_classes)
            lam: Mixing coefficient
        """
        if random.random() > 0.5:  # 50% probability
            # Convert labels to one-hot
            one_hot = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            return images, one_hot, 1.0
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size, device=images.device)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get bounding box
        W, H = images.size(3), images.size(2)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Clip bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Mix images
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Create soft labels
        one_hot_a = torch.zeros(batch_size, self.num_classes, device=labels.device)
        one_hot_a.scatter_(1, labels.unsqueeze(1), 1)
        one_hot_b = torch.zeros(batch_size, self.num_classes, device=labels.device)
        one_hot_b.scatter_(1, labels[indices].unsqueeze(1), 1)
        
        mixed_labels = lam * one_hot_a + (1 - lam) * one_hot_b
        
        return mixed_images, mixed_labels, lam


def soft_cross_entropy(pred: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy with soft targets (for CutMix)."""
    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def get_args():
    parser = argparse.ArgumentParser(description="HAM10000 Training")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./skin-cancer-mnist-ham10000')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_folds', type=int, default=5)
    
    # Model
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--embed_dim', type=int, default=96)
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6])
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--drop_path_rate', type=float, default=0.2)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    
    # Augmentation
    parser.add_argument('--use_cutmix', action='store_true', default=True)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    
    # Loss
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=10)
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_ham10000')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)
    
    # W&B
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='mambaderm')
    
    return parser.parse_args()


def create_dataloaders(args) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Create dataloaders with class-balanced sampling."""
    train_transforms = get_train_transforms(args.img_size)
    val_transforms = get_val_transforms(args.img_size)
    
    train_dataset = HAM10000Dataset(
        data_dir=args.data_dir,
        split='train',
        fold=args.fold,
        n_folds=args.n_folds,
        transform=train_transforms,
        quick_test=args.quick_test,
        quick_test_samples=500,
    )
    
    val_dataset = HAM10000Dataset(
        data_dir=args.data_dir,
        split='val',
        fold=args.fold,
        n_folds=args.n_folds,
        transform=val_transforms,
        quick_test=args.quick_test,
        quick_test_samples=200,
        norm_stats=train_dataset.get_norm_stats(),
    )
    
    # Class-balanced sampling
    sampler = WeightedRandomSampler(
        weights=train_dataset.get_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    class_weights = train_dataset.get_class_weights()
    
    return train_loader, val_loader, class_weights


def create_model(args) -> nn.Module:
    """Create MambaDerm model for 7-class classification."""
    # HAM10000 tabular features are already one-hot encoded:
    # - age (1) + sex (3) + localization (15) = 19 numerical features
    # - Pass all as numerical, no categorical embeddings needed
    model = MambaDerm(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        depths=args.depths,
        d_state=args.d_state,
        num_numerical_features=19,  # All pre-encoded features
        num_categorical_features=0,  # No categorical embeddings
        num_classes=7,  # 7 skin cancer classes
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        use_tabular=True,
        use_multi_scale=True,
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
    cutmix_fn: Optional[CutMixMultiClass] = None,
) -> Dict[str, float]:
    """Train for one epoch with CutMix."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    optimizer.zero_grad()
    
    for batch_idx, (images, tabular, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply CutMix
        use_soft_labels = False
        if cutmix_fn is not None:
            images, soft_labels, lam = cutmix_fn(images, labels)
            if lam < 1.0:  # CutMix was applied
                use_soft_labels = True
        
        # Forward
        with autocast('cuda', enabled=args.use_amp):
            outputs = model(images, tabular)
            
            if use_soft_labels:
                # Unified: use soft cross-entropy when CutMix was applied
                loss = soft_cross_entropy(outputs, soft_labels)
            else:
                # Use configured criterion for hard labels
                loss = criterion(outputs, labels)
            
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
    """Validate with multi-class metrics."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    metrics = MultiClassMetrics(num_classes=7, class_names=HAM10000_CLASS_NAMES)
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
    
    for images, tabular, labels in pbar:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast('cuda', enabled=args.use_amp):
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update metrics
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        metrics.update(labels.cpu(), y_pred=preds.cpu(), y_prob=probs.cpu())
        
        pbar.set_postfix({'loss': f"{total_loss / num_batches:.4f}"})
    
    # Compute metrics
    metric_results = metrics.compute()
    
    return {
        'val_loss': total_loss / num_batches,
        'val_balanced_acc': metric_results['balanced_accuracy'],
        'val_macro_f1': metric_results['macro_f1'],
        'per_class_acc': metric_results['per_class_accuracy'],
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    checkpoint_dir: Path,
    is_best: bool = False,
):
    """Save checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': {k: v.tolist() if hasattr(v, 'tolist') else v 
                    for k, v in metrics.items()},
    }
    
    # Save latest
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    
    # Save best
    if is_best:
        torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        print(f"  ✓ Saved best model with balanced_acc={metrics['val_balanced_acc']:.4f}")


# get_layer_wise_lr_groups is imported from utils.lr_groups
# (previously duplicated here — now shared across training scripts)


def main():
    args = get_args()
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    print("MambaDerm HAM10000 Training (Golden Pipeline)")
    print("7-Class Skin Cancer Classification")
    print("=" * 60)
    
    checkpoint_dir = Path(args.checkpoint_dir) / f"fold_{args.fold}"
    
    # Data
    print("\nCreating dataloaders...")
    train_loader, val_loader, class_weights = create_dataloaders(args)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Class weights: {class_weights.numpy().round(2)}")
    
    # Model
    print("\nCreating model...")
    model = create_model(args)
    model = model.to(device)
    
    # Loss
    class_weights = class_weights.to(device)
    criterion = MultiClassFocalLoss(
        num_classes=7,
        alpha=class_weights,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    print(f"Using MultiClassFocalLoss (γ={args.focal_gamma}, smoothing={args.label_smoothing})")
    
    # Optimizer with layer-wise LR decay
    param_groups = get_layer_wise_lr_groups(model, base_lr=args.lr, decay=args.lr_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
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
    scaler = GradScaler('cuda', enabled=args.use_amp)
    
    # CutMix
    cutmix_fn = None
    if args.use_cutmix:
        cutmix_fn = CutMixMultiClass(alpha=args.cutmix_alpha, num_classes=7)
        print(f"Using CutMix (α={args.cutmix_alpha})")
    
    # Early stopping
    early_stopper = EarlyStopping(patience=args.patience)
    
    # W&B
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"ham10000_fold{args.fold}",
                tags=["ham10000", f"fold{args.fold}"],
            )
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
            args.use_wandb = False
    
    # Training loop
    print("\nStarting training...")
    best_balanced_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print('='*60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, scheduler, device, args, epoch, cutmix_fn
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, args, epoch
        )
        
        metrics = {**train_metrics, **val_metrics}
        
        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss:      {metrics['train_loss']:.4f}")
        print(f"  Val Loss:        {metrics['val_loss']:.4f}")
        print(f"  Balanced Acc:    {metrics['val_balanced_acc']:.4f}")
        print(f"  Macro F1:        {metrics['val_macro_f1']:.4f}")
        print(f"  Per-class Acc:   {metrics['per_class_acc'].round(3)}")
        print(f"  LR:              {metrics['lr']:.2e}")
        
        # Check best
        is_best = metrics['val_balanced_acc'] > best_balanced_acc
        if is_best:
            best_balanced_acc = metrics['val_balanced_acc']
            best_epoch = epoch + 1
        
        # Save
        save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_dir, is_best)
        print(f"  Best Balanced Acc: {best_balanced_acc:.4f} (Epoch {best_epoch})")
        
        # W&B log
        if args.use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': metrics['train_loss'],
                'val/loss': metrics['val_loss'],
                'val/balanced_acc': metrics['val_balanced_acc'],
                'val/macro_f1': metrics['val_macro_f1'],
                'val/best_balanced_acc': best_balanced_acc,
                'lr': metrics['lr'],
            })
        
        # Early stopping
        if args.early_stopping and early_stopper(metrics['val_balanced_acc']):
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            print(f"  No improvement for {args.patience} epochs")
            break
    
    # W&B finish
    if args.use_wandb:
        import wandb
        wandb.log({'best_balanced_acc': best_balanced_acc, 'best_epoch': best_epoch})
        wandb.finish()
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Balanced Accuracy: {best_balanced_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print('='*60)
    
    return best_balanced_acc


if __name__ == "__main__":
    main()
