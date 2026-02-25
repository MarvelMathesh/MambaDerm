"""
MambaDerm Cross-Validation Training Script

5-fold stratified group cross-validation
with proper aggregation and statistical reporting.

Usage:
    python train_cv.py --data_dir ../isic-2024-challenge --n_folds 5
    python train_cv.py --data_dir ../isic-2024-challenge --n_folds 5 --use_wandb
    
Output:
    - Per-fold checkpoints in checkpoints/fold_{i}/
    - Aggregated CV results with mean ± std
    - CSV results file for reproducibility
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Import train.py's main training function
from train import (
    set_seed,
    create_dataloaders,
    create_model,
    train_epoch,
    validate,
    save_checkpoint,
    EarlyStopping,
    get_layer_wise_lr_groups,
)

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_cosine_schedule_with_warmup, pAUCMetric
from utils.losses import MultiObjectiveLoss, FocalLoss
from data.augmentations import get_mixup_cutmix


def get_cv_args():
    parser = argparse.ArgumentParser(description="MambaDerm Cross-Validation Training")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='../isic-2024-challenge')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                       help='Specific folds to run (default: all)')
    
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
    parser.add_argument('--lr_decay', type=float, default=0.8)
    
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
    parser.add_argument('--results_dir', type=str, default='results')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Multiple seeds for statistical rigor (default: single seed)')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--quick_test', action='store_true')
    parser.add_argument('--log_interval', type=int, default=100)
    
    # W&B
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='mambaderm')
    
    return parser.parse_args()


def train_single_fold(args, fold: int) -> dict:
    """
    Train a single fold and return metrics.
    
    Args:
        args: Training arguments
        fold: Fold index (0-based)
        
    Returns:
        dict with fold results including best_pauc, best_epoch, etc.
    """
    args.fold = fold
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(args.checkpoint_dir) / f"fold_{fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{args.n_folds}")
    print(f"{'='*60}")
    
    # Data
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = create_model(args)
    model = model.to(device)
    
    # Loss
    if args.loss_type == 'multi':
        criterion = MultiObjectiveLoss(
            focal_weight=0.5,
            auc_weight=0.3,
            smooth_weight=0.2,
        )
    else:
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    # Optimizer with layer-wise LR
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
    
    # MixUp/CutMix
    mixup_fn = None
    if args.use_mixup:
        mixup_fn = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
        )
    
    # Early stopping
    early_stopper = EarlyStopping(patience=args.patience, mode='max')
    
    # W&B
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"fold{fold}_seed{args.seed}",
                group=f"cv_seed{args.seed}",
                tags=[f"fold{fold}", "cv"],
                reinit=True,
            )
        except ImportError:
            pass
    
    # Training loop
    best_pauc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
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
        
        print(f"  Fold {fold+1} Epoch {epoch+1}: "
              f"train_loss={metrics['train_loss']:.4f}, "
              f"val_loss={metrics['val_loss']:.4f}, "
              f"val_pAUC={metrics['val_pauc']:.4f}")
        
        # Track best
        is_best = metrics['val_pauc'] > best_pauc
        if is_best:
            best_pauc = metrics['val_pauc']
            best_epoch = epoch + 1
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            metrics, checkpoint_dir, is_best
        )
        
        # W&B log
        if wandb_run is not None:
            import wandb
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': metrics['train_loss'],
                'val/loss': metrics['val_loss'],
                'val/pAUC': metrics['val_pauc'],
                'val/best_pAUC': best_pauc,
            })
        
        # Early stopping
        if args.early_stopping and early_stopper(metrics['val_pauc']):
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Finish W&B run for this fold
    if wandb_run is not None:
        import wandb
        wandb.log({'best_pAUC': best_pauc, 'best_epoch': best_epoch})
        wandb.finish()
    
    return {
        'fold': fold,
        'best_pauc': best_pauc,
        'best_epoch': best_epoch,
        'final_train_loss': train_metrics['train_loss'],
        'final_val_loss': val_metrics['val_loss'],
    }


def main():
    args = get_cv_args()
    
    # Determine folds to run
    folds = args.folds if args.folds is not None else list(range(args.n_folds))
    seeds = args.seeds if args.seeds is not None else [args.seed]
    
    print("=" * 70)
    print("MambaDerm Cross-Validation Training")
    print("=" * 70)
    print(f"Folds: {folds}")
    print(f"Seeds: {seeds}")
    print(f"Architecture: embed_dim={args.embed_dim}, depths={args.depths}")
    print(f"Training: epochs={args.epochs}, lr={args.lr}, wd={args.weight_decay}")
    print(f"Loss: {args.loss_type}")
    print("=" * 70)
    
    # Results storage
    all_results = []
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED: {seed}")
        print(f"{'#'*70}")
        
        seed_results = []
        
        for fold in folds:
            # Use unique seed per (seed, fold) combination to avoid collisions
            # e.g. seed=42, fold=1 must differ from seed=43, fold=0
            args.seed = seed * args.n_folds + fold
            set_seed(args.seed)
            
            fold_result = train_single_fold(args, fold)
            fold_result['seed'] = seed
            seed_results.append(fold_result)
            all_results.append(fold_result)
        
        # Report seed-level results
        seed_paucs = [r['best_pauc'] for r in seed_results]
        print(f"\n{'='*60}")
        print(f"Seed {seed} Results:")
        for r in seed_results:
            print(f"  Fold {r['fold']}: pAUC={r['best_pauc']:.4f} (epoch {r['best_epoch']})")
        print(f"  Mean pAUC: {np.mean(seed_paucs):.4f} ± {np.std(seed_paucs):.4f}")
        print(f"{'='*60}")
    
    # Final aggregated results
    total_time = time.time() - start_time
    all_paucs = [r['best_pauc'] for r in all_results]
    
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"\nAll results ({len(all_results)} runs):")
    for r in all_results:
        print(f"  Seed {r['seed']} Fold {r['fold']}: "
              f"pAUC={r['best_pauc']:.4f} (epoch {r['best_epoch']})")
    
    print(f"\n{'─'*40}")
    print(f"pAUC@80%TPR = {np.mean(all_paucs):.4f} ± {np.std(all_paucs):.4f}")
    print(f"Min: {np.min(all_paucs):.4f}, Max: {np.max(all_paucs):.4f}")
    print(f"Total training time: {total_time/3600:.1f}h")
    print(f"{'─'*40}")
    
    # Save results to CSV
    csv_path = results_dir / f"cv_results_{int(time.time())}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to: {csv_path}")
    
    # Save summary
    summary_path = results_dir / f"cv_summary_{int(time.time())}.txt"
    with open(summary_path, 'w') as f:
        f.write("MambaDerm Cross-Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Architecture: embed_dim={args.embed_dim}, depths={args.depths}\n")
        f.write(f"Training: epochs={args.epochs}, lr={args.lr}, wd={args.weight_decay}\n")
        f.write(f"Loss: {args.loss_type}\n")
        f.write(f"Folds: {folds}, Seeds: {seeds}\n\n")
        f.write(f"pAUC@80%TPR = {np.mean(all_paucs):.4f} ± {np.std(all_paucs):.4f}\n")
        f.write(f"Min: {np.min(all_paucs):.4f}, Max: {np.max(all_paucs):.4f}\n")
        f.write(f"Total time: {total_time/3600:.1f}h\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
