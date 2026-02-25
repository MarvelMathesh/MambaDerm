"""
Layer-wise Learning Rate Groups

Creates parameter groups with decaying learning rates for earlier layers,
allowing the backbone to fine-tune more slowly while heads train at full LR.
"""

import torch.nn as nn
from typing import List, Dict


def get_layer_wise_lr_groups(
    model: nn.Module,
    base_lr: float,
    decay: float = 0.8,
) -> List[Dict]:
    """
    Create parameter groups with layer-wise learning rate decay.
    
    Deeper (later) stages get higher LR; earlier stages get decayed LR.
    Tabular encoder and classifier heads use full base_lr.
    
    Args:
        model: MambaDerm model with a .backbone attribute
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
            groups.append({
                'params': params,
                'lr': lr,
                'name': f'stage_{num_stages - 1 - i}',
            })
    
    # Patch embed: lowest LR
    patch_params = [
        p for p in model.backbone.patch_embed.parameters()
        if p.requires_grad
    ]
    seen_params.update(id(p) for p in patch_params)
    if patch_params:
        groups.append({
            'params': patch_params,
            'lr': base_lr * (decay ** num_stages),
            'name': 'patch_embed',
        })
    
    # Position embeddings and backbone norm
    backbone_other = [
        p for n, p in model.backbone.named_parameters()
        if p.requires_grad and id(p) not in seen_params
    ]
    seen_params.update(id(p) for p in backbone_other)
    if backbone_other:
        groups.append({
            'params': backbone_other,
            'lr': base_lr * decay,
            'name': 'backbone_other',
        })
    
    # All non-backbone params (tabular encoder, fusion, classifier) at base_lr
    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in seen_params
    ]
    if other_params:
        groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'head',
        })
    
    return groups
