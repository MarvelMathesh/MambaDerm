"""
MambaDerm: Hybrid CNN-Mamba for Efficient Skin Lesion Classification

A novel architecture combining:
- ConvNeXt-style CNN stem for local feature extraction
- Vision Mamba (VMamba) backbone for O(n) global context
- Tabular encoder for clinical metadata
- Local-Global gating for adaptive feature fusion

Reference: Proposed architecture from ISIC 2024 analysis
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .cnn_stem import ConvNeXtStem, PatchEmbed2D
from .mamba_block import VMambaLayer, VisionMambaBlock
from .tabular_encoder import TabularEncoder, TabularTokenEncoder
from .local_global_gate import LocalGlobalGate, MultiModalFusion


class MambaDerm(nn.Module):
    """
    MambaDerm: Hybrid CNN-Mamba for Skin Lesion Classification
    
    Architecture overview:
    1. CNN Stem: Patchify + local feature extraction (ConvNeXt blocks)
    2. VMamba Backbone: Stack of bidirectional Mamba layers
    3. Tabular Encoder: Clinical metadata encoding
    4. Fusion: Local-global gating + cross-modal attention
    5. Classifier: Global pool + MLP head
    
    Args:
        img_size: Input image size (default: 224)
        in_channels: Number of input channels (default: 3)
        stem_channels: CNN stem hidden dimension (default: 96)
        d_model: VMamba model dimension (default: 192)
        n_mamba_layers: Number of VMamba layers (default: 4)
        d_state: SSM state dimension (default: 16)
        d_conv: Local conv width in SSM (default: 4)
        expand_factor: SSM expansion factor (default: 2)
        num_numerical_features: Number of numerical tabular features (default: 34)
        num_categorical_features: Number of categorical tabular features (default: 6)
        num_classes: Number of output classes (default: 1 for binary)
        dropout: Dropout rate (default: 0.1)
        use_tabular: Whether to use tabular features (default: True)
        use_cross_attention: Use cross-attention for tabular fusion (default: True)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        stem_channels: int = 96,
        d_model: int = 192,
        n_mamba_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        num_numerical_features: int = 34,
        num_categorical_features: int = 6,
        num_classes: int = 1,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        use_tabular: bool = True,
        use_cross_attention: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.d_model = d_model
        self.use_tabular = use_tabular
        self.num_classes = num_classes
        self.num_numerical_features = num_numerical_features  # Store for forward pass
        self.gradient_checkpointing = gradient_checkpointing
        
        # 1. CNN Stem for local feature extraction
        self.stem = ConvNeXtStem(
            in_channels=in_channels,
            stem_channels=stem_channels,
            out_channels=d_model,
            kernel_size=4,
            stride=4,
            num_blocks=2,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        
        # Calculate number of patches
        self.patch_size = 4
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )
        
        # 2. VMamba Backbone (stacked Mamba layers)
        self.mamba_layers = nn.ModuleList([
            VMambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand_factor,
                mlp_ratio=4.0,
                dropout=dropout,
                bidirectional=True,
            )
            for _ in range(n_mamba_layers)
        ])
        
        # Norm after Mamba layers
        self.norm = nn.LayerNorm(d_model)
        
        # 3. Tabular Encoder (optional)
        if use_tabular:
            self.tabular_encoder = TabularEncoder(
                num_numerical=num_numerical_features,
                num_categorical=num_categorical_features,
                hidden_dim=128,
                output_dim=d_model,
                dropout=dropout,
            )
        
        # 4. Local-Global Fusion
        self.fusion = MultiModalFusion(
            dim=d_model,
            num_heads=8,
            dropout=dropout,
            use_cross_attention=use_cross_attention and use_tabular,
        )
        
        # 5. Classification Head
        classifier_dim = d_model
        if use_tabular:
            classifier_dim = d_model + d_model  # Image + Tabular
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights with truncated normal distribution."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward_features(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features before classification head.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            tabular: Tabular features of shape (B, num_features)
            
        Returns:
            image_features: Pooled image features (B, d_model)
            tabular_features: Encoded tabular features (B, d_model) or None
        """
        B = x.shape[0]
        
        # CNN Stem: local feature extraction
        local_features, spatial_shape = self.stem(x)  # (B, L, D)
        
        # Add position embeddings
        local_features = local_features + self.pos_embed
        
        # VMamba: global context modeling
        global_features = local_features
        for mamba_layer in self.mamba_layers:
            if self.gradient_checkpointing and self.training:
                global_features = checkpoint(mamba_layer, global_features, use_reentrant=False)
            else:
                global_features = mamba_layer(global_features)
        
        global_features = self.norm(global_features)
        
        # Encode tabular features
        tabular_features = None
        if self.use_tabular and tabular is not None:
            # Split tabular into numerical and categorical (using configurable num_numerical)
            numerical = tabular[:, :self.num_numerical_features]
            categorical = tabular[:, self.num_numerical_features:].long() if tabular.shape[1] > self.num_numerical_features else None
            
            tabular_features = self.tabular_encoder(numerical, categorical)  # (B, d_model)
        
        # Local-Global fusion with optional tabular cross-attention
        fused_features = self.fusion(
            local_features,
            global_features,
            tabular_features,
        )
        
        # Global average pooling
        image_features = fused_features.mean(dim=1)  # (B, d_model)
        
        return image_features, tabular_features
    
    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            tabular: Tabular features of shape (B, num_features)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        # Extract features
        image_features, tabular_features = self.forward_features(x, tabular)
        
        # Combine image and tabular for classification
        if self.use_tabular and tabular_features is not None:
            combined = torch.cat([image_features, tabular_features], dim=-1)
        else:
            combined = image_features
        
        # Classification
        logits = self.classifier(combined)
        
        return logits


class MambaDermLite(nn.Module):
    """
    Lightweight version of MambaDerm for faster inference.
    
    Reduces parameters by:
    - Fewer Mamba layers (2 instead of 4)
    - Smaller hidden dimensions
    - No cross-attention fusion
    """
    
    def __init__(
        self,
        img_size: int = 224,
        num_numerical_features: int = 35,
        num_categorical_features: int = 6,
        num_classes: int = 1,
    ):
        super().__init__()
        
        # Lighter configuration
        self.model = MambaDerm(
            img_size=img_size,
            stem_channels=64,
            d_model=128,
            n_mamba_layers=2,
            d_state=8,
            expand_factor=2,
            num_numerical_features=num_numerical_features,
            num_categorical_features=num_categorical_features,
            num_classes=num_classes,
            dropout=0.1,
            use_cross_attention=False,
        )
    
    def forward(self, x, tabular=None):
        return self.model(x, tabular)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """Get model information including parameter count and structure."""
    total_params = count_parameters(model)
    
    # Count parameters by component
    component_params = {}
    for name, module in model.named_children():
        component_params[name] = count_parameters(module)
    
    return {
        'total_params': total_params,
        'total_params_millions': total_params / 1e6,
        'component_params': component_params,
    }


if __name__ == "__main__":
    # Test the model
    print("Testing MambaDerm...")
    
    # Create model
    model = MambaDerm(
        img_size=224,
        d_model=192,
        n_mamba_layers=4,
        num_classes=1,
    )
    
    # Print model info
    info = get_model_info(model)
    print(f"Total parameters: {info['total_params_millions']:.2f}M")
    print(f"Parameters by component:")
    for name, count in info['component_params'].items():
        print(f"  {name}: {count / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    tabular = torch.randn(batch_size, 40)  # 34 numerical + 6 categorical
    
    with torch.no_grad():
        output = model(x, tabular)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Tabular shape: {tabular.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze()}")
    
    print("\nMambaDerm test passed!")
