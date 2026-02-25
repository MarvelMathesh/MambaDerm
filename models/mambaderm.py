"""
MambaDerm: Hybrid CNN-Mamba Architecture

A novel architecture featuring:
- Hierarchical 3-stage Mamba backbone with multi-scale patches
- Multi-scale patch embedding (4×4, 8×8, 16×16)
- Gated bidirectional SSM fusion
- Multi-head cross-modal attention with feature pyramid
- Stochastic depth regularization (properly wired)
- Dimension-safe local-global gating with projection
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .hierarchical_mamba import (
    HierarchicalMambaBackbone,
    MultiScalePatchEmbed,
)
from .tabular_encoder import TabularEncoder


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-modal attention for image-tabular fusion.
    
    Image features attend to tabular features for context-aware
    representation learning.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q from image, K/V from tabular
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
    
    def forward(
        self,
        query: torch.Tensor,      # Image features (B, L, D)
        key_value: torch.Tensor,  # Tabular features (B, 1, D) or (B, N, D)
    ) -> torch.Tensor:
        B, L, C = query.shape
        _, N, _ = key_value.shape
        
        # Normalize
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)
        
        # Project
        q = self.q_proj(query).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeaturePyramidFusion(nn.Module):
    """
    Feature Pyramid Network-style fusion of multi-stage features.
    
    Combines features from all hierarchical stages for rich
    multi-scale representation.
    """
    
    def __init__(self, dims: List[int], out_dim: int):
        super().__init__()
        
        # Lateral connections (1×1 conv equivalent)
        self.laterals = nn.ModuleList([
            nn.Linear(dim, out_dim) for dim in dims
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * len(dims), out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of (B, L_i, C_i) from each stage
            
        Returns:
            fused: (B, out_dim) global feature
        """
        # Project each stage
        projected = []
        for feat, lateral in zip(features, self.laterals):
            # Global average pool then project
            pooled = feat.mean(dim=1)  # B, C_i
            projected.append(lateral(pooled))
        
        # Concatenate and fuse
        concat = torch.cat(projected, dim=-1)
        fused = self.fusion(concat)
        
        return fused


class EnhancedMultiModalFusion(nn.Module):
    """
    Enhanced multimodal fusion with:
    - Local-global gating
    - Multi-head cross-attention
    - Feature pyramid integration
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        
        # Local-global gating
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1),
        )
        
        # Cross-modal attention
        if use_cross_attention:
            self.cross_attn = MultiHeadCrossAttention(
                dim=dim,
                num_heads=num_heads,
                attn_drop=dropout,
                proj_drop=dropout,
            )
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        local_feat: torch.Tensor,    # B, L, D (from early stage)
        global_feat: torch.Tensor,   # B, L, D (from final stage)
        tabular_feat: Optional[torch.Tensor] = None,  # B, D
    ) -> torch.Tensor:
        """Fuse local, global, and tabular features."""
        
        # Handle size mismatch by adaptive pooling
        if local_feat.shape[1] != global_feat.shape[1]:
            L_global = global_feat.shape[1]
            # Use adaptive average pooling over the sequence dimension
            # Rearrange to (B, D, L) for pooling, then back
            local_feat = local_feat.permute(0, 2, 1)  # B, D, L_local
            local_feat = F.adaptive_avg_pool1d(local_feat, L_global)  # B, D, L_global
            local_feat = local_feat.permute(0, 2, 1)  # B, L_global, D
        
        # Gated fusion of local and global
        combined = torch.cat([local_feat, global_feat], dim=-1)
        gates = self.gate(combined)  # B, L, 2
        
        fused = gates[..., 0:1] * local_feat + gates[..., 1:2] * global_feat
        
        # Cross-modal attention with tabular
        if self.use_cross_attention and tabular_feat is not None:
            if tabular_feat.dim() == 2:
                tabular_feat = tabular_feat.unsqueeze(1)  # B, 1, D
            
            cross_out = self.cross_attn(fused, tabular_feat)
            fused = fused + self.dropout(cross_out)
        
        fused = self.norm(fused)
        
        return fused


class MambaDerm(nn.Module):
    """
    MambaDerm: World-Class Hybrid CNN-Mamba Architecture
    
    Features:
    1. Hierarchical 3-stage backbone with multi-scale patches
    2. Feature pyramid fusion across stages
    3. Enhanced gated local-global fusion
    4. Multi-head cross-modal attention
    5. Stochastic depth regularization
    
    Args:
        img_size: Input image size (default: 224)
        in_channels: Number of input channels
        embed_dim: Base embedding dimension (default: 96)
        depths: Layers per stage (default: [2, 2, 6])
        num_numerical_features: Numerical tabular features
        num_categorical_features: Categorical tabular features
        num_classes: Output classes (1 for binary)
        dropout: Dropout rate
        drop_path_rate: Stochastic depth rate
        use_tabular: Whether to use tabular features
    """
    
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: Optional[List[int]] = None,
        d_state: int = 16,
        num_numerical_features: int = 33,
        num_categorical_features: int = 6,
        num_classes: int = 1,
        dropout: float = 0.15,
        drop_path_rate: float = 0.2,
        use_tabular: bool = True,
        use_multi_scale: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        # Handle mutable default argument
        if depths is None:
            depths = [2, 2, 6]
        
        self.use_tabular = use_tabular
        self.num_classes = num_classes
        self.num_numerical_features = num_numerical_features
        self.gradient_checkpointing = gradient_checkpointing
        
        # Compute dimensions for each stage
        # Note: dims are INPUT dims, output dims are different due to downsampling
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.dims = dims
        final_dim = dims[-1]
        
        # Compute OUTPUT dimensions for each stage (after downsampling)
        # Stage i outputs dims[i+1] if downsampled, else dims[i]
        stage_out_dims = []
        for i in range(len(depths)):
            if i < len(depths) - 1:  # Stages with downsample
                stage_out_dims.append(dims[i + 1])
            else:  # Last stage (no downsample)
                stage_out_dims.append(dims[i])
        self.stage_out_dims = stage_out_dims
        
        # Hierarchical backbone
        self.backbone = HierarchicalMambaBackbone(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            dims=dims,
            d_state=d_state,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            use_multi_scale_embed=use_multi_scale,
        )
        
        # Tabular encoder
        if use_tabular:
            self.tabular_encoder = TabularEncoder(
                num_numerical=num_numerical_features,
                num_categorical=num_categorical_features,
                hidden_dim=128,
                output_dim=final_dim,
                dropout=dropout,
            )
        
        # Feature pyramid fusion - use stage OUTPUT dims
        self.pyramid_fusion = FeaturePyramidFusion(stage_out_dims, final_dim)
        
        # Local feature projection (align second-to-last stage dim to final_dim)
        if len(depths) > 1 and stage_out_dims[-2] != final_dim:
            self.local_proj = nn.Sequential(
                nn.Linear(stage_out_dims[-2], final_dim),
                nn.LayerNorm(final_dim),
            )
        else:
            self.local_proj = nn.Identity()
        
        # Enhanced multimodal fusion
        self.fusion = EnhancedMultiModalFusion(
            dim=final_dim,
            num_heads=8,
            dropout=dropout,
            use_cross_attention=use_tabular,
        )
        
        # Classification head
        classifier_dim = final_dim
        if use_tabular:
            classifier_dim = final_dim + final_dim  # Image + Tabular
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, final_dim),
            nn.LayerNorm(final_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.LayerNorm(final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        # Skip SSM-specific modules — they use specialized initialization
        # (A_log, D, dt_proj) from the Mamba paper
        if hasattr(m, 'A_log') or hasattr(m, 'dt_proj'):
            return
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward_features(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features before classifier."""
        
        # Hierarchical backbone
        if self.gradient_checkpointing and self.training:
            final_feat, stage_features = checkpoint(
                self.backbone, x, use_reentrant=False
            )
        else:
            final_feat, stage_features = self.backbone(x)
        
        # Encode tabular
        tabular_feat = None
        if self.use_tabular and tabular is not None:
            numerical = tabular[:, :self.num_numerical_features]
            categorical: Optional[torch.Tensor] = None
            if tabular.shape[1] > self.num_numerical_features:
                categorical = tabular[:, self.num_numerical_features:].long()
            tabular_feat = self.tabular_encoder(numerical, categorical)
        
        # Feature pyramid fusion (global feature from all stages)
        pyramid_feat = self.pyramid_fusion(stage_features)
        
        # Use final features for fusion
        # Project local features to match final dimension
        if len(stage_features) > 1:
            local_feat = self.local_proj(stage_features[-2])  # Now (B, L, final_dim)
        else:
            local_feat = final_feat
        
        fused = self.fusion(local_feat, final_feat, tabular_feat)
        
        # Global average pooling
        image_feat = fused.mean(dim=1)  # B, D
        
        # Add pyramid features
        image_feat = image_feat + pyramid_feat
        
        return image_feat, tabular_feat
    
    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass."""
        
        image_feat, tabular_feat = self.forward_features(x, tabular)
        
        # Combine for classification
        if self.use_tabular and tabular_feat is not None:
            combined = torch.cat([image_feat, tabular_feat], dim=-1)
        else:
            combined = image_feat
        
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
        num_numerical_features: int = 33,
        num_categorical_features: int = 6,
        num_classes: int = 1,
    ):
        super().__init__()
        
        # Lighter configuration using depths parameter
        self.model = MambaDerm(
            img_size=img_size,
            embed_dim=64,
            depths=[1, 1, 2],
            d_state=8,
            num_numerical_features=num_numerical_features,
            num_categorical_features=num_categorical_features,
            num_classes=num_classes,
            dropout=0.15,
            use_multi_scale=False,
        )
    
    def forward(self, x, tabular=None):
        return self.model(x, tabular)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing MambaDerm...")
    
    model = MambaDerm(
        img_size=224,
        embed_dim=96,
        depths=[2, 2, 6],
        num_classes=1,
    )
    
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")
    
    x = torch.randn(2, 3, 224, 224)
    tabular = torch.randn(2, 40)
    
    with torch.no_grad():
        out = model(x, tabular)
    
    print(f"Input: {x.shape}")
    print(f"Tabular: {tabular.shape}")
    print(f"Output: {out.shape}")
    
    print("\nMambaDerm test passed!")
