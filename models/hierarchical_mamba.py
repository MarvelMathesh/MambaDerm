"""
Hierarchical Vision Mamba Backbone

Multi-scale hierarchical architecture for improved feature extraction:
- Stage 1: 56×56 resolution (fine details)
- Stage 2: 28×28 resolution (medium features)
- Stage 3: 14×14 resolution (global context)

Reference: VMamba, Swin Transformer hierarchical design
"""

import math
from typing import Optional, Sequence, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .mamba_block import VisionMambaBlock, VMambaLayer


class ConvDownsample(nn.Module):
    """
    Depthwise-separable convolution for downsampling between stages.
    
    Preserves more spatial information than PatchMerging (2×2 concat).
    Uses strided depthwise conv + pointwise expansion.
    """
    
    def __init__(self, dim: int, out_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim * 2
        
        # Depthwise conv with stride 2 for spatial reduction
        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=2, padding=1,
            groups=dim, bias=False,
        )
        # Pointwise conv for channel expansion
        self.pw_conv = nn.Conv2d(dim, self.out_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B,C,H,W
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        H_new, W_new = x.shape[2], x.shape[3]
        x = x.permute(0, 2, 3, 1).reshape(B, H_new * W_new, self.out_dim)  # B,L',C'
        x = self.norm(x)
        return x, H_new, W_new


class SpatialChannelAttention(nn.Module):
    """
    CBAM-lite: Channel Attention + Spatial Attention.
    
    Sequentially applies channel and spatial attention for
    feature refinement after each stage.
    """
    
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        # Channel attention (SE-style)
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: (B, H*W, C)"""
        B, L, C = x.shape
        
        # Channel attention: GAP → FC → scale
        gap = x.mean(dim=1)  # B, C
        ch_gate = self.channel_gate(gap).unsqueeze(1)  # B, 1, C
        x = x * ch_gate
        
        # Spatial attention: concat(max, mean) along channel → conv → scale
        x_2d = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B,C,H,W
        sp_max = x_2d.max(dim=1, keepdim=True)[0]  # B,1,H,W
        sp_mean = x_2d.mean(dim=1, keepdim=True)  # B,1,H,W
        sp_gate = self.spatial_conv(torch.cat([sp_max, sp_mean], dim=1))  # B,1,H,W
        x_2d = x_2d * sp_gate
        
        return x_2d.permute(0, 2, 3, 1).reshape(B, L, C)


class MultiScalePatchEmbed(nn.Module):
    """
    Multi-scale patch embedding with feature pyramid.
    
    Creates patches at multiple scales (4×4, 8×8, 16×16) and fuses them
    for richer multi-scale representations.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        embed_dim: int = 96,
        scales: Optional[List[int]] = None,
    ):
        super().__init__()
        
        # Handle mutable default argument
        if scales is None:
            scales = [4, 8, 16]
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.scales = scales
        
        # Create patch embeddings at each scale
        self.patch_embeds = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for scale in scales:
            self.patch_embeds.append(
                nn.Conv2d(
                    in_channels, embed_dim,
                    kernel_size=scale, stride=scale
                )
            )
            self.norms.append(nn.LayerNorm(embed_dim))
        
        # Fusion: upsample smaller scales and combine
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * len(scales), embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Target resolution (from smallest scale - 4×4 patches)
        self.target_H = img_size // min(scales)
        self.target_W = img_size // min(scales)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, 3, H, W)
            
        Returns:
            x: (B, L, embed_dim) fused multi-scale features
            H, W: Spatial dimensions
        """
        B = x.shape[0]
        features = []
        
        for i, (patch_embed, norm) in enumerate(zip(self.patch_embeds, self.norms)):
            # Get patches at this scale
            feat = patch_embed(x)  # B, C, H_i, W_i
            H_i, W_i = feat.shape[2], feat.shape[3]
            
            # Upsample to target resolution if needed
            if H_i != self.target_H or W_i != self.target_W:
                feat = F.interpolate(
                    feat, size=(self.target_H, self.target_W),
                    mode='bilinear', align_corners=False
                )
            
            # Flatten and normalize
            feat = rearrange(feat, 'b c h w -> b (h w) c')
            feat = norm(feat)
            features.append(feat)
        
        # Concatenate and fuse
        x = torch.cat(features, dim=-1)  # B, L, embed_dim * num_scales
        x = self.fusion(x)
        
        return x, self.target_H, self.target_W


class HierarchicalMambaStage(nn.Module):
    """
    Single stage of hierarchical Mamba backbone.
    
    Contains multiple VMamba layers + CBAM attention + optional downsampling.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rates: Optional[Sequence[float]] = None,
        downsample: bool = True,
        out_dim: Optional[int] = None,
        use_freq_branch: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Per-layer stochastic depth rates from the global schedule
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        
        # Mamba layers with cross-scan and stochastic depth
        self.layers = nn.ModuleList([
            VMambaLayer(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path_rates[i],
                bidirectional=True,
                use_freq_branch=use_freq_branch and (i == 0),  # freq on first layer only
            )
            for i in range(depth)
        ])
        
        # CBAM attention after stage
        self.attention = SpatialChannelAttention(dim)
        
        # Downsampling
        if downsample:
            self.downsample = ConvDownsample(dim, out_dim)
        else:
            self.downsample = None
    
    def forward(
        self, x: torch.Tensor, H: int, W: int
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, L, C)
            H, W: Spatial dimensions
            
        Returns:
            x: (B, L', C') features after stage
            H_new, W_new: New spatial dimensions
        """
        for layer in self.layers:
            x = layer(x, H, W)
        
        # CBAM refinement
        x = self.attention(x, H, W)
        
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        
        return x, H, W


class HierarchicalMambaBackbone(nn.Module):
    """
    Hierarchical Vision Mamba Backbone.
    
    3-stage pyramid architecture:
    - Stage 1: 56×56, 2 layers, dim=96
    - Stage 2: 28×28, 2 layers, dim=192
    - Stage 3: 14×14, 4 layers, dim=384
    
    Args:
        img_size: Input image size
        in_channels: Number of input channels
        embed_dim: Base embedding dimension
        depths: Number of layers in each stage
        dims: Dimension for each stage
        d_state: SSM state dimension
        drop_path_rate: Stochastic depth rate
    """
    
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        use_multi_scale_embed: bool = True,
    ):
        super().__init__()
        
        # Handle mutable default arguments
        if depths is None:
            depths = [2, 2, 4]
        
        self.num_stages = len(depths)
        dims = dims or [embed_dim * (2 ** i) for i in range(self.num_stages)]
        self.dims = dims
        
        # Patch embedding
        if use_multi_scale_embed:
            self.patch_embed = MultiScalePatchEmbed(
                img_size=img_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                scales=[4, 8, 16],
            )
            initial_H = img_size // 4
            initial_W = img_size // 4
        else:
            # Use a proper channel-only LayerNorm via a wrapper
            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4),
            )
            self._patch_embed_norm = nn.LayerNorm(embed_dim)
            initial_H = img_size // 4
            initial_W = img_size // 4
        
        self.use_multi_scale_embed = use_multi_scale_embed
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, initial_H * initial_W, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        # Build stages (freq branch only in stage 1 for texture sensitivity)
        self.stages = nn.ModuleList()
        cur = 0
        
        for i in range(self.num_stages):
            stage = HierarchicalMambaStage(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path_rates=dpr[cur:cur + depths[i]],
                downsample=(i < self.num_stages - 1),  # No downsample at last stage
                out_dim=dims[i + 1] if i < self.num_stages - 1 else None,
                use_freq_branch=(i == 0),  # Frequency branch on first stage
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # Final norm
        self.norm = nn.LayerNorm(dims[-1])
        
        self.initial_H = initial_H
        self.initial_W = initial_W
        self.out_dim = dims[-1]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 3, H, W) input images
            
        Returns:
            x: (B, L, C) final features
            features: List of features from each stage
        """
        # Patch embedding
        if self.use_multi_scale_embed:
            x, H, W = self.patch_embed(x)
        else:
            x = self.patch_embed(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self._patch_embed_norm(x)
            H, W = self.initial_H, self.initial_W
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Process through stages
        features = []
        for stage in self.stages:
            x, H, W = stage(x, H, W)
            features.append(x)
        
        x = self.norm(x)
        
        return x, features
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get final features only."""
        x, _ = self.forward(x)
        return x


if __name__ == "__main__":
    # Test hierarchical backbone
    print("Testing HierarchicalMambaBackbone...")
    
    backbone = HierarchicalMambaBackbone(
        img_size=224,
        embed_dim=96,
        depths=[2, 2, 4],
        drop_path_rate=0.1,
    )
    
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        out, features = backbone(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Stage features: {[f.shape for f in features]}")
    
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")
    
    print("\nHierarchicalMambaBackbone test passed!")
