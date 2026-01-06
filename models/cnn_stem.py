"""
CNN Stem Module

ConvNeXt-style stem for local feature extraction before VMamba blocks.
Provides strong inductive bias for early feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block: Depthwise Conv -> LayerNorm -> Linear -> GELU -> Linear
    
    Modernized ResNet block following the ConvNeXt paper design.
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        hidden_dim = int(dim * expansion_ratio)
        
        # Depthwise convolution
        self.dwconv = nn.Conv2d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )
        
        # LayerNorm (channels last format)
        self.norm = nn.LayerNorm(dim)
        
        # Pointwise convolutions (as linear layers)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Layer scale (learnable scaling, ConvNeXt innovation)
        self.layer_scale = nn.Parameter(torch.ones(dim) * 1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        residual = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute to channels last: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        # LayerNorm + MLP
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pwconv2(x)
        
        # Layer scale
        x = x * self.layer_scale
        
        # Permute back: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Residual connection with drop path
        x = residual + self.drop_path(x)
        
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for regularization.
    
    Reference: "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382)
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtStem(nn.Module):
    """
    ConvNeXt-style stem for MambaDerm.
    
    Patchifies input image and applies ConvNeXt blocks for local feature extraction.
    Output is flattened spatial dimensions for VMamba processing.
    
    Architecture:
        Input (224x224) -> Patch Embed (56x56) -> ConvNeXt Blocks -> Flatten -> (3136, D)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 96,
        out_channels: int = 192,
        kernel_size: int = 4,
        stride: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        
        # Patch embedding (patchify layer) - Conv2d only, no unused LayerNorm
        self.patch_conv = nn.Conv2d(
            in_channels, 
            stem_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            bias=True,
        )
        
        # GroupNorm for 2D feature maps (equivalent to LayerNorm for spatial features)
        self.patch_norm = nn.GroupNorm(1, stem_channels)
        
        # Stochastic depth rates for each block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        
        # ConvNeXt blocks for local feature extraction with drop path
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=stem_channels,
                kernel_size=7,
                expansion_ratio=4.0,
                dropout=dropout,
                drop_path=dpr[i],
            )
            for i in range(num_blocks)
        ])
        
        # Project to output dimension if different
        if stem_channels != out_channels:
            self.proj = nn.Conv2d(stem_channels, out_channels, kernel_size=1, bias=True)
            self.proj_norm = nn.GroupNorm(1, out_channels)
        else:
            self.proj = nn.Identity()
            self.proj_norm = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            features: Flattened features of shape (B, L, D) for VMamba
            spatial_shape: Tuple (H, W) of spatial dimensions before flattening
        """
        B, C, H, W = x.shape
        
        # Patch embedding: (B, 3, 224, 224) -> (B, stem_channels, 56, 56)
        x = self.patch_conv(x)
        x = self.patch_norm(x)
        
        # ConvNeXt blocks for local features
        for block in self.blocks:
            x = block(x)
        
        # Project to output dim
        x = self.proj(x)
        x = self.proj_norm(x)
        
        # Get spatial shape before flattening
        _, _, H_out, W_out = x.shape
        spatial_shape = (H_out, W_out)
        
        # Flatten spatial dimensions: (B, D, H, W) -> (B, H*W, D)
        x = x.flatten(2).transpose(1, 2)  # (B, L, D) where L = H*W
        
        return x, spatial_shape


class PatchEmbed2D(nn.Module):
    """
    Simple patch embedding without ConvNeXt blocks.
    
    For ablation studies or faster training.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 192,
        patch_size: int = 4,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            features: Flattened features of shape (B, L, D)
            spatial_shape: Tuple (H, W) of spatial dimensions
        """
        x = self.proj(x)  # (B, D, H/P, W/P)
        _, _, H_out, W_out = x.shape
        spatial_shape = (H_out, W_out)
        
        x = x.flatten(2).transpose(1, 2)  # (B, L, D)
        x = self.norm(x)
        
        return x, spatial_shape
