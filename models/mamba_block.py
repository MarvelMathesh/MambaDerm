"""
Vision Mamba Block

Implements the Selective State Space Model (Mamba) for vision tasks.
Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
and "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model"
"""

import math
from typing import Optional, Tuple
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Try to import mamba_ssm, fall back to pure PyTorch implementation
try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not available, using pure PyTorch implementation (slower)")


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
    
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Pure PyTorch implementation.
    
    This is a fallback when mamba-ssm CUDA kernels are not available.
    The selective mechanism allows input-dependent state transitions.
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension (N in paper)
        d_conv: Local convolution width
        expand: Expansion factor for inner dimension
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Depthwise conv for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # SSM parameters (input-dependent)
        # dt_rank: controls per-channel selectivity granularity (Mamba paper §3.4)
        self.dt_rank = max(1, math.ceil(d_model / 16))
        
        # dt (Δ), B, C projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt (Δ) projection - from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # A parameter (learned, not input-dependent)
        # Initialize A as -exp(uniform) for stable dynamics
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, L, D)
            
        Returns:
            Output tensor of shape (B, L, D)
        """
        batch, seq_len, dim = x.shape
        
        # Input projection and split into x and gate (z)
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Local convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :seq_len]  # Causal conv
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        
        # SSM parameters (input-dependent)
        x_proj = self.x_proj(x)  # (B, L, dt_rank + d_state*2)
        dt = x_proj[:, :, :self.dt_rank]
        B = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C = x_proj[:, :, self.dt_rank + self.d_state:]
        
        # dt projection
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)  # Ensure positive
        
        # A from log space (stable parameterization)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        # For efficiency, we compute the recurrence directly
        
        # Run selective scan (simplified recurrence)
        y = self._selective_scan(x, dt, A, B, C)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gated output
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y
    
    def _selective_scan(
        self,
        x: torch.Tensor,  # (B, L, d_inner)
        dt: torch.Tensor,  # (B, L, d_inner)
        A: torch.Tensor,  # (d_inner, d_state)
        B: torch.Tensor,  # (B, L, d_state)
        C: torch.Tensor,  # (B, L, d_state)
    ) -> torch.Tensor:
        """
        Selective scan (SSM recurrence).
        
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        # Output accumulator
        outputs = []
        
        # Discretized A: exp(dt * A)
        # dt: (B, L, d_inner), A: (d_inner, d_state)
        # We need (B, L, d_inner, d_state)
        
        for t in range(seq_len):
            dt_t = dt[:, t, :, None]  # (B, d_inner, 1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # (B, d_inner, d_state)
            B_bar = dt_t * B[:, t, None, :]  # (B, d_inner, d_state)
            
            # State update
            h = A_bar * h + B_bar * x[:, t, :, None]  # (B, d_inner, d_state)
            
            # Output
            y_t = (h * C[:, t, None, :]).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return y


class VisionMambaBlock(nn.Module):
    """
    Vision Mamba block with bidirectional scanning.
    
    Combines forward and backward SSM passes for images,
    following the VMamba paper approach. Uses gated fusion
    for adaptive forward/backward balance.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.bidirectional = bidirectional
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Forward SSM
        if MAMBA_AVAILABLE:
            self.ssm_forward = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.ssm_forward = SelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
        
        # Backward SSM (for bidirectional)
        if bidirectional:
            if MAMBA_AVAILABLE:
                self.ssm_backward = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            else:
                self.ssm_backward = SelectiveSSM(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
            
            # Gated fusion for bidirectional outputs
            # Adaptive control over forward vs backward contribution per-token
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, L, D) - flattened image patches
            
        Returns:
            Output tensor of shape (B, L, D)
        """
        residual = x
        x = self.norm(x)
        
        # Forward pass
        y_forward = self.ssm_forward(x)
        
        if self.bidirectional:
            # Backward pass (reverse sequence)
            x_backward = torch.flip(x, dims=[1])
            y_backward = self.ssm_backward(x_backward)
            y_backward = torch.flip(y_backward, dims=[1])  # Flip back
            
            # Gated fusion of forward and backward
            combined = torch.cat([y_forward, y_backward], dim=-1)
            gate = self.gate(combined)
            y = gate * y_forward + (1 - gate) * y_backward
        else:
            y = y_forward
        
        # Residual connection with stochastic depth
        y = self.drop_path(self.dropout(y)) + residual
        
        return y


class VMambaLayer(nn.Module):
    """
    Complete VMamba layer with SSM block and MLP.
    
    Similar to Transformer layer structure:
    x -> SSM Block -> MLP -> output
    
    Supports stochastic depth (drop path) for regularization.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        # SSM Block (drop_path applied inside VisionMambaBlock)
        self.ssm_block = VisionMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            drop_path=drop_path,
            bidirectional=bidirectional,
        )
        
        # MLP Block
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )
        
        # Drop path for MLP branch
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, L, D)
            
        Returns:
            Output tensor of shape (B, L, D)
        """
        # SSM block (residual + drop_path handled inside VisionMambaBlock)
        x = self.ssm_block(x)
        
        # MLP block with residual + stochastic depth
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        
        return x
