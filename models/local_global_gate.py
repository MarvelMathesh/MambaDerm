"""
Local-Global Gating Module

Adaptive fusion of local (CNN) and global (Mamba) features
with learnable gating mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LocalGlobalGate(nn.Module):
    """
    Learnable gating mechanism for fusing local and global features.
    
    Computes adaptive weights for combining:
    - Local features (from CNN stem)
    - Global features (from VMamba blocks)
    
    Gate formula:
        gate = σ(W_g · [local, global])
        output = gate ⊙ local + (1 - gate) ⊙ global
    """
    
    def __init__(
        self,
        dim: int,
        use_bias: bool = True,
        init_gate: float = 0.5,
    ):
        super().__init__()
        
        self.dim = dim
        
        # Gate projection
        self.gate_proj = nn.Linear(dim * 2, dim, bias=use_bias)
        
        # Initialize gate bias to produce init_gate value
        if use_bias:
            # sigmoid(x) = init_gate => x = log(init_gate / (1 - init_gate))
            init_bias = torch.log(torch.tensor(init_gate / (1 - init_gate)))
            self.gate_proj.bias.data.fill_(init_bias.item())
    
    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            local_feat: Local features of shape (B, L, D)
            global_feat: Global features of shape (B, L, D)
            
        Returns:
            Fused features of shape (B, L, D)
        """
        # Concatenate for gate computation
        combined = torch.cat([local_feat, global_feat], dim=-1)  # (B, L, 2D)
        
        # Compute gate
        gate = torch.sigmoid(self.gate_proj(combined))  # (B, L, D)
        
        # Gated fusion
        output = gate * local_feat + (1 - gate) * global_feat
        
        return output


class CrossModalAttention(nn.Module):
    """
    Cross-attention for fusing image and tabular features.
    
    Image features query tabular features to incorporate clinical context.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from image features
        self.q_proj = nn.Linear(dim, dim, bias=True)
        # Key, Value from tabular features
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        
        self.out_proj = nn.Linear(dim, dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_img = nn.LayerNorm(dim)
        self.norm_tab = nn.LayerNorm(dim)
    
    def forward(
        self,
        image_feat: torch.Tensor,
        tabular_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_feat: Image features of shape (B, L_img, D)
            tabular_feat: Tabular features of shape (B, L_tab, D)
            
        Returns:
            Updated image features of shape (B, L_img, D)
        """
        B, L_img, D = image_feat.shape
        L_tab = tabular_feat.shape[1]
        
        # Normalize inputs
        image_feat_norm = self.norm_img(image_feat)
        tabular_feat_norm = self.norm_tab(tabular_feat)
        
        # Compute Q, K, V
        Q = self.q_proj(image_feat_norm)  # (B, L_img, D)
        K = self.k_proj(tabular_feat_norm)  # (B, L_tab, D)
        V = self.v_proj(tabular_feat_norm)  # (B, L_tab, D)
        
        # Reshape for multi-head attention
        Q = Q.view(B, L_img, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L_tab, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_tab, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, L_img, L_tab)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = attn @ V  # (B, H, L_img, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L_img, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # Residual connection
        return image_feat + out


class MultiModalFusion(nn.Module):
    """
    Complete multimodal fusion module.
    
    Combines:
    - Local-Global gating for image features
    - Cross-modal attention for image-tabular fusion
    - Final projection
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
        
        # Local-Global fusion
        self.local_global_gate = LocalGlobalGate(dim)
        
        # Cross-modal attention (optional)
        if use_cross_attention:
            self.cross_attn = CrossModalAttention(dim, num_heads, dropout)
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
        tabular_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            local_feat: Local (CNN) features of shape (B, L, D)
            global_feat: Global (Mamba) features of shape (B, L, D)
            tabular_feat: Tabular features of shape (B, L_tab, D) or (B, D)
            
        Returns:
            Fused features of shape (B, L, D)
        """
        # Local-Global fusion
        fused = self.local_global_gate(local_feat, global_feat)
        
        # Cross-modal attention with tabular features
        if self.use_cross_attention and tabular_feat is not None:
            # If tabular_feat is (B, D), expand to (B, 1, D)
            if tabular_feat.dim() == 2:
                tabular_feat = tabular_feat.unsqueeze(1)
            
            fused = self.cross_attn(fused, tabular_feat)
        
        fused = self.norm(fused)
        
        return fused
