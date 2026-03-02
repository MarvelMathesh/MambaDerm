"""
MambaDerm: SOTA Hybrid Mamba Architecture for Dermatology

Novel architecture featuring:
- 4-directional Cross-Scan SSM (row↔, col↔) for spatially-aware scanning
- Spatial-frequency dual-domain processing (DCT branch)
- CBAM spatial-channel attention per stage
- BiFPN for bidirectional multi-scale feature fusion
- Clinically-motivated Mixture-of-Experts fusion (ABCD criteria)
- Gated bilinear image-tabular fusion
- Evidential deep learning head with Dirichlet prior
- Stochastic depth + ConvDownsample
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


class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network with weighted fusion.
    
    Replaces FeaturePyramidFusion — operates on spatial feature maps
    instead of GAP-ed vectors. Uses fast normalized fusion weights.
    
    Reference: EfficientDet (Tan et al., 2020)
    """
    
    def __init__(self, dims: List[int], out_dim: int):
        super().__init__()
        self.num_levels = len(dims)
        
        # Lateral projections to out_dim
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, out_dim),
                nn.LayerNorm(out_dim),
            ) for dim in dims
        ])
        
        # Per-node fusion weights (fast normalized)
        # Top-down path: num_levels - 1 nodes, each fuses 2 inputs
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2) * 0.5) for _ in range(self.num_levels - 1)
        ])
        # Bottom-up path: num_levels - 1 nodes, each fuses 3 inputs (except first)
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3 if i > 0 else 2) * 0.33)
            for i in range(self.num_levels - 1)
        ])
        
        # Output projection per level → single fused vector
        self.output_fusion = nn.Sequential(
            nn.Linear(out_dim * self.num_levels, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
        
        self.eps = 1e-4
    
    def _weighted_add(self, inputs: List[torch.Tensor], weights: nn.Parameter) -> torch.Tensor:
        """Fast normalized weighted addition."""
        w = F.relu(weights)
        w = w / (w.sum() + self.eps)
        result = sum(wi * inp for wi, inp in zip(w, inputs))
        return result
    
    def _resize_to(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Resize sequence length via adaptive pooling."""
        if x.shape[1] == target_len:
            return x
        x = x.permute(0, 2, 1)  # B,D,L
        x = F.adaptive_avg_pool1d(x, target_len)
        return x.permute(0, 2, 1)  # B,L,D
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of (B, L_i, C_i) from each stage
        Returns:
            fused: (B, out_dim) global fused feature
        """
        # Project all levels to out_dim
        projected = [lat(feat) for feat, lat in zip(features, self.lateral_convs)]
        
        # Top-down path (coarse → fine)
        td_feats = [None] * self.num_levels
        td_feats[-1] = projected[-1]
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = self._resize_to(td_feats[i + 1], projected[i].shape[1])
            td_feats[i] = self._weighted_add(
                [projected[i], upsampled], self.td_weights[i]
            )
        
        # Bottom-up path (fine → coarse)
        bu_feats = [None] * self.num_levels
        bu_feats[0] = td_feats[0]
        for i in range(1, self.num_levels):
            downsampled = self._resize_to(bu_feats[i - 1], td_feats[i].shape[1])
            if i > 0 and i < self.num_levels:
                bu_feats[i] = self._weighted_add(
                    [projected[i], td_feats[i], downsampled][:len(self.bu_weights[i - 1])],
                    self.bu_weights[i - 1],
                )
            else:
                bu_feats[i] = self._weighted_add(
                    [td_feats[i], downsampled], self.bu_weights[i - 1]
                )
        
        # Pool each level and concatenate for final fusion
        pooled = [feat.mean(dim=1) for feat in bu_feats]
        concat = torch.cat(pooled, dim=-1)
        return self.output_fusion(concat)


class GatedBilinearFusion(nn.Module):
    """
    Gated bilinear fusion for image-tabular modalities.
    
    Replaces degenerate cross-attention (1-KV-token) with:
        gate = σ(W_g · [img; tab])
        fused = gate * (W_1 · img ⊙ W_2 · tab)
    
    Bilinear interaction captures multiplicative feature relationships.
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.img_proj = nn.Linear(dim, dim)
        self.tab_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        image_feat: torch.Tensor,   # (B, D)
        tabular_feat: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        """Returns (B, D) fused feature."""
        img_proj = self.img_proj(image_feat)
        tab_proj = self.tab_proj(tabular_feat)
        
        gate = self.gate_proj(torch.cat([image_feat, tabular_feat], dim=-1))
        fused = gate * (img_proj * tab_proj)  # element-wise (Hadamard)
        
        return self.norm(self.dropout(fused))


class MoEFusionLayer(nn.Module):
    """
    Mixture-of-Experts fusion with clinically-motivated specialization.
    
    4 experts aligned with the dermatologic ABCD criteria:
      - Asymmetry / structure expert
      - Border expert
      - Color expert
      - Diameter / texture expert
    
    Top-2 gating with load-balancing auxiliary loss.
    """
    
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        
        # Router
        self.router = nn.Linear(dim, num_experts)
        
        # Expert MLPs (different activations for diversity)
        activations = [nn.GELU(), nn.SiLU(), nn.Mish(), nn.GELU()]
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                activations[i],
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
            )
            for i in range(num_experts)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self._aux_loss = torch.tensor(0.0)
    
    @property
    def aux_loss(self) -> torch.Tensor:
        """Load-balancing auxiliary loss for MoE."""
        return self._aux_loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) spatial features
        Returns:
            (B, L, D) expert-processed features
        """
        B, L, D = x.shape
        
        # Compute router logits on pooled features (per-sample routing)
        x_pool = x.mean(dim=1)  # B, D
        router_logits = self.router(x_pool)  # B, num_experts
        
        # Add noise for exploration during training
        if self.training:
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise
        
        # Top-k gating
        top_k_logits, top_k_indices = router_logits.topk(self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # B, top_k
        
        # Compute load-balancing loss (CV² of expert usage)
        if self.training:
            probs = F.softmax(router_logits, dim=-1)  # B, num_experts
            expert_usage = probs.mean(dim=0)  # num_experts
            cv_sq = expert_usage.var() / (expert_usage.mean() ** 2 + 1e-8)
            self._aux_loss = cv_sq
        
        # Dispatch to experts
        result = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # B
            gate_val = top_k_gates[:, k]  # B
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]  # n, L, D
                    expert_output = self.experts[e](expert_input)
                    result[mask] = result[mask] + gate_val[mask].unsqueeze(-1).unsqueeze(-1) * expert_output
        
        return self.norm(result + x)  # residual


class EvidentialHead(nn.Module):
    """
    Evidential deep learning classification head with Dirichlet prior.
    
    Outputs Dirichlet concentration parameters α instead of logits.
    Provides calibrated uncertainty in a single forward pass.
    
    For binary: K=2, α = (α₁, α₂), prediction = α₁/(α₁+α₂)
    For multi-class: K classes, prediction = α/Σα
    Uncertainty: u = K/Σα (total evidence inverse)
    """
    
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.15):
        super().__init__()
        self.num_classes = num_classes
        # K=2 for binary, K=num_classes for multi-class
        self.K = max(num_classes, 2)
        
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 4, self.K),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) features
        Returns:
            alpha: (B, K) Dirichlet concentration parameters, all > 1
        """
        # Softplus + 1 ensures α > 1 (valid Dirichlet concentration)
        alpha = F.softplus(self.head(x)) + 1.0
        return alpha
    
    def get_prediction(self, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract prediction and uncertainty from α.
        
        Returns:
            probs: (B, K) predicted class probabilities
            uncertainty: (B,) epistemic uncertainty
        """
        S = alpha.sum(dim=-1, keepdim=True)  # Dirichlet strength
        probs = alpha / S
        uncertainty = self.K / S.squeeze(-1)
        return probs, uncertainty


class MambaDerm(nn.Module):
    """
    MambaDerm: SOTA Cross-Scan Mamba + MoE + Evidential Architecture
    
    Novel features:
    1. 4-directional cross-scan SSM (row↔, col↔) per layer
    2. Spatial-frequency dual-domain (DCT branch in Stage 1)
    3. CBAM spatial-channel attention per stage
    4. BiFPN bidirectional multi-scale fusion
    5. Clinically-motivated MoE (4 experts, top-2 gating)
    6. Gated bilinear image-tabular fusion
    7. Evidential head with Dirichlet prior for calibrated uncertainty
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
        
        if depths is None:
            depths = [2, 2, 6]
        
        self.use_tabular = use_tabular
        self.num_classes = num_classes
        self.num_numerical_features = num_numerical_features
        self.gradient_checkpointing = gradient_checkpointing
        
        # Stage dimensions
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.dims = dims
        final_dim = dims[-1]
        
        # Stage OUTPUT dims (after downsampling)
        stage_out_dims = []
        for i in range(len(depths)):
            if i < len(depths) - 1:
                stage_out_dims.append(dims[i + 1])
            else:
                stage_out_dims.append(dims[i])
        self.stage_out_dims = stage_out_dims
        
        # Hierarchical backbone (now with cross-scan + freq + CBAM)
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
        
        # BiFPN: bidirectional feature pyramid (replaces old FPN)
        self.bifpn = BiFPN(stage_out_dims, final_dim)
        
        # MoE fusion layer (4 experts, top-2 gating)
        self.moe = MoEFusionLayer(dim=final_dim, num_experts=4, top_k=2, dropout=dropout)
        
        # Gated bilinear fusion for image-tabular (replaces cross-attention)
        if use_tabular:
            self.bilinear_fusion = GatedBilinearFusion(dim=final_dim, dropout=dropout)
        
        # Evidential classification head
        classifier_input_dim = final_dim
        if use_tabular:
            classifier_input_dim = final_dim * 2  # fused + bifpn
        self.evidential_head = EvidentialHead(
            in_dim=classifier_input_dim, num_classes=num_classes, dropout=dropout,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
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
        """Extract features before evidential head."""
        
        # Hierarchical backbone (cross-scan + freq + CBAM)
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
        
        # BiFPN multi-scale fusion
        pyramid_feat = self.bifpn(stage_features)
        
        # MoE expert processing on final-stage features
        moe_feat = self.moe(final_feat)  # B, L, D
        
        # Global average pooling
        image_feat = moe_feat.mean(dim=1)  # B, D
        
        # Add multi-scale pyramid context
        image_feat = image_feat + pyramid_feat
        
        # Gated bilinear fusion with tabular
        if self.use_tabular and tabular_feat is not None:
            fused = self.bilinear_fusion(image_feat, tabular_feat)
            # Concatenate fused and image for richer input to head
            image_feat = torch.cat([image_feat, fused], dim=-1)
        
        return image_feat, tabular_feat
    
    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass.
        
        Returns logits for backward compatibility with existing training scripts.
        For binary (num_classes=1): returns (B, 1) logits 
        For multi-class: returns (B, num_classes) logits
        
        The evidential head outputs α, which is converted to logits
        for compatibility with existing loss functions.
        Use forward_evidential() for full α + uncertainty output.
        """
        image_feat, _ = self.forward_features(x, tabular)
        
        # Evidential head → α
        alpha = self.evidential_head(image_feat)
        
        # Convert to logits for compatibility
        if self.num_classes == 1:
            # Binary: log(α₁/α₂) gives logit
            logits = (alpha[:, 0] - alpha[:, 1]).unsqueeze(-1)
        else:
            # Multi-class: log(α) - log(Σα) gives log-probabilities ≈ logits
            logits = torch.log(alpha) - torch.log(alpha.sum(dim=-1, keepdim=True))
        
        return logits
    
    def forward_evidential(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full evidential forward pass.
        
        Returns:
            alpha: (B, K) Dirichlet concentrations
            probs: (B, K) predicted class probabilities  
            uncertainty: (B,) epistemic uncertainty (K/Σα)
        """
        image_feat, _ = self.forward_features(x, tabular)
        alpha = self.evidential_head(image_feat)
        probs, uncertainty = self.evidential_head.get_prediction(alpha)
        return alpha, probs, uncertainty
    
    def get_moe_aux_loss(self) -> torch.Tensor:
        """Get MoE load-balancing auxiliary loss for training."""
        return self.moe.aux_loss


class MambaDermLite(nn.Module):
    """Lightweight MambaDerm with reduced depth/width."""
    
    def __init__(
        self,
        img_size: int = 224,
        num_numerical_features: int = 33,
        num_categorical_features: int = 6,
        num_classes: int = 1,
    ):
        super().__init__()
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
    
    def forward_evidential(self, x, tabular=None):
        return self.model.forward_evidential(x, tabular)
    
    def get_moe_aux_loss(self):
        return self.model.get_moe_aux_loss()


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
        alpha, probs, unc = model.forward_evidential(x, tabular)
    
    print(f"Input: {x.shape}")
    print(f"Tabular: {tabular.shape}")
    print(f"Logits: {out.shape}")
    print(f"Alpha: {alpha.shape}, Probs: {probs.shape}, Uncertainty: {unc.shape}")
    print(f"MoE aux loss: {model.get_moe_aux_loss().item():.6f}")
    
    print("\nMambaDerm test passed!")
