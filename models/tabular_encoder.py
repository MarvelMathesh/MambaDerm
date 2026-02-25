"""
Tabular Encoder Module

Encodes clinical metadata (numerical and categorical features) 
for fusion with image features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TabularEncoder(nn.Module):
    """
    Encodes tabular/clinical metadata for multimodal fusion.
    
    Processes numerical features with normalization and categorical 
    features with embeddings, then projects to matching image feature dimension.
    
    Features based on ISIC 2024 metadata:
    - Numerical: age, lesion measurements, color features (34 features)
    - Categorical: sex, anatomical site, tile type, etc. (6 features)
    """
    
    def __init__(
        self,
        num_numerical: int = 34,
        num_categorical: int = 6,
        categorical_cardinalities: Optional[List[int]] = None,
        embedding_dim: int = 8,
        hidden_dim: int = 128,
        output_dim: int = 192,
        dropout: float = 0.1,
        use_missingness_embedding: bool = True,
    ):
        super().__init__()
        
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.use_missingness_embedding = use_missingness_embedding
        
        # Default cardinalities for ISIC categorical features
        if categorical_cardinalities is None:
            # sex (3), anatom_site_general (7), tbp_tile_type (4), 
            # tbp_lv_location (10), tbp_lv_location_simple (5), attribution (5)
            categorical_cardinalities = [3, 7, 4, 10, 5, 5]
        
        self.categorical_cardinalities = categorical_cardinalities
        
        # Learnable missingness embeddings for each numerical feature
        if use_missingness_embedding:
            self.missing_embed = nn.Parameter(torch.zeros(num_numerical))
            nn.init.normal_(self.missing_embed, std=0.02)
        
        # Numerical feature processing (expanded for missingness indicator)
        num_input_dim = num_numerical * 2 if use_missingness_embedding else num_numerical
        self.num_norm = nn.BatchNorm1d(num_numerical)
        self.num_proj = nn.Linear(num_input_dim, hidden_dim)
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embedding_dim, padding_idx=0)  # +1 for unknown/padding
            for card in categorical_cardinalities
        ])
        
        # Total categorical dimension
        total_cat_dim = num_categorical * embedding_dim
        self.cat_proj = nn.Linear(total_cat_dim, hidden_dim)
        
        # Combined MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self.output_dim = output_dim
    
    def forward(
        self, 
        numerical: torch.Tensor, 
        categorical: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            numerical: Numerical features of shape (B, num_numerical)
            categorical: Categorical features of shape (B, num_categorical), 
                        as integer indices
            
        Returns:
            Encoded features of shape (B, output_dim)
        """
        batch_size = numerical.shape[0]
        
        # Process numerical features with missingness encoding
        # Create missingness mask before replacing NaN
        if self.use_missingness_embedding:
            is_missing = torch.isnan(numerical).float()  # (B, num_numerical)
            # Replace NaN with learned embedding values
            numerical = torch.where(
                torch.isnan(numerical),
                self.missing_embed.unsqueeze(0).expand(batch_size, -1),
                numerical
            )
        else:
            numerical = torch.nan_to_num(numerical, nan=0.0)
            is_missing = None
        
        # Apply normalization to the (NaN-free) numerical features
        num_normed = self.num_norm(numerical)
        
        # Concatenate missingness indicator if enabled
        if self.use_missingness_embedding and is_missing is not None:
            num_feat = torch.cat([num_normed, is_missing], dim=-1)
        else:
            num_feat = num_normed
        
        num_feat = self.num_proj(num_feat)  # (B, hidden_dim)
        num_feat = F.gelu(num_feat)
        
        # Process categorical features
        if categorical is not None and self.num_categorical > 0:
            cat_embeds = []
            for i, embed_layer in enumerate(self.cat_embeddings):
                cat_idx = categorical[:, i].long()
                # Clamp to valid range
                cat_idx = cat_idx.clamp(0, self.categorical_cardinalities[i])
                cat_embeds.append(embed_layer(cat_idx))
            
            cat_feat = torch.cat(cat_embeds, dim=-1)  # (B, num_cat * embed_dim)
            cat_feat = self.cat_proj(cat_feat)  # (B, hidden_dim)
            cat_feat = F.gelu(cat_feat)
        else:
            # If no categorical, use zeros
            cat_feat = torch.zeros(batch_size, self.mlp[0].in_features // 2, 
                                   device=numerical.device, dtype=numerical.dtype)
        
        # Combine and project
        combined = torch.cat([num_feat, cat_feat], dim=-1)  # (B, hidden_dim * 2)
        output = self.mlp(combined)  # (B, output_dim)
        
        return output



