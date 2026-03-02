"""
Tabular Transformer Encoder

Per-feature tokenization with self-attention across clinical metadata features.
Each numerical/categorical feature becomes a token; self-attention captures
feature interactions (e.g., age × lesion-size, site × morphology).
CLS token aggregation produces the final output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class TabularEncoder(nn.Module):
    """
    Transformer-based tabular encoder with per-feature tokenization.
    
    Each feature (numerical or categorical) is projected into a d-dimensional
    token. A learnable [CLS] token is prepended, and 2-layer self-attention
    captures cross-feature interactions. The [CLS] output is the final
    (B, output_dim) representation.
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
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.use_missingness_embedding = use_missingness_embedding
        self.output_dim = output_dim
        
        # Token dimension = hidden_dim
        token_dim = hidden_dim
        
        if categorical_cardinalities is None:
            categorical_cardinalities = [3, 7, 4, 10, 5, 5]
        self.categorical_cardinalities = categorical_cardinalities
        
        # Learnable missingness fill values
        if use_missingness_embedding:
            self.missing_embed = nn.Parameter(torch.zeros(num_numerical))
            nn.init.normal_(self.missing_embed, std=0.02)
        
        # Per-feature numerical projections: each scalar → token_dim
        # (with optional missingness indicator → 2 input dims per feature)
        feat_input_dim = 2 if use_missingness_embedding else 1
        self.num_tokenizers = nn.ModuleList([
            nn.Linear(feat_input_dim, token_dim)
            for _ in range(num_numerical)
        ])
        self.num_norm = nn.BatchNorm1d(num_numerical)
        
        # Categorical embeddings → token_dim each
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, token_dim, padding_idx=0)
            for card in categorical_cardinalities
        ])
        
        # [CLS] token
        total_tokens = num_numerical + num_categorical + 1  # +1 for CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Learnable positional embedding for all tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection from CLS token
        self.out_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(
        self,
        numerical: torch.Tensor,
        categorical: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            numerical: (B, num_numerical)
            categorical: (B, num_categorical) integer indices
        Returns:
            (B, output_dim)
        """
        B = numerical.shape[0]
        tokens = []
        
        # Handle missing values
        if self.use_missingness_embedding:
            is_missing = torch.isnan(numerical).float()
            numerical = torch.where(
                torch.isnan(numerical),
                self.missing_embed.unsqueeze(0).expand(B, -1),
                numerical
            )
        else:
            is_missing = None
            numerical = torch.nan_to_num(numerical, nan=0.0)
        
        # Normalize numerical features
        num_normed = self.num_norm(numerical)
        
        # Tokenize each numerical feature independently
        for i, tokenizer in enumerate(self.num_tokenizers):
            val = num_normed[:, i:i+1]  # (B, 1)
            if self.use_missingness_embedding and is_missing is not None:
                val = torch.cat([val, is_missing[:, i:i+1]], dim=-1)  # (B, 2)
            tok = tokenizer(val)  # (B, token_dim)
            tokens.append(tok.unsqueeze(1))  # (B, 1, token_dim)
        
        # Tokenize categorical features
        if categorical is not None and self.num_categorical > 0:
            for i, embed_layer in enumerate(self.cat_embeddings):
                cat_idx = categorical[:, i].long().clamp(0, self.categorical_cardinalities[i])
                tok = embed_layer(cat_idx)  # (B, token_dim)
                tokens.append(tok.unsqueeze(1))
        else:
            # Pad with zeros for missing categorical
            for _ in range(self.num_categorical):
                tokens.append(torch.zeros(B, 1, tokens[0].shape[-1],
                                         device=numerical.device, dtype=numerical.dtype))
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = [cls] + tokens
        
        # Stack all tokens: (B, 1+num_numerical+num_categorical, token_dim)
        x = torch.cat(tokens, dim=1)
        x = x + self.pos_embed
        
        # Self-attention across features
        x = self.transformer(x)
        
        # Extract CLS token
        cls_out = x[:, 0]  # (B, token_dim)
        
        return self.out_proj(cls_out)  # (B, output_dim)