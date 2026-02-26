"""
VSL-Native Diffusion Model for Sign Language Transition Generation

This module implements a transformer-based diffusion model trained directly
on VSL skeleton format (1659 features) for generating smooth transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from pathlib import Path


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ff_mult: int = 4
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class VSLDiffusionModel(nn.Module):
    """
    Diffusion model for VSL transition generation.
    
    Architecture:
        - Condition encoder: Encodes start/end poses
        - Timestep embedding: Sinusoidal embedding for diffusion timestep
        - Transformer: Denoising network
        - Output projection: Maps back to VSL format with Tanh activation
    """
    
    def __init__(
        self,
        input_dim: int = 1659,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 30
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_frames = max_frames
        
        # Condition encoder (start + end poses) - improved with 3 layers
        self.condition_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
        
        # Timestep embedding
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim // 4)
        
        # Length embedding
        self.length_embed = nn.Embedding(
            num_embeddings=max_frames + 1,
            embedding_dim=hidden_dim // 4
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for frames
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_frames, hidden_dim) * 0.02
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection with Tanh to bound predictions to [-1, 1]
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Bound output to [-1, 1] - prevents unbounded predictions
        )
    
    def forward(
        self,
        noisy_data: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        target_length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, num_frames, _ = noisy_data.shape
        
        cond_emb = self.condition_encoder(condition)
        time_emb = self.time_embed(timesteps)
        
        if target_length is None:
            target_length = torch.full((batch_size,), num_frames, dtype=torch.long, device=noisy_data.device)
        length_emb = self.length_embed(target_length)
        
        padding = torch.zeros(batch_size, self.hidden_dim // 4, device=noisy_data.device)
        context = torch.cat([cond_emb, time_emb, length_emb, padding], dim=-1)
        context = context.unsqueeze(1)
        
        x = self.input_proj(noisy_data)
        x = x + self.pos_encoding[:, :num_frames, :]
        x = x + context
        
        for block in self.transformer_blocks:
            x = block(x)
        
        predicted_noise = self.output_proj(x)
        return predicted_noise
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout': self.dropout,
                'max_frames': self.max_frames
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get('config', {})
        
        model = cls(
            input_dim=config.get('input_dim', 1659),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 8),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            max_frames=config.get('max_frames', 30)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
