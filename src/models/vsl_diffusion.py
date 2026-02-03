"""
VSL-Native Diffusion Model for Sign Language Transition Generation

This module implements a transformer-based diffusion model trained directly
on VSL skeleton format (1662 features) for generating smooth transitions.
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
        """
        Args:
            timesteps: (batch_size,) tensor of timestep indices
            
        Returns:
            (batch_size, dim) embeddings
        """
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
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            
        Returns:
            (batch, seq_len, hidden_dim)
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
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
        - Output projection: Maps back to VSL format
    
    Input:
        - noisy_data: (batch, num_frames, 1662) noisy skeleton sequence
        - timesteps: (batch,) diffusion timestep indices
        - condition: (batch, 1662*2) concatenated start/end poses
        
    Output:
        - predicted_noise: (batch, num_frames, 1662) predicted noise
    """
    
    def __init__(
        self,
        input_dim: int = 1662,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 30
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        
        # Condition encoder (start + end poses)
        self.condition_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Timestep embedding
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim // 2)
        
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
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(
        self,
        noisy_data: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for denoising.
        
        Args:
            noisy_data: (batch, num_frames, 1662) noisy skeleton sequence
            timesteps: (batch,) timestep indices [0, 999]
            condition: (batch, 1662*2) concatenated start/end poses
            
        Returns:
            predicted_noise: (batch, num_frames, 1662)
        """
        batch_size, num_frames, _ = noisy_data.shape
        
        # Encode condition (start + end poses)
        cond_emb = self.condition_encoder(condition)  # (batch, hidden_dim//2)
        
        # Encode timestep
        time_emb = self.time_embed(timesteps)  # (batch, hidden_dim//2)
        
        # Combine condition and time embeddings
        context = torch.cat([cond_emb, time_emb], dim=-1)  # (batch, hidden_dim)
        context = context.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Project input
        x = self.input_proj(noisy_data)  # (batch, num_frames, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :num_frames, :]
        
        # Add context to each frame
        x = x + context
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Project to output
        predicted_noise = self.output_proj(x)  # (batch, num_frames, 1662)
        
        return predicted_noise
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'max_frames': self.max_frames
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            max_frames=config['max_frames']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model


if __name__ == "__main__":
    # Test model
    print("Testing VSL Diffusion Model...")
    
    batch_size = 4
    num_frames = 18
    input_dim = 1662
    
    # Create dummy inputs
    noisy_data = torch.randn(batch_size, num_frames, input_dim)
    timesteps = torch.randint(0, 1000, (batch_size,))
    start_pose = torch.randn(batch_size, input_dim)
    end_pose = torch.randn(batch_size, input_dim)
    condition = torch.cat([start_pose, end_pose], dim=-1)
    
    # Create model
    model = VSLDiffusionModel(
        input_dim=input_dim,
        hidden_dim=512,
        num_layers=8,
        num_heads=8
    )
    
    # Forward pass
    predicted_noise = model(noisy_data, timesteps, condition)
    
    print(f"✅ Model test passed!")
    print(f"  Input shape: {noisy_data.shape}")
    print(f"  Output shape: {predicted_noise.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
