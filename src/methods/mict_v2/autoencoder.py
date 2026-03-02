#!/usr/bin/env python3
"""
Pose VAE — Stage 1 of MicT V2 (Latent Diffusion)

Architecture:
    Encoder: (T, 1659) → (T, latent_dim)   via MLP per-frame
    Decoder: (T, latent_dim) → (T, 1659)   via MLP per-frame

Usage:
    PYTHONPATH=. python src/methods/mict_v2/train_autoencoder.py \
        --data_dir /mnt/ngan/vsl_data/mict \
        --output_dir models/mict_v2/autoencoder \
        --latent_dim 128 \
        --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseEncoder(nn.Module):
    """
    Per-frame MLP encoder: (B, T, 1659) → (B, T, latent_dim)
    """
    def __init__(self, input_dim: int = 1659, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2),  # mu + log_var
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        """x: (B, T, input_dim) → mu (B, T, latent_dim), log_var (B, T, latent_dim)"""
        out = self.net(x)  # (B, T, latent_dim*2)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var


class PoseDecoder(nn.Module):
    """
    Per-frame MLP decoder: (B, T, latent_dim) → (B, T, input_dim)
    """
    def __init__(self, latent_dim: int = 128, output_dim: int = 1659, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        """z: (B, T, latent_dim) → (B, T, output_dim)"""
        return self.net(z)


class PoseVAE(nn.Module):
    """
    Variational Autoencoder cho pose sequences.
    Encoder/Decoder đều per-frame MLP (không Transformer) để:
        - Đơn giản, nhanh train
        - Mỗi frame được encode độc lập → dễ dùng trong diffusion
    """
    def __init__(self, input_dim: int = 1659, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.encoder = PoseEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = PoseDecoder(latent_dim, input_dim, hidden_dim)
        self.latent_dim = latent_dim
        self.input_dim  = input_dim
        self.config = {
            'input_dim':  input_dim,
            'latent_dim': latent_dim,
            'hidden_dim': hidden_dim,
        }

    def reparameterize(self, mu, log_var):
        """Sample z = mu + eps * std via reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x: (B, T, 1659)
        Returns: recon (B, T, 1659), mu (B, T, D), log_var (B, T, D)
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def encode(self, x, deterministic=True):
        """x: (B, T, 1659) → z: (B, T, latent_dim)"""
        mu, log_var = self.encoder(x)
        if deterministic:
            return mu
        return self.reparameterize(mu, log_var)

    def decode(self, z):
        """z: (B, T, latent_dim) → x: (B, T, 1659)"""
        return self.decoder(z)


def vae_loss(recon, target, mu, log_var, valid_mask, kl_weight=0.001):
    """
    VAE loss = Reconstruction (MSE) + β * KL divergence

    kl_weight: β — tradeoff giữa reconstruction fidelity vs latent space regularity
                Thường dùng warmup từ 0 → 0.001 trong training
    """
    # Reconstruction loss (MSE) — chỉ trên valid frames
    mask_exp = valid_mask.unsqueeze(-1)           # (B, T, 1)
    n_valid  = mask_exp.sum().clamp(min=1.0) * recon.shape[-1]
    recon_loss = ((recon - target) ** 2 * mask_exp).sum() / n_valid

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp())  # (B, T, D)
    kl_loss = (kl * mask_exp).sum() / mask_exp.sum().clamp(min=1.0)

    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss.item(), kl_loss.item()
