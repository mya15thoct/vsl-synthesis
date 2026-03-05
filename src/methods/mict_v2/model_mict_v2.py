#!/usr/bin/env python3
"""
MicT V2 — Latent Diffusion Model
arXiv 2508.04049 — Motion is the Choreographer

Kiến trúc:
  input_dim = latent_dim (128) thay vì 1659D raw pose
  Architecture: ObservationEncoder + DenoiserBlock
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Positional Encoding (sinusoidal, non-learned)
# ---------------------------------------------------------------------------

class SinPosEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:hidden_dim // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, max_steps: int = 1000):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# ObservationEncoder
# ---------------------------------------------------------------------------

class ObservationEncoder(nn.Module):
    def __init__(self, input_dim=1659, hidden_dim=512, num_heads=8,
                 num_layers=4, ff_mult=4, dropout=0.1, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc    = SinPosEncoding(hidden_dim, max_len, dropout)
        self.nl         = nn.LayerNorm(hidden_dim)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, masked_seq: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(masked_seq)
        x = self.pos_enc(x)
        x = self.nl(x)
        return self.encoder(x, src_key_padding_mask=padding_mask)


# ---------------------------------------------------------------------------
# DenoiserBlock
# ---------------------------------------------------------------------------

class DenoiserBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_self  = nn.LayerNorm(hidden_dim)
        self.self_attn  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff    = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mc: torch.Tensor,
                self_attn_mask=None, context_key_padding_mask=None,
                self_key_padding_mask=None) -> torch.Tensor:
        h = self.norm_self(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=self_attn_mask,
                                     key_padding_mask=self_key_padding_mask)
        x = x + attn_out
        h = self.norm_cross(x)
        cross_out, _ = self.cross_attn(h, mc, mc, key_padding_mask=context_key_padding_mask)
        x = x + cross_out
        x = x + self.ff(self.norm_ff(x))
        return x


# ---------------------------------------------------------------------------
# MicTDiffusionModel
# ---------------------------------------------------------------------------

class MicTDiffusionModel(nn.Module):
    def __init__(self, input_dim=1659, hidden_dim=512, num_heads=8,
                 enc_layers=4, dec_layers=6, ff_mult=4, dropout=0.1,
                 max_len=512, num_timesteps=1000):
        super().__init__()
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.num_timesteps = num_timesteps

        self.obs_encoder = ObservationEncoder(input_dim, hidden_dim, num_heads,
                                              enc_layers, ff_mult, dropout, max_len)
        self.noisy_proj  = nn.Linear(input_dim, hidden_dim)
        self.pos_enc     = SinPosEncoding(hidden_dim, max_len, dropout)
        self.time_embed  = TimestepEmbedding(hidden_dim)
        self.time_proj   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dec_layers  = nn.ModuleList([
            DenoiserBlock(hidden_dim, num_heads, ff_mult, dropout) for _ in range(dec_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noisy_seq, timesteps, masked_seq,
                padding_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        mc      = self.obs_encoder(masked_seq, padding_mask)
        x       = self.noisy_proj(noisy_seq)
        x       = self.pos_enc(x)
        t_emb   = self.time_proj(self.time_embed(timesteps))
        x       = x + t_emb.unsqueeze(1)
        for block in self.dec_layers:
            x = block(x, mc, context_key_padding_mask=padding_mask,
                      self_key_padding_mask=padding_mask)
        pred_x0 = self.out_proj(self.out_norm(x))
        return pred_x0, mc

    @property
    def config(self):
        return {
            'input_dim':     self.input_dim,
            'hidden_dim':    self.hidden_dim,
            'num_heads':     self.obs_encoder.encoder.layers[0].self_attn.num_heads,
            'enc_layers':    len(self.obs_encoder.encoder.layers),
            'dec_layers':    len(self.dec_layers),
            'num_timesteps': self.num_timesteps,
        }


# ---------------------------------------------------------------------------
# MicTDDPMScheduler
# ---------------------------------------------------------------------------

class MicTDDPMScheduler:
    def __init__(self, num_timesteps: int = 1000, s: float = 0.008):
        self.num_timesteps = num_timesteps
        steps    = num_timesteps + 1
        t_vals   = torch.linspace(0, num_timesteps, steps)
        f        = torch.cos(((t_vals / num_timesteps + s) / (1 + s)) * math.pi / 2) ** 2
        ab       = (f / f[0])[1:].clamp(min=1e-5)
        ab_prev  = torch.cat([torch.ones(1), ab[:-1]])
        betas    = (1 - ab / ab_prev).clamp(0, 0.999)
        alphas   = 1.0 - betas
        self.register('betas',                          betas)
        self.register('alphas',                         alphas)
        self.register('alphas_cumprod',                 ab)
        self.register('sqrt_alphas_cumprod',            ab.sqrt())
        self.register('sqrt_one_minus_alphas_cumprod', (1 - ab).sqrt())

    def register(self, name, tensor): setattr(self, name, tensor)

    def to(self, device):
        for attr in ['betas', 'alphas', 'alphas_cumprod',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        sqrt_ac  = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_omc = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ac * x0 + sqrt_omc * noise, noise

    @torch.no_grad()
    def ddpm_step(self, model, x_t, t_int, masked_seq, padding_mask=None):
        device   = x_t.device
        B        = x_t.shape[0]
        t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)
        pred_x0, _ = model(x_t, t_tensor, masked_seq, padding_mask)
        flat     = pred_x0.reshape(B, -1).abs()
        s        = torch.quantile(flat, 0.995, dim=1).clamp(min=1.0).view(B, 1, 1)
        pred_x0  = (pred_x0.clamp(-s, s) / s).clamp(0.0, 1.0)
        if t_int == 0: return pred_x0
        ab_t    = self.alphas_cumprod[t_int]
        ab_prev = self.alphas_cumprod[t_int - 1]
        beta_t  = self.betas[t_int]
        mean    = (ab_prev.sqrt() * beta_t / (1 - ab_t)) * pred_x0 + \
                  (self.alphas[t_int].sqrt() * (1 - ab_prev) / (1 - ab_t)) * x_t
        var     = beta_t * (1 - ab_prev) / (1 - ab_t)
        noise   = torch.randn_like(x_t) if t_int > 1 else torch.zeros_like(x_t)
        return mean + var.sqrt() * noise

    @torch.no_grad()
    def sample(self, model, masked_seq, num_inference_steps=50, padding_mask=None):
        device   = masked_seq.device
        B, T, D  = masked_seq.shape
        x_t      = torch.randn(B, T, D, device=device)
        I, N     = num_inference_steps, self.num_timesteps
        ts       = [int(N - 1 - (N - 1) * i / (I - 1)) for i in range(I)]
        ts[-1]   = 0
        for t_int in ts:
            x_t = self.ddpm_step(model, x_t, t_int, masked_seq, padding_mask)
        return x_t.clamp(0.0, 1.0)


__all__ = [
    'MicTDiffusionModel', 'MicTDDPMScheduler',
    'SinPosEncoding', 'ObservationEncoder', 'DenoiserBlock',
]
