#!/usr/bin/env python3
"""
Stage 2: Train MicT Diffusion trên Latent Space
MicT V2 — Latent Diffusion (arXiv 2508.04049 + VAE)

Pipeline:
    1. Load frozen Pose VAE (Stage 1 checkpoint)
    2. Encode all sequences: 1659D → 128D latent
    3. Train MicT diffusion trên latent sequences (128D)

Usage:
    PYTHONPATH=. python -m src.methods.mict_v2.train_mict_v2 \
        --data_dir /mnt/ngan/vsl_data/mict \
        --ae_path models/mict_v2/autoencoder/best.pt \
        --output_dir models/mict_v2/diffusion \
        --latent_dim 128 \
        --epochs 150
"""

import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from tqdm import tqdm

from src.methods.mict.dataset_mict import MicTDataset, collate_fn_mict
from src.methods.mict_v2.autoencoder import PoseVAE
from src.methods.mict_v2.model_mict_v2 import MicTDiffusionModel, MicTDDPMScheduler


# ---------------------------------------------------------------------------
# Loss — giống mict/ nhưng trên latent space (128D, không cần weighted loss)
# ---------------------------------------------------------------------------

def latent_loss(pred_z0, target_z0, valid_mask):
    """MAE loss trên latent space — không cần weighted (latent space cân bằng hơn)."""
    mask_exp = valid_mask.unsqueeze(-1)
    n_valid  = mask_exp.sum().clamp(min=1.0) * pred_z0.shape[-1]
    total    = (pred_z0 - target_z0).abs().mul(mask_exp).sum() / n_valid
    return total, total.item()


# ---------------------------------------------------------------------------
# Train / Val loops
# ---------------------------------------------------------------------------

def encode_batch(vae, seqs, device):
    """Encode (B, T, 1659) → (B, T, latent_dim) — deterministic (mu only)."""
    with torch.no_grad():
        return vae.encode(seqs.to(device), deterministic=True)


def to_velocity(latent):
    """
    Convert absolute latent sequence → velocity sequence.
    v[0] = z[0] (first frame velocity = itself, anchored from 0)
    v[t] = z[t] - z[t-1]  for t > 0
    Shape: (B, T, D) → (B, T, D)
    """
    prev   = torch.cat([torch.zeros_like(latent[:, :1, :]), latent[:, :-1, :]], dim=1)
    return latent - prev


def from_velocity(velocity):
    """
    Integrate velocity → absolute latent: z[t] = cumsum(v[0..t])
    Shape: (B, T, D) → (B, T, D)
    """
    return torch.cumsum(velocity, dim=1)


def train_epoch(model, scheduler, vae, dataloader, optimizer, device, mask_prob=0.3,
                use_velocity=False):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="  [train]", leave=False)

    for masked_seqs, interp_seqs, trans_masks in pbar:
        masked_seqs  = masked_seqs.to(device)
        interp_seqs  = interp_seqs.to(device)
        trans_masks  = trans_masks.to(device)
        B, T, _      = interp_seqs.shape

        valid_mask   = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()
        pad_key_mask = ~valid_mask.bool()

        # Encode sequences to latent
        latent_masked = encode_batch(vae, masked_seqs, device)  # (B, T, D_lat)
        latent_gt     = encode_batch(vae, interp_seqs, device)  # (B, T, D_lat)

        # Convert to velocity if enabled
        if use_velocity:
            target    = to_velocity(latent_gt)
            obs_input = to_velocity(latent_masked)
        else:
            target    = latent_gt
            obs_input = latent_masked

        # Additional random masking on word frames (paper: 30%)
        word_mask  = (1.0 - trans_masks)
        extra      = (torch.rand(B, T, device=device) < mask_prob).float() * word_mask
        obs_input  = obs_input * (1.0 - extra.unsqueeze(-1))

        # Diffusion
        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
        x_t, _ = scheduler.add_noise(target, t)

        pred_z0, _ = model(x_t, t, obs_input, pad_key_mask)

        loss, mae_v = latent_loss(pred_z0, target, valid_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'L': f'{mae_v:.4f}'})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, scheduler, vae, dataloader, device, mask_prob=0.3, use_velocity=False):
    model.eval()
    total_loss = 0.0

    for masked_seqs, interp_seqs, trans_masks in dataloader:
        masked_seqs  = masked_seqs.to(device)
        interp_seqs  = interp_seqs.to(device)
        trans_masks  = trans_masks.to(device)
        B, T, _      = interp_seqs.shape

        valid_mask   = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()
        pad_key_mask = ~valid_mask.bool()

        latent_masked = encode_batch(vae, masked_seqs, device)
        latent_gt     = encode_batch(vae, interp_seqs, device)

        if use_velocity:
            target    = to_velocity(latent_gt)
            obs_input = to_velocity(latent_masked)
        else:
            target    = latent_gt
            obs_input = latent_masked

        word_mask  = (1.0 - trans_masks)
        extra      = (torch.rand(B, T, device=device) < mask_prob).float() * word_mask
        obs_input  = obs_input * (1.0 - extra.unsqueeze(-1))

        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
        x_t, _ = scheduler.add_noise(target, t)

        pred_z0, _ = model(x_t, t, obs_input, pad_key_mask)
        loss, _    = latent_loss(pred_z0, target, valid_mask)
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train MicT V2 Stage 2 (Latent Diffusion)')
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--ae_path',     required=True, help='Path to Stage 1 VAE checkpoint')
    parser.add_argument('--output_dir',  default='models/mict_v2/diffusion')
    parser.add_argument('--latent_dim',  type=int, default=128)
    parser.add_argument('--hidden_dim',  type=int, default=512)
    parser.add_argument('--num_heads',   type=int, default=8)
    parser.add_argument('--enc_layers',  type=int, default=4)
    parser.add_argument('--dec_layers',  type=int, default=6)
    parser.add_argument('--max_len',     type=int, default=512)
    parser.add_argument('--epochs',      type=int,   default=150)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=3e-5)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--patience',    type=int,   default=30)
    parser.add_argument('--mask_prob',   type=float, default=0.3)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--use_velocity', action='store_true',
                        help='Diffuse trên velocity latent thay vì absolute latent')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMicT V2 Stage 2 — {'Velocity+' if args.use_velocity else ''}Latent Diffusion ({ae_cfg['latent_dim']}D)")
    print(f"  Device:    {device}")
    print(f"  AE:        {args.ae_path}")

    # Load frozen VAE
    ae_ckpt = torch.load(args.ae_path, map_location=device)
    ae_cfg  = ae_ckpt['config']
    vae = PoseVAE(**ae_cfg).to(device)
    vae.load_state_dict(ae_ckpt['model_state_dict'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"  Loaded VAE (epoch {ae_ckpt['epoch']}, val={ae_ckpt['val_loss']:.4f})")

    # Data
    train_ds = MicTDataset(args.data_dir, split='train')
    val_ds   = MicTDataset(args.data_dir, split='val')
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn_mict,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate_fn_mict,
                               pin_memory=True)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # MicT Diffusion model — input_dim = latent_dim (128) instead of 1659
    model = MicTDiffusionModel(
        input_dim    = args.latent_dim,   # ← KEY DIFFERENCE: 128 not 1659
        hidden_dim   = args.hidden_dim,
        num_heads    = args.num_heads,
        enc_layers   = args.enc_layers,
        dec_layers   = args.dec_layers,
        ff_mult      = 4,
        dropout      = 0.05,
        max_len      = args.max_len,
        num_timesteps= args.num_timesteps,
    ).to(device)

    scheduler = MicTDDPMScheduler(num_timesteps=args.num_timesteps).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup_ep  = min(5, args.epochs // 10)
    warmup_sch = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_ep)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_ep),
                                    eta_min=args.lr * 0.01)
    lr_sched   = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup_ep])

    best_val = float('inf')
    patience_counter = 0
    history  = {'train_loss': [], 'val_loss': []}

    with open(output_dir / 'config.json', 'w') as f:
        json.dump({**vars(args), 'ae_config': ae_cfg}, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, scheduler, vae, train_loader,
                                  optimizer, device, args.mask_prob, args.use_velocity)
        val_loss   = validate(model, scheduler, vae, val_loader, device,
                               args.mask_prob, args.use_velocity)
        lr_sched.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train: {train_loss:.4f}  |  Val: {val_loss:.4f}  |  LR: {lr:.2e}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'val_loss': val_loss, 'config': model.config, 'ae_config': ae_cfg,
            'use_velocity': args.use_velocity,
        }, output_dir / 'latest.pt')

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': best_val, 'config': model.config, 'ae_config': ae_cfg,
                'use_velocity': args.use_velocity,
            }, output_dir / 'best.pt')
            print(f"  ✓ Best saved (val={best_val:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break

        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nStage 2 complete! Best val: {best_val:.4f}")


if __name__ == '__main__':
    main()
