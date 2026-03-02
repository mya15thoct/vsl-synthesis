#!/usr/bin/env python3
"""
MicT V2 — Latent + Velocity Diffusion
Train Stage 1 (Pose VAE) + Stage 2 (Diffusion) trong 1 script

Khác với mict/:
    - Stage 1: Train Pose VAE (1659D → latent_dim)
    - Stage 2: Train MicT Diffusion trên latent space
    - Option: --use_velocity → diffuse v[t]=z[t]-z[t-1] thay vì z[t]

Usage:
    PYTHONPATH=. python -m src.methods.mict_v2.train_mict_v2 \\
        --data_dir /mnt/ngan/vsl_data/mict \\
        --output_dir ~/mya/vsl-synthesis/models/mict_v2 \\
        --use_velocity
"""

import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from tqdm import tqdm

from src.methods.mict.dataset_mict import MicTDataset, collate_fn_mict
from src.methods.mict_v2.autoencoder import PoseVAE, vae_loss
from src.methods.mict_v2.model_mict_v2 import MicTDiffusionModel, MicTDDPMScheduler


# ---------------------------------------------------------------------------
# Velocity helpers
# ---------------------------------------------------------------------------

def to_velocity(latent):
    """(B,T,D) absolute → velocity: v[t] = z[t] - z[t-1], v[0] = z[0]"""
    prev = torch.cat([torch.zeros_like(latent[:, :1, :]), latent[:, :-1, :]], dim=1)
    return latent - prev


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def latent_loss(pred, target, valid_mask):
    mask_exp = valid_mask.unsqueeze(-1)
    n_valid  = mask_exp.sum().clamp(min=1.0) * pred.shape[-1]
    return (pred - target).abs().mul(mask_exp).sum() / n_valid


# ---------------------------------------------------------------------------
# Stage 1: Train Pose VAE
# ---------------------------------------------------------------------------

def train_ae_epoch(vae, loader, optimizer, device, kl_weight):
    vae.train()
    total = 0.0
    for _, interp_seqs, _ in tqdm(loader, desc="  AE [train]", leave=False):
        interp_seqs = interp_seqs.to(device)
        valid_mask  = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()
        recon, mu, log_var = vae(interp_seqs)
        loss, _, _ = vae_loss(recon, interp_seqs, mu, log_var, valid_mask, kl_weight)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def val_ae(vae, loader, device, kl_weight):
    vae.eval()
    total = 0.0
    for _, interp_seqs, _ in loader:
        interp_seqs = interp_seqs.to(device)
        valid_mask  = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()
        recon, mu, log_var = vae(interp_seqs)
        loss, _, _ = vae_loss(recon, interp_seqs, mu, log_var, valid_mask, kl_weight)
        total += loss.item()
    return total / len(loader)


def train_stage1(args, train_loader, val_loader, device):
    """Train Pose VAE — Stage 1."""
    ae_dir = Path(args.output_dir) / 'autoencoder'
    ae_dir.mkdir(parents=True, exist_ok=True)
    ae_ckpt = ae_dir / 'best.pt'

    print(f"\n{'='*55}")
    print(f"  Stage 1/2: Pose VAE (1659D → {args.latent_dim}D)")
    print(f"{'='*55}")

    vae = PoseVAE(input_dim=1659, latent_dim=args.latent_dim,
                  hidden_dim=args.ae_hidden).to(device)
    opt = AdamW(vae.parameters(), lr=args.ae_lr, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=args.ae_epochs, eta_min=args.ae_lr*0.01)

    best_val = float('inf'); patience_ctr = 0
    for epoch in range(1, args.ae_epochs + 1):
        kw = args.kl_weight * min(1.0, epoch / max(1, args.kl_warmup))
        train_loss = train_ae_epoch(vae, train_loader, opt, device, kw)
        val_loss   = val_ae(vae, val_loader, device, kw)
        sched.step()
        print(f"  AE Epoch {epoch:3d}/{args.ae_epochs} | train={train_loss:.4f} val={val_loss:.4f} kl_w={kw:.4f}")
        if val_loss < best_val:
            best_val = val_loss; patience_ctr = 0
            torch.save({'epoch': epoch, 'model_state_dict': vae.state_dict(),
                        'val_loss': best_val, 'config': vae.config}, ae_ckpt)
            print(f"  ✓ AE best saved")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  ⚠ AE early stop epoch {epoch}"); break

    print(f"\nStage 1 done. Best val={best_val:.4f}  →  {ae_ckpt}")
    return vae, ae_ckpt


# ---------------------------------------------------------------------------
# Stage 2: Train MicT Diffusion on latent
# ---------------------------------------------------------------------------

def encode_batch(vae, seqs, device):
    with torch.no_grad():
        return vae.encode(seqs.to(device), deterministic=True)


def train_diff_epoch(model, scheduler, vae, loader, optimizer, device,
                     mask_prob, use_velocity):
    model.train()
    total = 0.0
    for masked_seqs, interp_seqs, trans_masks in tqdm(loader, desc="  Diff [train]", leave=False):
        masked_seqs = masked_seqs.to(device)
        interp_seqs = interp_seqs.to(device)
        trans_masks = trans_masks.to(device)
        B, T, _     = interp_seqs.shape
        valid_mask  = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()
        pad_kv      = ~valid_mask.bool()

        lm = encode_batch(vae, masked_seqs, device)
        lg = encode_batch(vae, interp_seqs, device)

        target    = to_velocity(lg) if use_velocity else lg
        obs_input = to_velocity(lm) if use_velocity else lm

        word_mask = (1.0 - trans_masks)
        extra     = (torch.rand(B, T, device=device) < mask_prob).float() * word_mask
        obs_input = obs_input * (1.0 - extra.unsqueeze(-1))

        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
        x_t, _ = scheduler.add_noise(target, t)
        pred, _ = model(x_t, t, obs_input, pad_kv)

        loss = latent_loss(pred, target, valid_mask)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_diff(model, scheduler, vae, loader, device, mask_prob, use_velocity):
    model.eval()
    total = 0.0
    for masked_seqs, interp_seqs, trans_masks in loader:
        masked_seqs = masked_seqs.to(device)
        interp_seqs = interp_seqs.to(device)
        trans_masks = trans_masks.to(device)
        B, T, _    = interp_seqs.shape
        valid_mask = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()
        pad_kv     = ~valid_mask.bool()

        lm = encode_batch(vae, masked_seqs, device)
        lg = encode_batch(vae, interp_seqs, device)

        target    = to_velocity(lg) if use_velocity else lg
        obs_input = to_velocity(lm) if use_velocity else lm
        word_mask = (1.0 - trans_masks)
        extra     = (torch.rand(B, T, device=device) < mask_prob).float() * word_mask
        obs_input = obs_input * (1.0 - extra.unsqueeze(-1))

        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
        x_t, _ = scheduler.add_noise(target, t)
        pred, _ = model(x_t, t, obs_input, pad_kv)
        total  += latent_loss(pred, target, valid_mask).item()
    return total / len(loader)


def train_stage2(args, vae, ae_ckpt, train_loader, val_loader, device):
    """Train MicT Diffusion on latent — Stage 2."""
    diff_dir = Path(args.output_dir) / 'diffusion'
    diff_dir.mkdir(parents=True, exist_ok=True)

    mode = 'Velocity+Latent' if args.use_velocity else 'Latent'
    print(f"\n{'='*55}")
    print(f"  Stage 2/2: {mode} Diffusion ({args.latent_dim}D)")
    print(f"{'='*55}")

    model = MicTDiffusionModel(
        input_dim    = args.latent_dim,
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
    print(f"  Diffusion params: {n_params:,}")

    optimizer  = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup_ep  = min(5, args.epochs // 10)
    lr_sched   = SequentialLR(optimizer, [
        LinearLR(optimizer, 0.1, 1.0, warmup_ep),
        CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_ep), eta_min=args.lr*0.01)
    ], milestones=[warmup_ep])

    ae_cfg  = torch.load(ae_ckpt, map_location='cpu')['config']
    best_val = float('inf'); patience_ctr = 0; history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_diff_epoch(model, scheduler, vae, train_loader,
                                       optimizer, device, args.mask_prob, args.use_velocity)
        val_loss   = val_diff(model, scheduler, vae, val_loader,
                               device, args.mask_prob, args.use_velocity)
        lr_sched.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{args.epochs} | train={train_loss:.4f} val={val_loss:.4f} lr={lr:.2e}")
        history.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})
        ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': val_loss, 'config': model.config, 'ae_config': ae_cfg,
                'use_velocity': args.use_velocity}
        torch.save(ckpt, diff_dir / 'latest.pt')
        if val_loss < best_val:
            best_val = val_loss; patience_ctr = 0
            torch.save(ckpt, diff_dir / 'best.pt')
            print(f"  ✓ Diff best saved")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  ⚠ Diff early stop epoch {epoch}"); break
        with open(diff_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nStage 2 done. Best val={best_val:.4f}  →  {diff_dir}/best.pt")
    return diff_dir / 'best.pt'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MicT V2 Training (Stage 1 + Stage 2)')

    # Data
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--output_dir',  default='models/mict_v2')
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--max_seq_len', type=int,   default=300)

    # Stage 1: Pose VAE
    parser.add_argument('--latent_dim',  type=int,   default=128)
    parser.add_argument('--ae_hidden',   type=int,   default=512)
    parser.add_argument('--ae_epochs',   type=int,   default=50)
    parser.add_argument('--ae_lr',       type=float, default=1e-4)
    parser.add_argument('--ae_batch',    type=int,   default=64)
    parser.add_argument('--kl_weight',   type=float, default=0.001)
    parser.add_argument('--kl_warmup',   type=int,   default=10)

    # Stage 2: Diffusion
    parser.add_argument('--hidden_dim',  type=int,   default=512)
    parser.add_argument('--num_heads',   type=int,   default=8)
    parser.add_argument('--enc_layers',  type=int,   default=4)
    parser.add_argument('--dec_layers',  type=int,   default=6)
    parser.add_argument('--max_len',     type=int,   default=512)
    parser.add_argument('--epochs',      type=int,   default=150)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=3e-5)
    parser.add_argument('--mask_prob',   type=float, default=0.3)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--use_velocity', action='store_true',
                        help='Diffuse trên velocity latent (v[t]=z[t]-z[t-1])')

    # Shared
    parser.add_argument('--patience',    type=int,   default=30)
    parser.add_argument('--seed',        type=int,   default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode = 'Velocity+Latent' if args.use_velocity else 'Latent'
    print(f"\nMicT V2 — {mode} Diffusion")
    print(f"  Device:     {device}")
    print(f"  Data:       {args.data_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Latent dim: {args.latent_dim}")

    # Shared data loaders
    train_ds = MicTDataset(args.data_dir, split='train', max_len=args.max_seq_len)
    val_ds   = MicTDataset(args.data_dir, split='val',   max_len=args.max_seq_len)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    ae_loader = DataLoader(train_ds, args.ae_batch, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn_mict,
                            pin_memory=True, drop_last=True)
    ae_val_loader = DataLoader(val_ds, args.ae_batch, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn_mict,
                                pin_memory=True)
    diff_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn_mict,
                              pin_memory=True, drop_last=True)
    diff_val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, collate_fn=collate_fn_mict,
                                  pin_memory=True)

    # Stage 1
    vae, ae_ckpt = train_stage1(args, ae_loader, ae_val_loader, device)

    # Freeze VAE for Stage 2
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Stage 2
    diff_ckpt = train_stage2(args, vae, ae_ckpt, diff_loader, diff_val_loader, device)

    print(f"\n{'='*55}")
    print(f"✓ MicT V2 Training Complete!")
    print(f"  VAE:       {ae_ckpt}")
    print(f"  Diffusion: {diff_ckpt}")
    print(f"\nInference:")
    print(f"  PYTHONPATH=. python -m src.methods.mict_v2.infer_mict_v2 \\")
    print(f"      --ae_path {ae_ckpt} \\")
    print(f"      --model_path {diff_ckpt} \\")
    print(f"      --word_dirs <word1_dir> <word2_dir>")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
