#!/usr/bin/env python3
"""
Stage 1: Train Pose VAE (Autoencoder)
MicT V2 — Latent Diffusion

Dùng lại data đã prepare từ mict/ (cùng .npz format).

Usage:
    PYTHONPATH=. python -m src.methods.mict_v2.train_autoencoder \
        --data_dir /mnt/ngan/vsl_data/mict \
        --output_dir models/mict_v2/autoencoder \
        --latent_dim 128 \
        --epochs 50 \
        --batch_size 64
"""

import argparse
import json
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.methods.mict.dataset_mict import MicTDataset, collate_fn_mict
from src.methods.mict_v2.autoencoder import PoseVAE, vae_loss


def train_epoch(model, dataloader, optimizer, device, kl_weight):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    pbar = tqdm(dataloader, desc="  [train]", leave=False)

    for masked_seqs, interp_seqs, trans_masks in pbar:
        interp_seqs = interp_seqs.to(device)   # (B, T, 1659) — GT target
        valid_mask = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()

        recon, mu, log_var = model(interp_seqs)
        loss, recon_v, kl_v = vae_loss(recon, interp_seqs, mu, log_var,
                                        valid_mask, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_recon += recon_v
        total_kl    += kl_v
        pbar.set_postfix({'recon': f'{recon_v:.4f}', 'kl': f'{kl_v:.4f}'})

    n = len(dataloader)
    return total_loss/n, total_recon/n, total_kl/n


@torch.no_grad()
def validate(model, dataloader, device, kl_weight):
    model.eval()
    total_loss = total_recon = 0.0

    for masked_seqs, interp_seqs, trans_masks in dataloader:
        interp_seqs = interp_seqs.to(device)
        valid_mask = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()
        recon, mu, log_var = model(interp_seqs)
        loss, recon_v, _ = vae_loss(recon, interp_seqs, mu, log_var,
                                     valid_mask, kl_weight)
        total_loss  += loss.item()
        total_recon += recon_v

    n = len(dataloader)
    return total_loss/n, total_recon/n


def main():
    parser = argparse.ArgumentParser(description='Train Pose VAE (MicT V2 Stage 1)')
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--output_dir',  default='models/mict_v2/autoencoder')
    parser.add_argument('--latent_dim',  type=int,   default=128)
    parser.add_argument('--hidden_dim',  type=int,   default=512)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--kl_weight',   type=float, default=0.001,
                        help='Beta for KL divergence (default: 0.001)')
    parser.add_argument('--kl_warmup',   type=int,   default=10,
                        help='Epochs to linearly warm up kl_weight from 0 to kl_weight')
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--patience',    type=int,   default=15)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPose VAE Training (MicT V2 Stage 1)")
    print(f"  Device:     {device}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  KL weight:  {args.kl_weight} (warmup {args.kl_warmup} epochs)")

    # Data — reuse mict dataset
    train_ds = MicTDataset(args.data_dir, split='train')
    val_ds   = MicTDataset(args.data_dir, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn_mict,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate_fn_mict,
                               pin_memory=True)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = PoseVAE(input_dim=1659, latent_dim=args.latent_dim,
                    hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    best_val  = float('inf')
    patience_counter = 0
    history   = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        # KL warmup: linearly increase kl_weight from 0 → target over kl_warmup epochs
        kw = args.kl_weight * min(1.0, epoch / max(1, args.kl_warmup))

        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, kw)
        val_loss, val_recon = validate(model, val_loader, device, kw)
        lr_scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.epochs}  |  KL_w={kw:.4f}  LR={lr:.2e}")
        print(f"  Train: loss={train_loss:.4f}  recon={train_recon:.4f}  kl={train_kl:.4f}")
        print(f"  Val:   loss={val_loss:.4f}  recon={val_recon:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Save latest
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_loss': val_loss, 'config': model.config},
                   output_dir / 'latest.pt')

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': best_val, 'config': model.config},
                       output_dir / 'best.pt')
            print(f"  ✓ Best saved (val={best_val:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break

        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nStage 1 complete! Best val: {best_val:.4f}")
    print(f"  Checkpoint: {output_dir}/best.pt")


if __name__ == '__main__':
    main()
