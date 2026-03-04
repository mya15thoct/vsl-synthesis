#!/usr/bin/env python3
"""
Training Script for MicT - Motion is the Choreographer
Continuous Motion Transition Generation (arXiv 2508.04049)

Loss = L_joint (MAE trên joint positions, tất cả valid frames)
     L_joint = (1/S) Σ_s |p0_s - p̂0_s|     ← Eq. 8 paper
     (chỉ MAE, KHÔNG có MSE noise term)

Training masking (paper: "random mask mechanism with probability 0.3"):
  - Mỗi training step, mask ngẫu nhiên 30% frame bất kỳ → tạo observation condition
  - Khác với inference: dùng fixed transition zeros từ dataset

Usage (server):
    PYTHONPATH=. python src/methods/mict/train_mict.py \\
        --data_dir /mnt/ngan/vsl_data/mict \\
        --output_dir /mnt/ngan/vsl_models/mict \\
        --epochs 100 --batch_size 32

Usage (local test):
    PYTHONPATH=. python src/methods/mict/train_mict.py \\
        --data_dir /path/to/mict --epochs 5 --batch_size 4
"""

import argparse
import json
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from tqdm import tqdm

from .model_mict import MicTDiffusionModel, MicTDDPMScheduler
from .dataset_mict import MicTDataset, collate_fn_mict


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mict_loss(
    pred_x0: torch.Tensor,          # (B, T, D) — model output
    target: torch.Tensor,            # (B, T, D) — ground truth (interpolated_seq)
    valid_mask: torch.Tensor,        # (B, T)    — 1.0 = non-padding frame
) -> tuple:
    """
    L_joint (Eq. 8, MicT paper): MAE trên TẤT CẢ frames (cả word lẫn transition).
    Target = interpolated_sequence (word frames = real, transitions = linear interp).

    Weighted: hand + pose landmarks có weight cao hơn face (vì loss thường bị face dominate).
      Feat layout (flat, 3 per kp, no visibility):
        pose  [0:99]     (33 kp × 3)
        lhand [99:162]   (21 kp × 3)
        rhand [162:225]  (21 kp × 3)
        face  [225:1629] (468 kp × 3)
        extra [1629:1659]
    """
    # Build per-feature weight vector (1659,)
    # Real layout: pose(0:99) + face(99:1533, 478kp×3) + lhand(1533:1596) + rhand(1596:1659)
    w = torch.ones(pred_x0.shape[-1], device=pred_x0.device, dtype=pred_x0.dtype)
    w[0:99]      = 3.0   # pose: upweight ×3
    w[1533:1659] = 5.0   # hands (lhand+rhand): upweight ×5
    # face (99:1533) stays at 1.0

    mask_exp = valid_mask.unsqueeze(-1)                         # (B, T, 1)
    weighted_mae = (pred_x0 - target).abs() * w.unsqueeze(0).unsqueeze(0)
    n_valid  = mask_exp.sum().clamp(min=1.0) * pred_x0.shape[-1]
    total    = (weighted_mae * mask_exp).sum() / n_valid
    return total, total.item()


def apply_random_mask(
    full_seqs: torch.Tensor,   # (B, T, D)
    mask_prob: float = 0.3,
) -> tuple:
    """
    Paper: "random mask mechanism (with a probability of 0.3)"
    Mỗi training step, mask ngẫu nhiên 30% frame bất kỳ → zero out → observation condition.

    Returns:
        observation_seq: (B, T, D) — frames với masked positions = 0
        random_mask:     (B, T)    — 1.0 = masked frame
    """
    B, T, D = full_seqs.shape
    # Sample per-frame mask: Bernoulli(mask_prob)
    random_mask = (torch.rand(B, T, device=full_seqs.device) < mask_prob).float()  # (B, T)
    mask_exp = random_mask.unsqueeze(-1)                       # (B, T, 1)
    observation_seq = full_seqs * (1.0 - mask_exp)             # zero out masked frames
    return observation_seq, random_mask


# ---------------------------------------------------------------------------
# Training & validation
# ---------------------------------------------------------------------------

def train_epoch(
    model, scheduler, dataloader, optimizer, device, epoch,
    mask_prob: float = 0.3,
):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]")

    for masked_seqs, interp_seqs, trans_masks in pbar:
        # masked_seqs:  (B, T_total, D) — word frames + zero transitions (obs condition)
        # interp_seqs:  (B, T_total, D) — word frames + LINEAR INTERP transitions (GT)
        # trans_masks:  (B, T_total)    — 1 = transition frame

        masked_seqs = masked_seqs.to(device)   # (B, T_total, D) — observation
        interp_seqs = interp_seqs.to(device)   # (B, T_total, D) — ground truth
        trans_masks = trans_masks.to(device)   # (B, T_total)

        B, T_total, D = masked_seqs.shape

        # valid_mask = 1 on ALL non-padding frames (word + transition)
        # Padding frames (after the last word) = 0 in interp_seqs, trans_masks not set
        # Use non-zero frames as valid (padding positions remain 0 in interp_seqs)
        valid_mask = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()  # (B, T_total)

        # pad_key_mask: True = padding position (zeros added by collate)
        # interp_seqs has real values (word or interpolated transitions) for non-padded positions
        pad_key_mask = (interp_seqs.abs().sum(dim=-1) < 1e-6)  # (B, T_total) True=padded

        # obs_seq: masked_seqs + additional 30% masking on word frames
        word_mask = (1.0 - trans_masks)         # (B, T_total) 1=word
        extra = (torch.rand(B, T_total, device=device) < mask_prob).float() * word_mask
        obs_seq = masked_seqs * (1.0 - extra.unsqueeze(-1))

        # Add noise to interpolated_seq (T_total) — ground truth target
        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
        x_t, _ = scheduler.add_noise(interp_seqs, t)

        # Model predicts clean interp_seqs
        pred_x0, _ = model(x_t, t, obs_seq, pad_key_mask)

        # Loss on ALL frames (word + transition), skip padding
        loss, mae_v = mict_loss(pred_x0, interp_seqs, valid_mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'L_joint': f"{mae_v:.4f}"})

    return total_loss / len(dataloader)


def validate(
    model, scheduler, dataloader, device,
    mask_prob: float = 0.3,
):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for masked_seqs, interp_seqs, trans_masks in tqdm(dataloader, desc="Validating"):
            masked_seqs = masked_seqs.to(device)
            interp_seqs = interp_seqs.to(device)
            trans_masks = trans_masks.to(device)

            B, T_total, D = masked_seqs.shape
            valid_mask = (interp_seqs.abs().sum(dim=-1) > 1e-6).float()
            pad_key_mask = ~valid_mask.bool()   # True = padded position

            word_mask = (1.0 - trans_masks)
            extra = (torch.rand(B, T_total, device=device) < mask_prob).float() * word_mask
            obs_seq = masked_seqs * (1.0 - extra.unsqueeze(-1))

            t = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
            x_t, _ = scheduler.add_noise(interp_seqs, t)

            pred_x0, _ = model(x_t, t, obs_seq, pad_key_mask)
            loss, _ = mict_loss(pred_x0, interp_seqs, valid_mask)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train MicT Transition Model')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Thư mục chứa train/ và val/ .npz files (từ prepare_data_mict.py)')
    parser.add_argument('--output_dir', type=str, default='models/mict',
                        help='Nơi lưu checkpoints')

    # Model
    parser.add_argument('--hidden_dim',  type=int, default=512)
    parser.add_argument('--num_heads',   type=int, default=8)
    parser.add_argument('--enc_layers',  type=int, default=4,
                        help='ObservationEncoder transformer layers')
    parser.add_argument('--dec_layers',  type=int, default=6,
                        help='Denoiser transformer layers')
    parser.add_argument('--max_len',     type=int, default=512,
                        help='Max sequence length for positional encoding')

    # Training
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=3e-5)
    parser.add_argument('--weight_decay',type=float, default=0.01)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--max_seq_len', type=int,   default=300,
                        help='Max frame length per sample (dataset truncation)')
    parser.add_argument('--patience',    type=int,   default=30,
                        help='Early stopping patience (0 = disabled)')

    # Loss & masking
    parser.add_argument('--mask_prob',   type=float, default=0.3,
                        help='Random masking probability during training (paper: 0.3)')

    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=1000)

    # Misc
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed',   type=int, default=42)

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nMicT Training (arXiv 2508.04049)")
    print(f"  Device:        {device}")
    print(f"  Data:          {args.data_dir}")
    print(f"  Output:        {args.output_dir}")
    print(f"  Mask prob:     {args.mask_prob} (paper: 0.3)")
    print(f"  Loss:          L_joint MAE (all valid frames, Eq. 8)")
    print(f"  Schedule:      cosine")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    print("\nLoading datasets...")
    train_dataset = MicTDataset(args.data_dir, split='train', max_len=args.max_seq_len)
    val_dataset   = MicTDataset(args.data_dir, split='val',   max_len=args.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn_mict,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_mict,
        pin_memory=True,
    )
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print("\nCreating model...")
    model = MicTDiffusionModel(
        input_dim    = 1659,
        hidden_dim   = args.hidden_dim,
        num_heads    = args.num_heads,
        enc_layers   = args.enc_layers,
        dec_layers   = args.dec_layers,
        ff_mult      = 4,
        dropout      = 0.05,   # Fix 4: lower dropout (was 0.1)
        max_len      = args.max_len,
        num_timesteps= args.num_timesteps,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    # DDPM scheduler — cosine schedule (paper)
    scheduler = MicTDDPMScheduler(
        num_timesteps = args.num_timesteps,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Fix 3: Warmup 5 epochs → Cosine decay
    warmup_epochs  = min(5, args.epochs // 10)
    warmup_sched   = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                               total_iters=warmup_epochs)
    cosine_sched   = CosineAnnealingLR(optimizer,
                                        T_max=max(1, args.epochs - warmup_epochs),
                                        eta_min=args.lr * 0.01)
    lr_scheduler   = SequentialLR(optimizer,
                                   schedulers=[warmup_sched, cosine_sched],
                                   milestones=[warmup_epochs])

    # Resume if requested
    start_epoch = 1
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch  = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}")

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    print(f"\nStarting training for {args.epochs} epochs...")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(
            model, scheduler, train_loader, optimizer, device, epoch,
            args.mask_prob,
        )
        val_loss = validate(
            model, scheduler, val_loader, device,
            args.mask_prob,
        )
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train L_joint: {train_loss:.4f}  |  Val L_joint: {val_loss:.4f}")
        print(f"  LR: {current_lr:.2e}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Checkpoint: latest every epoch
        latest_path = output_dir / 'latest.pt'
        torch.save({
            'epoch':               epoch,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'val_loss':            val_loss,
            'config':              model.config,
        }, latest_path)

        # Checkpoint: best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = output_dir / 'best.pt'
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':         best_val_loss,
                'config':           model.config,
            }, best_path)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})" if args.patience > 0 else "")

        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n⚠ Early stopping after {epoch} epochs. Best val: {best_val_loss:.4f}")
            break

    print(f"\nTraining complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints:   {output_dir}/best.pt")


if __name__ == "__main__":
    main()
