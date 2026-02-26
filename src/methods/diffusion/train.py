#!/usr/bin/env python3
"""
Training Script for VSL Diffusion Model

Usage:
    python -m src.methods.diffusion.train --data_dir /mnt/ngan/vsl_data/diffusion
    # OR from project root:
    python src/methods/diffusion/train.py --data_dir /mnt/ngan/vsl_data/diffusion
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import json

from .model import VSLDiffusionModel
from .dataset import VSLTransitionDataset, collate_fn
from .scheduler import SimpleDDPMScheduler
from src.core.skeleton_utils import combined_constraint_loss


def compute_metrics(predicted, target, mask=None):
    if mask is not None:
        mask = mask.unsqueeze(-1)
        predicted = predicted * mask
        target = target * mask
        num_valid = mask.sum() * predicted.shape[-1]
    else:
        num_valid = predicted.numel()
    
    mse = F.mse_loss(predicted, target, reduction='sum') / num_valid
    pred_vel = predicted[:, 1:] - predicted[:, :-1]
    smoothness = pred_vel.var()
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    jerk = pred_acc.var()
    
    return {'mse': mse.item(), 'smoothness': smoothness.item(), 'jerk': jerk.item()}


def train_epoch(model, dataloader, optimizer, scheduler_diffusion, device, epoch):
    model.train()
    total_loss = 0
    total_metrics = {'mse': 0, 'smoothness': 0, 'jerk': 0}
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for start_pose, end_pose, ground_truth, mask, target_length in pbar:
        start_pose = start_pose.to(device)
        end_pose = end_pose.to(device)
        ground_truth = ground_truth.to(device)
        mask = mask.to(device)
        target_length = target_length.to(device)
        
        batch_size, num_frames, _ = ground_truth.shape
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, scheduler_diffusion.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        # Add noise
        noise = torch.randn_like(ground_truth)
        noisy_data = scheduler_diffusion.add_noise(ground_truth, noise, timesteps)
        
        # Predict noise
        condition = torch.cat([start_pose, end_pose], dim=-1)
        predicted_noise = model(noisy_data, timesteps, condition, target_length)
        
        # MSE loss on valid frames only
        mask_expanded = mask.unsqueeze(-1)
        num_valid_elements = mask_expanded.sum() * ground_truth.shape[-1]
        mse_loss = F.mse_loss(
            predicted_noise * mask_expanded,
            noise * mask_expanded,
            reduction='sum'
        ) / num_valid_elements
        
        # Physical constraints
        predicted_skeleton = noisy_data - predicted_noise
        loss, loss_dict = combined_constraint_loss(
            predicted_skeleton, mse_loss,
            bone_weight=0.1, smooth_weight=0.05,
            symmetry_weight=0.02, range_weight=0.1, perceptual_weight=0.15
        )
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        with torch.no_grad():
            metrics = compute_metrics(predicted_noise, noise, mask)
        
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        pbar.set_postfix({
            'total': f"{loss.item():.4f}",
            'mse': f"{loss_dict['mse']:.4f}",
            'bone': f"{loss_dict['bone']:.4f}",
            'percept': f"{loss_dict['perceptual']:.4f}"
        })
    
    return total_loss / len(dataloader), {k: v / len(dataloader) for k, v in total_metrics.items()}


def validate(model, dataloader, scheduler_diffusion, device):
    model.eval()
    total_loss = 0
    total_metrics = {'mse': 0, 'smoothness': 0, 'jerk': 0}
    
    with torch.no_grad():
        for start_pose, end_pose, ground_truth, mask, target_length in tqdm(dataloader, desc="Validating"):
            start_pose = start_pose.to(device)
            end_pose = end_pose.to(device)
            ground_truth = ground_truth.to(device)
            mask = mask.to(device)
            target_length = target_length.to(device)
            
            batch_size, num_frames, _ = ground_truth.shape
            timesteps = torch.randint(
                0, scheduler_diffusion.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()
            
            noise = torch.randn_like(ground_truth)
            noisy_data = scheduler_diffusion.add_noise(ground_truth, noise, timesteps)
            
            condition = torch.cat([start_pose, end_pose], dim=-1)
            predicted_noise = model(noisy_data, timesteps, condition, target_length)
            
            mask_expanded = mask.unsqueeze(-1)
            num_valid_elements = mask_expanded.sum() * ground_truth.shape[-1]
            mse_loss = F.mse_loss(
                predicted_noise * mask_expanded,
                noise * mask_expanded,
                reduction='sum'
            ) / num_valid_elements
            
            predicted_skeleton = noisy_data - predicted_noise
            loss, loss_dict = combined_constraint_loss(
                predicted_skeleton, mse_loss,
                bone_weight=0.1, smooth_weight=0.05,
                symmetry_weight=0.02, range_weight=0.1, perceptual_weight=0.15
            )
            
            metrics = compute_metrics(predicted_noise, noise, mask)
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
    
    return total_loss / len(dataloader), {k: v / len(dataloader) for k, v in total_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description='Train VSL Diffusion Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models/vsl_diffusion_v3',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=8, help='Transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention heads')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining VSL Diffusion Model v3")
    print(f"  Device: {device}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\nLoading datasets...")
    train_dataset = VSLTransitionDataset(args.data_dir, split='train')
    val_dataset = VSLTransitionDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    print("\nCreating model...")
    model = VSLDiffusionModel(
        input_dim=1659,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    scheduler_diffusion = SimpleDDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2"
    )
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler_diffusion, device, epoch)
        val_loss, val_metrics = validate(model, val_loader, scheduler_diffusion, device)
        scheduler_lr.step()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train MSE: {train_metrics['mse']:.4f} | Val MSE: {val_metrics['mse']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(str(output_dir / 'best.pt'))
            print(f"  ✓ Saved best model")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
        
        model.save(str(output_dir / 'latest.pt'))
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            print(f"  Best val loss: {best_val_loss:.4f}")
            break
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
