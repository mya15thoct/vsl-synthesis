#!/usr/bin/env python3
"""
Training Script for VSL Diffusion Model

Self-supervised training on transition examples extracted from word videos.

Usage:
    python scripts/train_diffusion.py --data_dir data/diffusion --epochs 50
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import DDPMScheduler
from pathlib import Path
import sys
from tqdm import tqdm
import json
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.vsl_diffusion import VSLDiffusionModel
from src.models.dataset import VSLTransitionDataset, collate_fn


def compute_metrics(predicted, target, mask=None):
    """
    Compute training metrics.
    
    Args:
        predicted: (batch, num_frames, 1662)
        target: (batch, num_frames, 1662)
        mask: (batch, num_frames) optional mask for variable lengths
        
    Returns:
        dict of metrics
    """
    if mask is not None:
        # Apply mask
        mask = mask.unsqueeze(-1)  # (batch, num_frames, 1)
        predicted = predicted * mask
        target = target * mask
        num_valid = mask.sum()
    else:
        num_valid = predicted.numel()
    
    # MSE loss
    mse = F.mse_loss(predicted, target, reduction='sum') / num_valid
    
    # Smoothness (velocity variance)
    pred_vel = predicted[:, 1:] - predicted[:, :-1]
    smoothness = pred_vel.var()
    
    # Jerk (acceleration variance)
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    jerk = pred_acc.var()
    
    return {
        'mse': mse.item(),
        'smoothness': smoothness.item(),
        'jerk': jerk.item()
    }


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler_diffusion,
    device,
    epoch
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_metrics = {'mse': 0, 'smoothness': 0, 'jerk': 0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (start_pose, end_pose, ground_truth, mask) in enumerate(pbar):
        # Move to device
        start_pose = start_pose.to(device)
        end_pose = end_pose.to(device)
        ground_truth = ground_truth.to(device)
        mask = mask.to(device)
        
        batch_size, num_frames, _ = ground_truth.shape
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, scheduler_diffusion.config.num_train_timesteps,
            (batch_size,),
            device=device
        ).long()
        
        # Add noise to ground truth
        noise = torch.randn_like(ground_truth)
        noisy_data = scheduler_diffusion.add_noise(ground_truth, noise, timesteps)
        
        # Create condition (concatenate start + end poses)
        condition = torch.cat([start_pose, end_pose], dim=-1)
        
        # Predict noise
        predicted_noise = model(noisy_data, timesteps, condition)
        
        # Compute loss (only on valid frames)
        mask_expanded = mask.unsqueeze(-1)  # (batch, num_frames, 1)
        loss = F.mse_loss(
            predicted_noise * mask_expanded,
            noise * mask_expanded,
            reduction='sum'
        ) / mask.sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(predicted_noise, noise, mask)
        
        # Update totals
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{metrics['mse']:.4f}"
        })
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


def validate(model, dataloader, scheduler_diffusion, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    total_metrics = {'mse': 0, 'smoothness': 0, 'jerk': 0}
    
    with torch.no_grad():
        for start_pose, end_pose, ground_truth, mask in tqdm(dataloader, desc="Validating"):
            start_pose = start_pose.to(device)
            end_pose = end_pose.to(device)
            ground_truth = ground_truth.to(device)
            mask = mask.to(device)
            
            batch_size, num_frames, _ = ground_truth.shape
            
            # Sample timesteps
            timesteps = torch.randint(
                0, scheduler_diffusion.config.num_train_timesteps,
                (batch_size,),
                device=device
            ).long()
            
            # Add noise
            noise = torch.randn_like(ground_truth)
            noisy_data = scheduler_diffusion.add_noise(ground_truth, noise, timesteps)
            
            # Predict
            condition = torch.cat([start_pose, end_pose], dim=-1)
            predicted_noise = model(noisy_data, timesteps, condition)
            
            # Loss
            mask_expanded = mask.unsqueeze(-1)
            loss = F.mse_loss(
                predicted_noise * mask_expanded,
                noise * mask_expanded,
                reduction='sum'
            ) / mask.sum()
            
            # Metrics
            metrics = compute_metrics(predicted_noise, noise, mask)
            
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train VSL Diffusion Model')
    parser.add_argument('--data_dir', type=str, default='data/diffusion',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models/vsl_diffusion',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Training VSL Diffusion Model")
    print(f"  Device: {device}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = VSLTransitionDataset(args.data_dir, split='train')
    val_dataset = VSLTransitionDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = VSLDiffusionModel(
        input_dim=1662,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Create diffusion scheduler (fix numpy compatibility)
    # Use from_config to avoid numpy version issues
    from diffusers import SchedulerMixin
    
    scheduler_config = {
        "num_train_timesteps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon",
        "clip_sample": False,
        "beta_start": 0.0001,
        "beta_end": 0.02,
    }
    
    scheduler_diffusion = DDPMScheduler.from_config(scheduler_config)
    
    # Optimizer and LR scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\n🎯 Starting training for {args.epochs} epochs...")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler_diffusion, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, scheduler_diffusion, device
        )
        
        # Update LR
        scheduler_lr.step()
        
        # Log
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train MSE: {train_metrics['mse']:.4f} | Val MSE: {val_metrics['mse']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(output_dir / 'best.pt'))
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save latest
        model.save(str(output_dir / 'latest.pt'))
        
        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
