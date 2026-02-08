#!/usr/bin/env python3
"""
Test individual diffusion components to isolate the issue
"""
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vsl_diffusion import VSLDiffusionModel
from src.models.custom_scheduler import SimpleDDPMScheduler

print("="*60)
print("DIFFUSION COMPONENT TESTING")
print("="*60)

# Load model
print("\n1. Loading model...")
checkpoint = torch.load('models/vsl_diffusion_v2/best.pt', map_location='cuda')
model = VSLDiffusionModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
model.eval()
print("✓ Model loaded")

# Load training data sample
print("\n2. Loading training data...")
data = np.load('/mnt/ngan/vsl_data/diffusion/train/transition_000000.npz')
gt = torch.from_numpy(np.array(data['ground_truth'])).float().cuda()  # (5, 1659)
start = torch.from_numpy(np.array(data['start_pose'])).float().cuda()  # (1659,)
end = torch.from_numpy(np.array(data['end_pose'])).float().cuda()  # (1659,)

print(f"  Ground truth: {gt.shape}, range [{gt.min():.3f}, {gt.max():.3f}]")
print(f"  Start pose: {start.shape}, range [{start.min():.3f}, {start.max():.3f}]")
print(f"  End pose: {end.shape}, range [{end.min():.3f}, {end.max():.3f}]")

# Test 1: Model forward pass with clean data
print("\n3. TEST 1: Model prediction on clean data")
print("-" * 60)
condition = torch.cat([start, end]).unsqueeze(0)  # (1, 3318)
target_length = torch.tensor([5], dtype=torch.long).cuda()

# Add small noise
noise = torch.randn_like(gt) * 0.1
noisy_gt = torch.clamp(gt + noise, 0, 1).unsqueeze(0)  # (1, 5, 1659)

with torch.no_grad():
    # Test at different timesteps
    for t_val in [999, 500, 100, 10]:
        t = torch.tensor([t_val]).cuda()
        pred = model(noisy_gt, t, condition, target_length)
        print(f"  t={t_val:3d}: pred range [{pred.min():7.3f}, {pred.max():7.3f}], "
              f"mean {pred.mean():7.3f}, std {pred.std():6.3f}")

# Test 2: Scheduler add_noise and denoise
print("\n4. TEST 2: Scheduler noise/denoise cycle")
print("-" * 60)
scheduler = SimpleDDPMScheduler(num_train_timesteps=1000, num_inference_steps=50)

# Add noise at max timestep
clean = gt.unsqueeze(0)  # (1, 5, 1659)
t_max = torch.tensor([999]).cuda()
noise = torch.randn_like(clean)

noisy = scheduler.add_noise(clean, noise, t_max)
print(f"  Clean: [{clean.min():.3f}, {clean.max():.3f}]")
print(f"  Noise: [{noise.min():.3f}, {noise.max():.3f}]")
print(f"  Noisy (t=999): [{noisy.min():.3f}, {noisy.max():.3f}]")

# Try to denoise with perfect noise prediction
print("\n  Denoising with perfect noise prediction:")
x_t = noisy.clone()
for i, timestep in enumerate(scheduler.timesteps):
    # Perfect prediction: use the actual noise we added
    output = scheduler.step(noise, timestep.item(), x_t)
    x_t = output.prev_sample
    
    if i in [0, 10, 20, 30, 40, 49]:
        print(f"    Step {i:2d} (t={timestep:3d}): [{x_t.min():7.3f}, {x_t.max():7.3f}]")

print(f"  Final: [{x_t.min():.3f}, {x_t.max():.3f}]")
print(f"  Target: [{clean.min():.3f}, {clean.max():.3f}]")
print(f"  Error: {(x_t - clean).abs().mean():.6f}")

# Test 3: Full inference loop with model
print("\n5. TEST 3: Full inference with model predictions")
print("-" * 60)

# Start from uniform noise [0,1]
x_t = torch.rand(1, 5, 1659).cuda()
print(f"  Initial noise: [{x_t.min():.3f}, {x_t.max():.3f}]")

with torch.no_grad():
    for i, timestep in enumerate(scheduler.timesteps):
        t = torch.tensor([timestep]).cuda()
        
        # Model predicts noise
        noise_pred = model(x_t, t, condition, target_length)
        
        # Denoise
        output = scheduler.step(noise_pred, timestep.item(), x_t)
        x_t = output.prev_sample
        
        if i in [0, 10, 20, 30, 40, 49]:
            print(f"    Step {i:2d} (t={timestep:3d}): "
                  f"pred[{noise_pred.min():7.3f}, {noise_pred.max():7.3f}], "
                  f"x_t[{x_t.min():7.3f}, {x_t.max():7.3f}]")

print(f"\n  Final output: [{x_t.min():.3f}, {x_t.max():.3f}]")
print(f"  Ground truth: [{clean.min():.3f}, {clean.max():.3f}]")
print(f"  Error: {(x_t - clean).abs().mean():.6f}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("If TEST 2 works but TEST 3 fails → Model predictions are wrong")
print("If TEST 2 fails → Scheduler implementation is broken")
print("If both fail → Fundamental issue with diffusion setup")
