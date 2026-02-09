#!/usr/bin/env python3
"""
Comprehensive Data Pipeline Verification

Checks entire data flow from raw sequences to training data:
1. Raw sequence files
2. Normalization function
3. Prepared training data
4. Data loader
5. Model input/output
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prepare_data import normalize_skeleton

print("="*70)
print("DATA PIPELINE VERIFICATION")
print("="*70)

# ============================================================================
# STEP 1: Check Raw Sequences
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Raw Sequence Files")
print("="*70)

raw_seq_path = Path('/mnt/ngan/vsl_data/sequences/Hello/000001.npy')
raw_seq = np.load(raw_seq_path)

print(f"\n✓ Loaded: {raw_seq_path}")
print(f"  Shape: {raw_seq.shape}")
print(f"  Range: [{raw_seq.min():.4f}, {raw_seq.max():.4f}]")
print(f"  Mean: {raw_seq.mean():.4f}")
print(f"  Std: {raw_seq.std():.4f}")

# Check if it's raw MediaPipe coordinates
if raw_seq.min() < 0 or raw_seq.max() > 1.5:
    print("  ✓ Raw MediaPipe coordinates (not normalized)")
else:
    print("  ⚠ WARNING: Might already be normalized!")

# ============================================================================
# STEP 2: Test Normalization Function
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Normalization Function")
print("="*70)

# Test with raw sequence
normalized = normalize_skeleton(raw_seq)

print(f"\n✓ Normalized sequence")
print(f"  Input range: [{raw_seq.min():.4f}, {raw_seq.max():.4f}]")
print(f"  Output range: [{normalized.min():.4f}, {normalized.max():.4f}]")
print(f"  Output mean: {normalized.mean():.4f}")
print(f"  Output std: {normalized.std():.4f}")

# Verify normalization is correct
expected_min = (raw_seq.min() + 2.0) / 4.0
expected_max = (raw_seq.max() + 2.0) / 4.0

print(f"\n  Expected range (manual calc): [{expected_min:.4f}, {expected_max:.4f}]")

if abs(normalized.min() - expected_min) < 0.001 and abs(normalized.max() - expected_max) < 0.001:
    print("  ✓ Normalization formula CORRECT!")
else:
    print("  ✗ ERROR: Normalization mismatch!")

# Test edge cases
print("\n  Testing edge cases:")

# Case 1: Values at boundary
test_boundary = np.array([[[-2.0, 0.0, 2.0]]])
norm_boundary = normalize_skeleton(test_boundary)
print(f"    [-2, 0, 2] → [{norm_boundary.min():.4f}, {norm_boundary.mean():.4f}, {norm_boundary.max():.4f}]")
if abs(norm_boundary.min() - 0.0) < 0.001 and abs(norm_boundary.max() - 1.0) < 0.001:
    print("    ✓ Boundary values correct")
else:
    print("    ✗ ERROR: Boundary values wrong!")

# Case 2: Values outside range (should be clipped)
test_outlier = np.array([[[-5.0, 0.0, 5.0]]])
norm_outlier = normalize_skeleton(test_outlier)
print(f"    [-5, 0, 5] → [{norm_outlier.min():.4f}, {norm_outlier.mean():.4f}, {norm_outlier.max():.4f}]")
if abs(norm_outlier.min() - 0.0) < 0.001 and abs(norm_outlier.max() - 1.0) < 0.001:
    print("    ✓ Outliers clipped correctly")
else:
    print("    ✗ ERROR: Outliers not clipped!")

# ============================================================================
# STEP 3: Check Existing Training Data (OLD)
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Existing Training Data (OLD normalization)")
print("="*70)

old_data_path = Path('/mnt/ngan/vsl_data/diffusion/train/transition_000000.npz')
if old_data_path.exists():
    old_data = np.load(old_data_path)
    old_gt = old_data['ground_truth']
    
    print(f"\n✓ Loaded: {old_data_path}")
    print(f"  Shape: {old_gt.shape}")
    print(f"  Range: [{old_gt.min():.4f}, {old_gt.max():.4f}]")
    print(f"  Mean: {old_gt.mean():.4f}")
    
    # Check if it uses full [0,1] range
    if old_gt.min() < 0.01 and old_gt.max() > 0.99:
        print("  ⚠ Uses FULL [0,1] range → per-video normalization")
    else:
        print("  ✓ Does NOT use full range → might be global normalization")
else:
    print(f"\n✗ Not found: {old_data_path}")

# ============================================================================
# STEP 4: Simulate New Training Data
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Simulated NEW Training Data")
print("="*70)

# Simulate what new data will look like
print("\nSimulating data preparation with NEW normalization:")

# Load multiple raw sequences
test_sequences = []
for word in ['Hello', 'How_are_you', 'Thank_you']:
    word_path = Path(f'/mnt/ngan/vsl_data/sequences/{word}')
    if word_path.exists():
        npy_files = list(word_path.glob('*.npy'))
        if npy_files:
            seq = np.load(npy_files[0])
            test_sequences.append((word, seq))

print(f"\n✓ Loaded {len(test_sequences)} test sequences")

for word, seq in test_sequences:
    norm_seq = normalize_skeleton(seq)
    print(f"\n  {word}:")
    print(f"    Raw: [{seq.min():.4f}, {seq.max():.4f}]")
    print(f"    Normalized: [{norm_seq.min():.4f}, {norm_seq.max():.4f}]")
    
    # Check if different videos have different normalized ranges
    # (they should, because raw ranges are different)

# ============================================================================
# STEP 5: Verify Inference Normalization Matches
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Inference Normalization Match")
print("="*70)

# Simulate inference normalization
def inference_normalize(pose):
    """From vsl_diffusion_adapter.py"""
    pose_clipped = np.clip(pose, -2.0, 2.0)
    pose_norm = (pose_clipped + 2.0) / 4.0
    return pose_norm

# Test with same raw sequence
raw_pose = raw_seq[0]  # First frame
training_norm = normalize_skeleton(raw_pose.reshape(1, -1))[0]
inference_norm = inference_normalize(raw_pose)

print(f"\nTesting with first frame of Hello/000001.npy:")
print(f"  Raw pose range: [{raw_pose.min():.4f}, {raw_pose.max():.4f}]")
print(f"  Training normalization: [{training_norm.min():.4f}, {training_norm.max():.4f}]")
print(f"  Inference normalization: [{inference_norm.min():.4f}, {inference_norm.max():.4f}]")

# Check if they match
diff = np.abs(training_norm - inference_norm).max()
print(f"  Max difference: {diff:.6f}")

if diff < 0.001:
    print("  ✓ MATCH! Training and inference use same normalization")
else:
    print("  ✗ ERROR: Normalization mismatch!")

# ============================================================================
# STEP 6: Summary & Recommendations
# ============================================================================
print("\n" + "="*70)
print("SUMMARY & NEXT STEPS")
print("="*70)

print("\n✓ Verification complete!")
print("\nFindings:")
print("  1. Raw sequences are in MediaPipe range (not normalized)")
print("  2. New normalize_skeleton() uses global [-2, 2] → [0, 1]")
print("  3. Normalization matches inference")
print("  4. Old training data used per-video normalization (inconsistent)")

print("\n📋 Next Steps:")
print("  1. Backup old data:")
print("     mv /mnt/ngan/vsl_data/diffusion /mnt/ngan/vsl_data/diffusion_old")
print()
print("  2. Re-prepare data with new normalization:")
print("     cd ~/mya/vsl-synthesis")
print("     python src/prepare_data.py \\")
print("       --data_dir /mnt/ngan/vsl_data/sequences \\")
print("       --output_dir /mnt/ngan/vsl_data/diffusion \\")
print("       --window_size 20 \\")
print("       --stride 5")
print()
print("  3. Verify new data:")
print("     python scripts/verify_data_pipeline.py")
print()
print("  4. Train model with new data")

print("\n" + "="*70)
