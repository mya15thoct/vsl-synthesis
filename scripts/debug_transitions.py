#!/usr/bin/env python3
"""
Debug script to test transition generation in pipeline
"""
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.concatenation import concatenate_sequences, get_transition_boundaries
from src.methods.interpolation import cubic_spline_interpolation

# Test files
word_files = [
    '/mnt/ngan/vsl_data/sequences/Hello/000001.npy',
    '/mnt/ngan/vsl_data/sequences/How_are_you/000001.npy'
]

transition_frames = 10

print("Step 1: Concatenate sequences...")
concatenated = concatenate_sequences(word_files, transition_frames)
print(f"  Shape: {concatenated.shape}")
print(f"  Range: [{concatenated.min():.4f}, {concatenated.max():.4f}]")

print("\nStep 2: Get transition boundaries...")
boundaries = get_transition_boundaries(word_files, transition_frames)
print(f"  Boundaries: {boundaries}")

print("\nStep 3: Generate transitions with spline...")
for i, (start_idx, end_idx, start_pose_idx, end_pose_idx) in enumerate(boundaries):
    print(f"\n  Transition {i+1}:")
    print(f"    Gap: frames {start_idx}-{end_idx}")
    print(f"    Start pose idx: {start_pose_idx}")
    print(f"    End pose idx: {end_pose_idx}")
    
    start_pose = concatenated[start_pose_idx]
    end_pose = concatenated[end_pose_idx]
    
    print(f"    Start pose shape: {start_pose.shape}")
    print(f"    End pose shape: {end_pose.shape}")
    print(f"    Start pose range: [{start_pose.min():.4f}, {start_pose.max():.4f}]")
    print(f"    End pose range: [{end_pose.min():.4f}, {end_pose.max():.4f}]")
    
    # Check if gap is zeros before filling
    gap_before = concatenated[start_idx:end_idx]
    print(f"    Gap before (all zeros?): {np.all(gap_before == 0)}")
    
    # Generate transition
    transition = cubic_spline_interpolation(start_pose, end_pose, transition_frames)
    print(f"    Transition shape: {transition.shape}")
    print(f"    Transition range: [{transition.min():.4f}, {transition.max():.4f}]")
    
    # Fill transition
    concatenated[start_idx:end_idx] = transition
    
    # Check if gap is filled
    gap_after = concatenated[start_idx:end_idx]
    print(f"    Gap after (all zeros?): {np.all(gap_after == 0)}")
    print(f"    Gap after range: [{gap_after.min():.4f}, {gap_after.max():.4f}]")

print("\n\nFinal concatenated range:", f"[{concatenated.min():.4f}, {concatenated.max():.4f}]")
print("Done!")
