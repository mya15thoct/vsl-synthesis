#!/usr/bin/env python3
"""
Rest Pose Detection and Trimming

Detects "rest pose" frames at the end of sign language sequences
(where hands are lowered and body is still) and optionally trims them.

Usage:
    # Check only (no trimming)
    python scripts/trim_rest_pose.py --check_only

    # Trim and save to new directory
    python scripts/trim_rest_pose.py \
        --input_dir /mnt/ngan/vsl_data/sequences \
        --output_dir /mnt/ngan/vsl_data/sequences_trimmed

    # Trim in-place (CAREFUL: overwrites original files)
    python scripts/trim_rest_pose.py \
        --input_dir /mnt/ngan/vsl_data/sequences \
        --inplace
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json


# MediaPipe keypoint indices for wrist and hand landmarks
# In the 1659-feature format: pose is first 99 features (33 kp x 3: x,y,z, NO visibility)
# Wrist keypoints: left=15, right=16
LEFT_WRIST_IDX = 15    # MediaPipe pose landmark 15 = left wrist
RIGHT_WRIST_IDX = 16   # MediaPipe pose landmark 16 = right wrist

# Each pose landmark has 3 values (x, y, z) — no visibility in 1659 format
POSE_FEATURE_SIZE = 3
POSE_TOTAL = 33 * POSE_FEATURE_SIZE  # 99


def get_wrist_positions(skeleton_flat):
    """
    Extract wrist positions from flattened skeleton.
    
    Args:
        skeleton_flat: (frames, 1659) or (1659,) flattened skeleton
        
    Returns:
        left_wrist: (frames, 3) or (3,) xyz positions
        right_wrist: (frames, 3) or (3,) xyz positions
    """
    was_1d = skeleton_flat.ndim == 1
    if was_1d:
        skeleton_flat = skeleton_flat[np.newaxis, :]

    # Extract pose section (first 99 features = 33 × 3, no visibility)
    pose = skeleton_flat[:, :POSE_TOTAL].reshape(-1, 33, POSE_FEATURE_SIZE)

    left_wrist  = pose[:, LEFT_WRIST_IDX,  :3]   # (frames, 3)
    right_wrist = pose[:, RIGHT_WRIST_IDX, :3]   # (frames, 3)

    if was_1d:
        return left_wrist[0], right_wrist[0]
    return left_wrist, right_wrist


def compute_motion(skeleton_flat, smooth_window=3):
    """
    Compute per-frame motion using wrist velocity.
    
    Args:
        skeleton_flat: (frames, 1659)
        smooth_window: Window size for smoothing
        
    Returns:
        motion: (frames-1,) motion magnitude per frame
    """
    left_wrist, right_wrist = get_wrist_positions(skeleton_flat)
    
    # Velocity = difference between consecutive frames
    left_vel = np.diff(left_wrist, axis=0)   # (frames-1, 3)
    right_vel = np.diff(right_wrist, axis=0)  # (frames-1, 3)
    
    # Motion magnitude = L2 norm of velocity
    left_motion = np.sqrt((left_vel ** 2).sum(axis=-1))
    right_motion = np.sqrt((right_vel ** 2).sum(axis=-1))
    
    # Combined motion
    motion = (left_motion + right_motion) / 2
    
    # Smooth to reduce noise
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        motion = np.convolve(motion, kernel, mode='same')
    
    return motion


def detect_rest_frames(skeleton_flat, motion_threshold=0.01, min_rest_length=3):
    """
    Detect rest pose frames at the end of the sequence.
    
    Rest pose = low motion for consecutive frames at the end.
    
    Args:
        skeleton_flat: (frames, 1659)
        motion_threshold: Motion below this = still/rest
        min_rest_length: Minimum consecutive still frames to consider rest
        
    Returns:
        trim_end: Index to trim at (exclusive), or len(skeleton) if no rest detected
        rest_start: Frame where rest begins, or None
    """
    if len(skeleton_flat) < min_rest_length + 2:
        return len(skeleton_flat), None
    
    motion = compute_motion(skeleton_flat)
    num_frames = len(skeleton_flat)
    
    # Find rest frames at the end (scan from end to beginning)
    rest_start = None
    
    for i in range(len(motion) - 1, min_rest_length - 2, -1):
        # Check if this window is all still
        window_start = max(0, i - min_rest_length + 1)
        window_motion = motion[window_start:i + 1]
        
        if window_motion.max() < motion_threshold:
            rest_start = window_start
        else:
            break  # Found motion, stop scanning
    
    if rest_start is None:
        return num_frames, None
    
    # Trim to rest_start + 1 (keep one "approaching rest" frame)
    trim_end = max(rest_start, 5)  # Keep at least 5 frames
    return trim_end, rest_start


def analyze_sequence(filepath, motion_threshold=0.01, min_rest_length=3):
    """
    Analyze a single sequence file.
    
    Returns:
        dict with analysis results
    """
    try:
        data = np.load(filepath)
        
        # Handle (frames, 553, 3) and (frames, 1659) formats
        if data.ndim == 3:
            data_flat = data.reshape(data.shape[0], -1)
        else:
            data_flat = data
        
        num_frames = len(data_flat)
        trim_end, rest_start = detect_rest_frames(
            data_flat, motion_threshold, min_rest_length
        )
        
        frames_trimmed = num_frames - trim_end
        has_rest = rest_start is not None
        
        return {
            'file': str(filepath),
            'num_frames': num_frames,
            'trim_end': trim_end,
            'rest_start': rest_start,
            'frames_trimmed': frames_trimmed,
            'has_rest': has_rest,
            'trim_ratio': frames_trimmed / num_frames if num_frames > 0 else 0
        }
    except Exception as e:
        return {
            'file': str(filepath),
            'error': str(e),
            'has_rest': False
        }


def main():
    parser = argparse.ArgumentParser(description='Detect and trim rest poses')
    parser.add_argument('--input_dir', type=str,
                        default='/mnt/ngan/vsl_data/sequences',
                        help='Input directory with word sequences')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for trimmed sequences (if None, uses input_dir)')
    parser.add_argument('--inplace', action='store_true',
                        help='Trim files in place (overwrites originals)')
    parser.add_argument('--check_only', action='store_true',
                        help='Only check, do not trim')
    parser.add_argument('--motion_threshold', type=float, default=0.01,
                        help='Motion threshold for rest detection (default: 0.01)')
    parser.add_argument('--min_rest_length', type=int, default=3,
                        help='Min consecutive still frames to consider rest (default: 3)')
    parser.add_argument('--sample_words', type=int, default=5,
                        help='Number of words to sample for check_only mode')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Find all .npy files
    word_folders = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(word_folders)} word folders in {input_dir}")
    
    # ============================================================
    # STEP 1: Check mode - analyze sample
    # ============================================================
    if args.check_only:
        print(f"\n{'='*60}")
        print("CHECK MODE: Analyzing sample sequences")
        print(f"{'='*60}")
        
        sample_words = word_folders[:args.sample_words]
        all_results = []
        
        for word_folder in sample_words:
            npy_files = sorted(word_folder.glob('*.npy'))[:3]  # 3 files per word
            print(f"\nWord: {word_folder.name}")
            
            for npy_file in npy_files:
                result = analyze_sequence(
                    npy_file, args.motion_threshold, args.min_rest_length
                )
                all_results.append(result)
                
                if 'error' in result:
                    print(f"  {npy_file.name}: ERROR - {result['error']}")
                else:
                    status = "HAS REST" if result['has_rest'] else "no rest"
                    print(f"  {npy_file.name}: {result['num_frames']} frames, "
                          f"{status}, "
                          f"trim {result['frames_trimmed']} frames "
                          f"({result['trim_ratio']*100:.1f}%)")
        
        # Summary
        has_rest = [r for r in all_results if r.get('has_rest')]
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"  Sequences checked: {len(all_results)}")
        print(f"  Sequences with rest pose: {len(has_rest)} ({len(has_rest)/len(all_results)*100:.0f}%)")
        if has_rest:
            avg_trim = np.mean([r['trim_ratio'] for r in has_rest])
            print(f"  Average trim ratio: {avg_trim*100:.1f}%")
            print(f"\n⚠ Rest poses detected! Run without --check_only to trim.")
        else:
            print(f"\n✓ No significant rest poses found. Data looks clean!")
        return
    
    # ============================================================
    # STEP 2: Trim mode
    # ============================================================
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.inplace:
        output_dir = input_dir
    else:
        print("ERROR: Specify --output_dir or --inplace")
        return
    
    print(f"\n{'='*60}")
    print(f"TRIM MODE")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")
    
    stats = {
        'total_files': 0,
        'files_trimmed': 0,
        'files_unchanged': 0,
        'total_frames_removed': 0,
        'errors': 0
    }
    
    for word_folder in tqdm(word_folders, desc="Processing words"):
        npy_files = sorted(word_folder.glob('*.npy'))
        
        for npy_file in npy_files:
            stats['total_files'] += 1
            
            result = analyze_sequence(npy_file, args.motion_threshold, args.min_rest_length)
            
            if 'error' in result:
                stats['errors'] += 1
                continue
            
            # Load original data
            data = np.load(npy_file)
            trim_end = result['trim_end']
            
            # Prepare output path
            rel_path = npy_file.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            if result['has_rest'] and result['frames_trimmed'] > 0:
                # Trim!
                if data.ndim == 3:
                    trimmed = data[:trim_end]
                else:
                    trimmed = data[:trim_end]
                
                np.save(out_path, trimmed)
                stats['files_trimmed'] += 1
                stats['total_frames_removed'] += result['frames_trimmed']
            else:
                # No rest, copy as-is (or skip if inplace)
                if not args.inplace:
                    np.save(out_path, data)
                stats['files_unchanged'] += 1
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Trimmed: {stats['files_trimmed']}")
    print(f"  Unchanged: {stats['files_unchanged']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Total frames removed: {stats['total_frames_removed']}")
    print(f"  Output: {output_dir}")
    
    # Save stats
    stats_path = output_dir / 'trim_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {stats_path}")


if __name__ == '__main__':
    main()
