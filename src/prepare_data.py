#!/usr/bin/env python3
"""
Data Preprocessing for VSL Diffusion Model

Extract transition examples from word videos for self-supervised training.

Usage:
    python src/prepare_data.py --data_dir /mnt/ngan/vsl_data/sequences
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import from same package
from core.concatenation import load_skeleton_sequence


def normalize_skeleton(skeleton):
    """
    Normalize skeleton coordinates to [0, 1] range.
    
    Args:
        skeleton: (frames, 1662) or (frames, 554, 3) skeleton data
        
    Returns:
        Normalized skeleton in same shape, values in [0, 1]
    """
    original_shape = skeleton.shape
    
    # Flatten if needed
    if skeleton.ndim == 3:
        skeleton_flat = skeleton.reshape(skeleton.shape[0], -1)
    else:
        skeleton_flat = skeleton
    
    # Normalize to [0, 1] using min-max scaling
    min_val = skeleton_flat.min()
    max_val = skeleton_flat.max()
    
    if max_val > min_val:
        normalized = (skeleton_flat - min_val) / (max_val - min_val)
    else:
        # All values are the same, set to 0.5
        normalized = np.full_like(skeleton_flat, 0.5)
    
    # Reshape back to original shape
    return normalized.reshape(original_shape)


def extract_transitions_from_video(
    video_path: Path,
    window_sizes: list = [5, 10, 15, 20],
    stride: int = 5,
    min_frames: int = 15
):
    """
    Extract transition examples from a single word video with multiple scales.
    
    Args:
        video_path: Path to .npy video file
        window_sizes: List of transition lengths to extract (multi-scale)
        stride: Step size for sliding window
        min_frames: Minimum video length to process
        
    Yields:
        dict with 'start_pose', 'end_pose', 'ground_truth', 'metadata'
    """
    try:
        # Load video
        video = load_skeleton_sequence(video_path)  # (frames, keypoints, 3)
        
        # Flatten to (frames, 1662) if needed
        if video.ndim == 3:
            num_frames = video.shape[0]
            video_flat = video.reshape(num_frames, -1)
        else:
            video_flat = video
            num_frames = video_flat.shape[0]
        
        # Skip videos that are too short
        if num_frames < min_frames:
            return
        
        # Extract transitions with multiple window sizes
        for window_size in window_sizes:
            # Skip if video too short for this window
            if num_frames < window_size:
                continue
                
            # Extract transitions with sliding window
            for start_idx in range(0, num_frames - window_size + 1, stride):
                end_idx = start_idx + window_size
                transition = video_flat[start_idx:end_idx]
                
                # Normalize transition to [0, 1]
                transition_normalized = normalize_skeleton(transition)
                
                yield {
                    'start_pose': transition_normalized[0],      # (1662,)
                    'end_pose': transition_normalized[-1],       # (1662,)
                    'ground_truth': transition_normalized[1:-1], # (window_size-2, 1662)
                    'num_frames': window_size - 2,    # Actual transition length
                    'metadata': {
                        'source_video': str(video_path.name),
                        'start_frame': start_idx,
                        'end_frame': end_idx,
                        'window_size': window_size
                    }
                }
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return


def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    window_size: int = 20,
    stride: int = 5,
    train_split: float = 0.9,
    max_videos_per_word: int = None
):
    """
    Prepare training dataset from word videos.
    
    Args:
        data_dir: Directory containing word folders
        output_dir: Where to save processed data
        window_size: Transition length in frames
        stride: Sliding window stride
        train_split: Fraction for training (rest is validation)
        max_videos_per_word: Limit videos per word (for testing)
    """
    print(f"\nScanning for word videos in: {data_dir}")
    
    # Find all word folders
    word_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(word_folders)} word folders")
    
    # Collect all video files
    all_videos = []
    for word_folder in tqdm(word_folders, desc="Scanning words"):
        npy_files = sorted(word_folder.glob("*.npy"))
        
        # Limit videos per word if specified
        if max_videos_per_word:
            npy_files = npy_files[:max_videos_per_word]
        
        all_videos.extend(npy_files)
    
    print(f"Found {len(all_videos)} total videos")
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Process videos and extract transitions
    print(f"\n Extracting transitions (window={window_size}, stride={stride})...")
    
    train_transitions = []
    val_transitions = []
    stats = {
        'total_videos': len(all_videos),
        'total_transitions': 0,
        'train_transitions': 0,
        'val_transitions': 0,
        'window_size': window_size,
        'stride': stride,
        'skipped_videos': 0
    }
    
    for video_idx, video_path in enumerate(tqdm(all_videos, desc="Processing videos")):
        # Determine train/val split
        is_train = (video_idx / len(all_videos)) < train_split
        
        transitions_found = 0
        for transition in extract_transitions_from_video(
            video_path, 
            window_sizes=[7, 12, 17, 22],  # Multi-scale: 5, 10, 15, 20 frame transitions
            stride=stride
        ):
            if is_train:
                train_transitions.append(transition)
            else:
                val_transitions.append(transition)
            
            transitions_found += 1
        
        if transitions_found == 0:
            stats['skipped_videos'] += 1
    
    stats['train_transitions'] = len(train_transitions)
    stats['val_transitions'] = len(val_transitions)
    stats['total_transitions'] = stats['train_transitions'] + stats['val_transitions']
    
    # Save transitions
    print(f"\nSaving transitions...")
    print(f"  Train: {len(train_transitions)} examples")
    print(f"  Val: {len(val_transitions)} examples")
    
    # Save train data
    for idx, transition in enumerate(tqdm(train_transitions, desc="Saving train")):
        save_path = train_dir / f"transition_{idx:06d}.npz"
        np.savez_compressed(
            save_path,
            start_pose=transition['start_pose'],
            end_pose=transition['end_pose'],
            ground_truth=transition['ground_truth'],
            metadata=json.dumps(transition['metadata'])
        )
    
    # Save val data
    for idx, transition in enumerate(tqdm(val_transitions, desc="Saving val")):
        save_path = val_dir / f"transition_{idx:06d}.npz"
        np.savez_compressed(
            save_path,
            start_pose=transition['start_pose'],
            end_pose=transition['end_pose'],
            ground_truth=transition['ground_truth'],
            metadata=json.dumps(transition['metadata'])
        )
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset preparation complete!")
    print(f"  Total transitions: {stats['total_transitions']}")
    print(f"  Train: {stats['train_transitions']}")
    print(f"  Val: {stats['val_transitions']}")
    print(f"  Skipped videos: {stats['skipped_videos']}")
    print(f"  Output: {output_dir}")
    print(f"  Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data for VSL diffusion model'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing word videos (e.g., /mnt/ngan/vsl_data/sequences)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/ngan/vsl_data/diffusion',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=20,
        help='Transition length in frames (default: 20)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=5,
        help='Sliding window stride (default: 5)'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.9,
        help='Train/val split ratio (default: 0.9)'
    )
    parser.add_argument(
        '--max_videos_per_word',
        type=int,
        default=None,
        help='Limit videos per word (for testing)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    prepare_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        window_size=args.window_size,
        stride=args.stride,
        train_split=args.train_split,
        max_videos_per_word=args.max_videos_per_word
    )


if __name__ == "__main__":
    main()
