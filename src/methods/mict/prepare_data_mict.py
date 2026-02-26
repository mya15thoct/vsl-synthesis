#!/usr/bin/env python3
"""
Data Preparation for MicT (Motion is the Choreographer)
arXiv 2508.04049

Khác với prepare_data.py (sliding window → start/end/gt):
- MicT cần: ghép N từ liên tiếp → sequence đầy đủ + masked sequence (zero frames tại transition)

Usage:
    PYTHONPATH=. python src/methods/mict/prepare_data_mict.py \
        --data_dir /mnt/ngan/vsl_data/sequences \
        --output_dir /mnt/ngan/vsl_data/mict \
        --transition_frames 10
"""

import argparse
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import itertools


GLOBAL_MIN = -2.0
GLOBAL_MAX = 2.0


def normalize_skeleton(skeleton):
    """Normalize to [0,1] using fixed global range."""
    clipped = np.clip(skeleton, GLOBAL_MIN, GLOBAL_MAX)
    return (clipped - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)


def load_npy(path):
    """Load .npy và flatten nếu cần → (frames, 1659)."""
    data = np.load(str(path)).astype(np.float32)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    return data


def build_sentence_sample(word_paths, transition_frames=10):
    """
    Ghép N từ thành 1 sequence câu với transition frames (zero) ở giữa.

    Args:
        word_paths: list of Path — .npy files cho từng từ trong câu
        transition_frames: số frame zero chèn giữa 2 từ

    Returns:
        dict:
            full_sequence: (T_total, 1659) — sequence hoàn chỉnh (signs + transitions zero)
            masked_sequence: (T_total, 1659) — giống full nhưng transition=0
            frame_mask: (T_total,) bool — True = transition frame (cần sinh)
            word_lengths: list[int] — số frame mỗi từ
            num_words: int
    """
    segments = []
    for p in word_paths:
        seg = load_npy(p)
        seg = normalize_skeleton(seg)
        segments.append(seg)

    feat_dim = segments[0].shape[1]  # 1659
    transition_zeros = np.zeros((transition_frames, feat_dim), dtype=np.float32)

    # Ghép: [word0] [zeros] [word1] [zeros] ... [wordN-1]
    full_parts = []
    mask_parts = []
    word_lengths = []

    for i, seg in enumerate(segments):
        full_parts.append(seg)
        mask_parts.append(np.zeros_like(seg))  # word frames: mask=False (observed)
        word_lengths.append(len(seg))

        if i < len(segments) - 1:
            full_parts.append(transition_zeros.copy())
            mask_parts.append(np.ones((transition_frames, feat_dim), dtype=np.float32))  # transition: mask=True

    full_sequence = np.concatenate(full_parts, axis=0)   # (T_total, 1659)
    frame_mask = np.concatenate(mask_parts, axis=0)       # (T_total, 1659)

    # masked_sequence: observed frames giữ nguyên, transition frames = 0
    masked_sequence = full_sequence * (1.0 - frame_mask)  # zeros at transition

    return {
        'full_sequence': full_sequence,       # (T_total, 1659)
        'masked_sequence': masked_sequence,   # (T_total, 1659)
        'frame_mask': frame_mask[:, 0],       # (T_total,) — 1 = transition
        'word_lengths': word_lengths,
        'num_words': len(segments),
    }


def prepare_mict_dataset(
    data_dir: Path,
    output_dir: Path,
    transition_frames: int = 10,
    min_words: int = 2,
    max_words: int = 5,
    samples_per_combo: int = 1, # This parameter is not used in the current implementation
    max_samples: int = 50000,
    train_split: float = 0.9,
    seed: int = 42,
):
    """
    Tạo dataset MicT-style:
    - Lấy ngẫu nhiên 2–5 từ từ sequences/
    - Ghép thành 1 sequence với transition zeros
    - Lưu .npz
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"\nScanning word folders in: {data_dir}")
    word_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(word_folders)} words")

    # Map: word_name → list of .npy files
    word_files = {}
    for folder in word_folders:
        npys = sorted(folder.glob("*.npy"))
        if npys:
            word_files[folder.name] = npys

    word_names = list(word_files.keys())
    print(f"Words with data: {len(word_names)}")

    # Tạo output dirs
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # First pass: Generate recipes for samples (chosen words and their paths)
    # This avoids accumulating full data in RAM
    sample_recipes = []
    print(f"\nGenerating sentence sample recipes (min={min_words}, max={max_words} words)...")

    while len(sample_recipes) < max_samples:
        # Chọn ngẫu nhiên N từ
        n_words = random.randint(min_words, max_words)
        if len(word_names) < n_words:
            print(f"Warning: Not enough unique words ({len(word_names)}) to pick {n_words}. Stopping recipe generation.")
            break
        chosen_words = random.sample(word_names, n_words)

        # Chọn ngẫu nhiên 1 video cho mỗi từ
        paths = [random.choice(word_files[w]) for w in chosen_words]
        sample_recipes.append({'words': chosen_words, 'paths': paths})

        if len(sample_recipes) % 5000 == 0:
            print(f"  Generated {len(sample_recipes)} recipes...")

    print(f"Total recipes generated: {len(sample_recipes)}")

    # Shuffle và chia train/val recipes
    random.shuffle(sample_recipes)
    split_idx = int(len(sample_recipes) * train_split)
    train_recipes = sample_recipes[:split_idx]
    val_recipes = sample_recipes[split_idx:]

    # Second pass: Build and save samples incrementally
    print(f"\nSaving {len(train_recipes)} train, {len(val_recipes)} val samples...")

    stats = {
        'total': len(sample_recipes),
        'train': len(train_recipes),
        'val': len(val_recipes),
        'transition_frames': transition_frames,
        'min_words': min_words,
        'max_words': max_words,
    }

    for idx, recipe in enumerate(tqdm(train_recipes, desc="Saving train")):
        try:
            s = build_sentence_sample(recipe['paths'], transition_frames)
            s['words'] = recipe['words']
            s['num_words'] = len(recipe['words']) # Ensure num_words is set correctly
            np.savez_compressed(
                train_dir / f"sample_{idx:06d}.npz",
                full_sequence=s['full_sequence'],
                masked_sequence=s['masked_sequence'],
                frame_mask=s['frame_mask'].astype(np.float32),
                word_lengths=np.array(s['word_lengths'], dtype=np.int32),
                metadata=json.dumps({'words': s['words'], 'num_words': s['num_words']})
            )
        except Exception as e:
            print(f"Error processing train sample {idx} (words: {recipe['words']}): {e}")
            continue

    for idx, recipe in enumerate(tqdm(val_recipes, desc="Saving val")):
        try:
            s = build_sentence_sample(recipe['paths'], transition_frames)
            s['words'] = recipe['words']
            s['num_words'] = len(recipe['words']) # Ensure num_words is set correctly
            np.savez_compressed(
                val_dir / f"sample_{idx:06d}.npz",
                full_sequence=s['full_sequence'],
                masked_sequence=s['masked_sequence'],
                frame_mask=s['frame_mask'].astype(np.float32),
                word_lengths=np.array(s['word_lengths'], dtype=np.int32),
                metadata=json.dumps({'words': s['words'], 'num_words': s['num_words']})
            )
        except Exception as e:
            print(f"Error processing val sample {idx} (words: {recipe['words']}): {e}")
            continue

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Saved to {output_dir}")
    print(f"  Train: {len(train_recipes)} | Val: {len(val_recipes)}")


def main():
    parser = argparse.ArgumentParser(description='Prepare MicT training data')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/mnt/ngan/vsl_data/mict')
    parser.add_argument('--transition_frames', type=int, default=10)
    parser.add_argument('--min_words', type=int, default=2)
    parser.add_argument('--max_words', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=50000)
    parser.add_argument('--train_split', type=float, default=0.9)
    args = parser.parse_args()

    prepare_mict_dataset(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        transition_frames=args.transition_frames,
        min_words=args.min_words,
        max_words=args.max_words,
        max_samples=args.max_samples,
        train_split=args.train_split,
    )


if __name__ == "__main__":
    main()
