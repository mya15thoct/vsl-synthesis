#!/usr/bin/env python3
"""
Intra-Word Sliding Window Data Preparation for MicT.

Thay vì dùng linear interpolation làm GT cho transitions,
strategy mới: cắt 1 video từ làm 3 phần bằng sliding window,
dùng phần giữa (thật) làm GT → model học real motion dynamics.

Format output tương thích hoàn toàn với MicTDataset (masked/interp/mask).

Usage:
    PYTHONPATH=. python src/methods/mict/prepare_data_intra.py \\
        --data_dir /mnt/ngan/vsl_data/sequences \\
        --output_dir /mnt/ngan/vsl_data/mict_intra \\
        --gap_size 10 \\
        --min_context 10 \\
        --stride 3
"""

import argparse
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Reuse helpers from prepare_data_mict
from src.methods.mict.prepare_data_mict import (
    load_npy,
    canonical_normalize_skeleton,
    adaptive_trim,
    normalize_skeleton,
)

GLOBAL_MIN = -2.0
GLOBAL_MAX =  2.0


def sliding_window_samples(seg, gap_size=10, min_context=10, stride=3):
    """
    Tạo tất cả samples sliding window từ 1 video segment.

    Args:
        seg:         (T, 1659) normalized sequence
        gap_size:    số frames bị che (= transition_frames khi inference)
        min_context: số frames context tối thiểu ở mỗi bên
        stride:      bước nhảy của cửa sổ

    Yields:
        dict với masked_sequence, interpolated_sequence (=real frames), frame_mask
    """
    T = len(seg)
    feat_dim = seg.shape[1]

    # Cần ít nhất: min_context + gap_size + min_context frames
    min_len = min_context + gap_size + min_context
    if T < min_len:
        return  # video quá ngắn

    # Slide gap từ vị trí min_context đến T - min_context - gap_size
    gap_start_min = min_context
    gap_start_max = T - min_context - gap_size

    for gap_start in range(gap_start_min, gap_start_max + 1, stride):
        gap_end = gap_start + gap_size  # exclusive

        # Context: toàn bộ frame bên ngoài gap
        context_before = seg[:gap_start]    # (gap_start, 1659)
        real_gap       = seg[gap_start:gap_end]  # (gap_size, 1659) — GT thật
        context_after  = seg[gap_end:]      # (T-gap_end, 1659)

        # masked_sequence: zero tại gap (input model)
        masked_parts = [
            context_before,
            np.zeros((gap_size, feat_dim), dtype=np.float32),
            context_after,
        ]
        masked_seq = np.concatenate(masked_parts, axis=0)  # (T, 1659)

        # interpolated_sequence (GT): real frames tại gap thay vì linear interp
        gt_seq = seg.copy()   # full real sequence = GT

        # frame_mask: 1 = gap position (transition), 0 = context
        frame_mask = np.zeros(T, dtype=np.float32)
        frame_mask[gap_start:gap_end] = 1.0

        yield {
            'masked_sequence':       masked_seq,
            'interpolated_sequence': gt_seq,
            'frame_mask':            frame_mask,
            'gap_start':             gap_start,
            'gap_size':              gap_size,
            'total_frames':          T,
        }


def prepare_intra_dataset(
    data_dir: Path,
    output_dir: Path,
    gap_size: int = 10,
    min_context: int = 10,
    stride: int = 3,
    max_samples: int = 100000,
    train_split: float = 0.9,
    hip_margin: float = 0.05,
    canonical_norm: bool = True,
):
    train_dir = output_dir / 'train'
    val_dir   = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Collect all .npy paths
    all_npys = sorted(data_dir.rglob('*.npy'))
    print(f"\nFound {len(all_npys)} .npy files in {data_dir}")
    print(f"Gap size: {gap_size} | Min context: {min_context} | Stride: {stride}")

    all_samples = []

    for npy_path in tqdm(all_npys, desc="Generating samples"):
        try:
            seg = load_npy(npy_path)
            if canonical_norm:
                seg = canonical_normalize_skeleton(seg)
            seg = adaptive_trim(seg, hip_margin=hip_margin)
            seg = normalize_skeleton(seg)

            if len(seg) < min_context * 2 + gap_size:
                continue  # too short

            for sample in sliding_window_samples(seg, gap_size, min_context, stride):
                sample['word'] = npy_path.parent.name
                sample['source'] = str(npy_path)
                all_samples.append(sample)

                if len(all_samples) >= max_samples:
                    break

        except Exception as e:
            print(f"  Skip {npy_path.name}: {e}")
            continue

        if len(all_samples) >= max_samples:
            break

    print(f"\nTotal samples: {len(all_samples)}")

    # Shuffle và split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_split)
    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:]

    print(f"Saving {len(train_samples)} train, {len(val_samples)} val...")

    def save_samples(samples, out_dir, desc):
        for idx, s in enumerate(tqdm(samples, desc=desc)):
            np.savez_compressed(
                out_dir / f"sample_{idx:06d}.npz",
                masked_sequence       = s['masked_sequence'],
                interpolated_sequence = s['interpolated_sequence'],
                frame_mask            = s['frame_mask'],
                word_lengths          = np.array([s['gap_start'],
                                                  s['gap_size'],
                                                  s['total_frames'] - s['gap_start'] - s['gap_size']],
                                                 dtype=np.int32),
                metadata=json.dumps({
                    'word':      s['word'],
                    'gap_start': s['gap_start'],
                    'gap_size':  s['gap_size'],
                    'num_words': 1,
                })
            )

    save_samples(train_samples, train_dir, "Saving train")
    save_samples(val_samples,   val_dir,   "Saving val")

    stats = {
        'total': len(all_samples),
        'train': len(train_samples),
        'val':   len(val_samples),
        'gap_size': gap_size,
        'min_context': min_context,
        'stride': stride,
        'data_type': 'intra_word_sliding_window',
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! → {output_dir}")
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)}")


def main():
    parser = argparse.ArgumentParser(description='Intra-Word Sliding Window Data Prep for MicT')
    parser.add_argument('--data_dir',     required=True,  help='sequences/ directory')
    parser.add_argument('--output_dir',   default='/mnt/ngan/vsl_data/mict_intra')
    parser.add_argument('--gap_size',     type=int,   default=10,
                        help='Gap size (= transition_frames at inference)')
    parser.add_argument('--min_context',  type=int,   default=10,
                        help='Min context frames on each side of gap')
    parser.add_argument('--stride',       type=int,   default=3,
                        help='Sliding window stride')
    parser.add_argument('--max_samples',  type=int,   default=100000)
    parser.add_argument('--train_split',  type=float, default=0.9)
    parser.add_argument('--hip_margin',   type=float, default=0.05)
    parser.add_argument('--no_canonical', action='store_true')
    args = parser.parse_args()

    prepare_intra_dataset(
        data_dir      = Path(args.data_dir),
        output_dir    = Path(args.output_dir),
        gap_size      = args.gap_size,
        min_context   = args.min_context,
        stride        = args.stride,
        max_samples   = args.max_samples,
        train_split   = args.train_split,
        hip_margin    = args.hip_margin,
        canonical_norm= not args.no_canonical,
    )


if __name__ == '__main__':
    main()
