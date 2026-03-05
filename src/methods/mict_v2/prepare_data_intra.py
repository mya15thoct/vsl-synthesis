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
from scipy.interpolate import interp1d

GLOBAL_MIN = -2.0
GLOBAL_MAX =  2.0

IDX_L_HIP      = 23 * 3
IDX_R_HIP      = 24 * 3
IDX_L_SHOULDER = 11 * 3
IDX_R_SHOULDER = 12 * 3


def load_npy(path):
    data = np.load(str(path)).astype(np.float32)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    return data


def normalize_skeleton(skeleton):
    clipped = np.clip(skeleton, GLOBAL_MIN, GLOBAL_MAX)
    return (clipped - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)


def canonical_normalize_skeleton(data_flat):
    lhip_xy = data_flat[:, IDX_L_HIP:IDX_L_HIP+2]
    rhip_xy = data_flat[:, IDX_R_HIP:IDX_R_HIP+2]
    lsho_xy = data_flat[:, IDX_L_SHOULDER:IDX_L_SHOULDER+2]
    rsho_xy = data_flat[:, IDX_R_SHOULDER:IDX_R_SHOULDER+2]
    hip_center = (lhip_xy + rhip_xy) / 2
    sho_center = (lsho_xy + rsho_xy) / 2
    hip_mean   = hip_center.mean(axis=0)
    torso_h    = max(np.linalg.norm(hip_center - sho_center, axis=1).mean(), 1e-3)
    shift      = np.array([0.5, 0.5], dtype=np.float32) - hip_mean
    result     = data_flat.copy()
    result[:, 0::3] += shift[0]
    result[:, 1::3] += shift[1]
    scale = 0.25 / torso_h
    result[:, 0::3] = (result[:, 0::3] - 0.5) * scale + 0.5
    result[:, 1::3] = (result[:, 1::3] - 0.5) * scale + 0.5
    return result


def adaptive_trim(seg, hip_margin=0.05, min_keep=10, window=5):
    LW_Y  = seg[:, 46]
    RW_Y  = seg[:, 49]
    HIP_Y = (seg[:, 70] + seg[:, 73]) / 2
    threshold_y = HIP_Y - hip_margin
    active = (LW_Y < threshold_y) | (RW_Y < threshold_y)
    kernel = np.ones(window) / window
    smooth = np.convolve(active.astype(float), kernel, mode='same')
    start_cands = np.where(smooth > 0.4)[0]
    end_cands   = np.where(smooth > 0.7)[0]
    if len(start_cands) == 0:
        pad = len(seg) // 10
        return seg[pad: len(seg) - pad] if len(seg) > 2 * pad + min_keep else seg
    start = int(start_cands[0])
    end   = min(int(end_cands[-1]) + 1, len(seg)) if len(end_cands) > 0 \
            else min(int(start_cands[-1]) + 1, len(seg))
    if end - start < min_keep:
        mid = (start + end) // 2
        start = max(0, mid - min_keep // 2)
        end   = min(len(seg), start + min_keep)
    return seg[start:end]


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def time_scale_segment(seg, scale_range=(0.8, 1.2)):
    """
    Resample segment theo tốc độ ngẫu nhiên.
    """
    T, D = seg.shape
    scale = random.uniform(*scale_range)
    new_T = max(4, int(round(T * scale)))
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_T)
    f = interp1d(x_old, seg, axis=0, kind='linear')
    return f(x_new).astype(np.float32)

def add_gaussian_noise(seg, std=0.008):
    """
    Thêm Gaussian noise nhỏ vào keypoints trong không gian [0,1].
    """
    noise = np.random.normal(0.0, std, seg.shape).astype(np.float32)
    return np.clip(seg + noise, 0.0, 1.0)


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
    stride: int = 1,
    max_samples: int = 100000,
    train_split: float = 0.9,
    hip_margin: float = 0.05,
    canonical_norm: bool = True,
    augment: bool = True,
    noise_std: float = 0.008,
    time_scale_range: tuple = (0.8, 1.2),
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
            seg_raw = load_npy(npy_path)
            if canonical_norm:
                seg_raw = canonical_normalize_skeleton(seg_raw)
            seg_raw = adaptive_trim(seg_raw, hip_margin=hip_margin)
            seg_base = normalize_skeleton(seg_raw)

            # Generate samples for original segment
            if len(seg_base) >= min_context * 2 + gap_size:
                for sample in sliding_window_samples(seg_base, gap_size, min_context, stride):
                    sample['word'] = npy_path.parent.name
                    sample['source'] = str(npy_path)
                    sample['is_aug'] = False
                    all_samples.append(sample)

            # Generate samples for augmented segment if enabled
            if augment:
                seg_aug = time_scale_segment(seg_base, scale_range=time_scale_range)
                seg_aug = add_gaussian_noise(seg_aug, std=noise_std)
                
                if len(seg_aug) >= min_context * 2 + gap_size:
                    for sample in sliding_window_samples(seg_aug, gap_size, min_context, stride):
                        sample['word'] = npy_path.parent.name
                        sample['source'] = f"{npy_path}_aug"
                        sample['is_aug'] = True
                        all_samples.append(sample)

        except Exception as e:
            print(f"  Skip {npy_path.name}: {e}")
            continue

        if len(all_samples) >= max_samples:
            break

    print(f"\nTotal samples: {len(all_samples)}")

    # Shuffle và split
    random.shuffle(all_samples)
    
    # Split train/val CAREFULLY: val MUST NOT contain augmented samples
    # We first collect clean samples, split them, then assign ALL augmented samples to train
    clean_samples = [s for s in all_samples if not s['is_aug']]
    aug_samples   = [s for s in all_samples if s['is_aug']]
    
    split_idx = int(len(clean_samples) * train_split)
    train_samples = clean_samples[:split_idx] + aug_samples
    val_samples   = clean_samples[split_idx:]
    
    # Shuffle train again so clean and aug are mixed
    random.shuffle(train_samples)
    
    # Truncate to max_samples if needed (keep ratio of aug/clean random by truncating mixed array)
    train_samples = train_samples[:int(max_samples * train_split)]
    val_samples   = val_samples[:int(max_samples * (1 - train_split))]

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
    parser.add_argument('--stride',       type=int,   default=1,
                        help='Sliding window stride (default: 1 = mỗi vị trí gap đều lấy)')
    parser.add_argument('--max_samples',  type=int,   default=100000)
    parser.add_argument('--train_split',  type=float, default=0.9)
    parser.add_argument('--hip_margin',   type=float, default=0.05)
    parser.add_argument('--no_canonical', action='store_true')
    parser.add_argument('--no_aug',       action='store_true',
                        help='Tắt augmentation (time scaling + gaussian noise)')
    parser.add_argument('--noise_std',    type=float, default=0.008,
                        help='Gaussian noise std (default: 0.008, trong [0,1] space)')
    parser.add_argument('--time_scale_min', type=float, default=0.8,
                        help='Time scale min (default: 0.8 = 20%% faster)')
    parser.add_argument('--time_scale_max', type=float, default=1.2,
                        help='Time scale max (default: 1.2 = 20%% slower)')
    args = parser.parse_args()

    print(f"\nAugmentation: {'OFF (--no_aug)' if args.no_aug else 'ON'}")
    if not args.no_aug:
        print(f"  Time scaling: ×{args.time_scale_min}–{args.time_scale_max}")
        print(f"  Gaussian noise: std={args.noise_std}")

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
        augment       = not args.no_aug,
        noise_std     = args.noise_std,
        time_scale_range=(args.time_scale_min, args.time_scale_max),
    )


if __name__ == '__main__':
    main()
