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


# MediaPipe pose indices (x3 = no visibility format, 33kp x 3)
IDX_L_HIP      = 23 * 3   # features 69-71
IDX_R_HIP      = 24 * 3
IDX_L_SHOULDER = 11 * 3
IDX_R_SHOULDER = 12 * 3


def canonical_normalize_skeleton(data_flat):
    """
    Normalize skeleton để loại bỏ scale/position khác nhau giữa các video.

    Bước 1: Translate — dịch sao cho hip center trung bình = (0.5, 0.5)
    Bước 2: Scale — chia cho torso height trung bình (shoulder → hip)

    Giữ nguyên tất cả các kp khác (hand, face) vì chúng có coordinate
    tương đối với đầu mối pose.
    """
    T = data_flat.shape[0]

    # Extract hip and shoulder y-coords
    lhip_xy  = data_flat[:, IDX_L_HIP:IDX_L_HIP+2]       # (T, 2)
    rhip_xy  = data_flat[:, IDX_R_HIP:IDX_R_HIP+2]
    lsho_xy  = data_flat[:, IDX_L_SHOULDER:IDX_L_SHOULDER+2]
    rsho_xy  = data_flat[:, IDX_R_SHOULDER:IDX_R_SHOULDER+2]

    hip_center  = (lhip_xy + rhip_xy)   / 2   # (T, 2) per frame
    sho_center  = (lsho_xy + rsho_xy)   / 2

    # Use MEAN across frames for stable canonical transform
    hip_mean    = hip_center.mean(axis=0)      # (2,) [x_mean, y_mean]
    torso_h     = np.linalg.norm(
        hip_center - sho_center, axis=1
    ).mean()                                    # scalar
    torso_h     = max(torso_h, 1e-3)

    # Translate: shift hip center to (0.5, 0.5)
    target_hip  = np.array([0.5, 0.5], dtype=np.float32)
    shift       = target_hip - hip_mean         # (2,)

    result = data_flat.copy()
    # Apply shift to ALL x,y pairs (every 3rd feature starting at 0 and 1)
    # Feature layout: [x0, y0, z0, x1, y1, z1, ...]
    result[:, 0::3] += shift[0]   # shift all x
    result[:, 1::3] += shift[1]   # shift all y

    # Scale around new hip center (0.5, 0.5)
    canonical_torso = 0.25   # target torso height in normalized space
    scale = canonical_torso / torso_h
    # Scale x,y relative to hip center
    result[:, 0::3] = (result[:, 0::3] - 0.5) * scale + 0.5
    result[:, 1::3] = (result[:, 1::3] - 0.5) * scale + 0.5

    return result


def adaptive_trim(seg, hip_margin=0.05, min_keep=10, window=5):
    """
    Trim rest frames bằng vị trí tay — không dùng motion (tránh noise).

    Khi ký: tay PHẢI giơ lên cao hơn hông (quy tắc vật lý).
    Khi nghỉ: tay ở bên hông, ngang hoặc thấp hơn hông.

    Sau canonical_normalize_skeleton:
        - Hip center y ≈ 0.5 (trong image coord, y lớn hơn = thấp hơn)
        - Wrist lúc nghỉ ≈ y ≥ 0.5
        - Wrist lúc ký ≈ y < 0.5 - hip_margin

    Pose layout (no visibility, x,y,z per kp):
        kp 11: L_SHOULDER  → feat [33,34,35]
        kp 12: R_SHOULDER  → feat [36,37,38]
        kp 15: L_WRIST     → feat [45,46,47]  ← y = feat 46
        kp 16: R_WRIST     → feat [48,49,50]  ← y = feat 49
        kp 23: L_HIP       → feat [69,70,71]  ← y = feat 70
        kp 24: R_HIP       → feat [72,73,74]  ← y = feat 73
    """
    LW_Y   = seg[:, 46]                        # left wrist y  (T,)
    RW_Y   = seg[:, 49]                        # right wrist y
    HIP_Y  = (seg[:, 70] + seg[:, 73]) / 2    # hip center y  (T,)

    # Active = wrist is above hip (smaller y = higher in image)
    threshold_y = HIP_Y - hip_margin           # (T,)
    active = (LW_Y < threshold_y) | (RW_Y < threshold_y)  # (T,) bool

    # Smooth with window to remove isolated glitches
    kernel      = np.ones(window) / window
    smooth      = np.convolve(active.astype(float), kernel, mode='same')
    active_s    = smooth > 0.4   # majority of window must be active

    candidates = np.where(active_s)[0]
    if len(candidates) == 0:
        # Fallback: if position-based fails (bad tracking), keep middle 80%
        pad = len(seg) // 10
        return seg[pad: len(seg) - pad] if len(seg) > 2 * pad + min_keep else seg

    start = int(candidates[0])
    end   = min(int(candidates[-1]) + 2, len(seg))

    if end - start < min_keep:
        mid   = (start + end) // 2
        start = max(0, mid - min_keep // 2)
        end   = min(len(seg), start + min_keep)

    return seg[start:end]




def build_sentence_sample(word_paths, transition_frames=10,
                           hip_margin=0.05, min_seg_frames=8,
                           canonical_norm=True):
    """
    Ghép N từ thành 1 sequence câu.

    full_sequence:         word frames only (T_words, 1659)
    masked_sequence:       word frames + zero transitions (T_total, 1659)  — inference obs
    interpolated_sequence: word frames + LINEAR INTERP transitions (T_total, 1659) — training GT

    Adaptive trim: wrist phải cao hơn hông ít nhất hip_margin (position-based)
    """
    segments = []
    for p in word_paths:
        seg = load_npy(p)
        if canonical_norm:
            seg = canonical_normalize_skeleton(seg)
        seg = adaptive_trim(seg, hip_margin=hip_margin)
        seg = normalize_skeleton(seg)
        if len(seg) < min_seg_frames:
            return None
        segments.append(seg)

    feat_dim = segments[0].shape[1]  # 1659
    word_lengths = [len(seg) for seg in segments]

    # full_sequence = concat word frames only (no transitions)
    full_sequence = np.concatenate(segments, axis=0)   # (T_words, 1659)

    # Build T_total sequences
    masked_parts = []
    interp_parts = []
    mask_parts   = []

    for i, seg in enumerate(segments):
        masked_parts.append(seg)
        interp_parts.append(seg)
        mask_parts.append(np.zeros(len(seg), dtype=np.float32))   # word → mask=0

        if i < len(segments) - 1:
            # last frame of word[i], first frame of word[i+1]
            end_pose   = segments[i][-1]       # (1659,)
            start_pose = segments[i + 1][0]    # (1659,)

            # zeros transition (for inference obs)
            masked_parts.append(np.zeros((transition_frames, feat_dim), dtype=np.float32))

            # linear interpolation transition (for training GT)
            alphas = np.linspace(0.0, 1.0, transition_frames + 2)[1:-1]  # exclude endpoints
            interp = np.stack([
                (1.0 - a) * end_pose + a * start_pose
                for a in alphas
            ], axis=0)  # (transition_frames, 1659)
            interp_parts.append(interp)

            mask_parts.append(np.ones(transition_frames, dtype=np.float32))  # transition → 1

    masked_sequence       = np.concatenate(masked_parts, axis=0)  # (T_total, 1659)
    interpolated_sequence = np.concatenate(interp_parts, axis=0)  # (T_total, 1659)
    frame_mask            = np.concatenate(mask_parts,   axis=0)  # (T_total,)

    return {
        'full_sequence':         full_sequence,          # (T_words, 1659)
        'masked_sequence':       masked_sequence,        # (T_total, 1659) — zeros at transitions
        'interpolated_sequence': interpolated_sequence,  # (T_total, 1659) — lerp at transitions
        'frame_mask':            frame_mask,             # (T_total,) — 1=transition
        'word_lengths':          word_lengths,
        'num_words':             len(segments),
    }


def prepare_mict_dataset(
    data_dir: Path,
    output_dir: Path,
    transition_frames: int = 10,
    min_words: int = 2,
    max_words: int = 5,
    samples_per_combo: int = 1,
    max_samples: int = 50000,
    train_split: float = 0.9,
    seed: int = 42,
    hip_margin: float = 0.05,
    canonical_norm: bool = True,
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

    # Map: word_name → list of .npy files (chỉ lấy file có đúng feature dim)
    EXPECTED_DIM = 1659
    word_files = {}
    skipped_words = []
    for folder in word_folders:
        npys = sorted(folder.glob("*.npy"))
        if not npys:
            continue
        # Kiểm tra dim bằng cách đọc file đầu tiên
        try:
            sample = np.load(str(npys[0])).astype(np.float32)
            if sample.ndim == 3:
                sample = sample.reshape(sample.shape[0], -1)
            if sample.shape[1] != EXPECTED_DIM:
                skipped_words.append(f"{folder.name} (dim={sample.shape[1]})")
                continue
        except Exception:
            skipped_words.append(f"{folder.name} (unreadable)")
            continue
        word_files[folder.name] = npys

    word_names = list(word_files.keys())
    print(f"Words with data: {len(word_names)}")
    if skipped_words:
        print(f"Skipped {len(skipped_words)} words with wrong dim: {skipped_words}")

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
            s = build_sentence_sample(
                recipe['paths'], transition_frames,
                motion_threshold, 8, canonical_norm
            )
            if s is None:
                continue
            s['words'] = recipe['words']
            s['num_words'] = len(recipe['words']) # Ensure num_words is set correctly
            np.savez_compressed(
                train_dir / f"sample_{idx:06d}.npz",
                full_sequence=s['full_sequence'],
                masked_sequence=s['masked_sequence'],
                interpolated_sequence=s['interpolated_sequence'],
                frame_mask=s['frame_mask'].astype(np.float32),
                word_lengths=np.array(s['word_lengths'], dtype=np.int32),
                metadata=json.dumps({'words': s['words'], 'num_words': s['num_words']})
            )
        except Exception as e:
            print(f"Error processing train sample {idx} (words: {recipe['words']}): {e}")
            continue

    for idx, recipe in enumerate(tqdm(val_recipes, desc="Saving val")):
        try:
            s = build_sentence_sample(
                recipe['paths'], transition_frames,
                hip_margin, 8, canonical_norm
            )
            if s is None:
                continue

            s['words'] = recipe['words']
            s['num_words'] = len(recipe['words']) # Ensure num_words is set correctly
            np.savez_compressed(
                val_dir / f"sample_{idx:06d}.npz",
                full_sequence=s['full_sequence'],
                masked_sequence=s['masked_sequence'],
                interpolated_sequence=s['interpolated_sequence'],
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
    parser.add_argument('--hip_margin', type=float, default=0.05,
                        help='Wrist phải cao hơn hông ít nhất margin này (0.05 = 5%% torso height)')
    parser.add_argument('--no_canonical', action='store_true',
                        help='Tắt canonical normalization (scale/position)')
    args = parser.parse_args()

    prepare_mict_dataset(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        transition_frames=args.transition_frames,
        min_words=args.min_words,
        max_words=args.max_words,
        max_samples=args.max_samples,
        train_split=args.train_split,
        hip_margin=args.hip_margin,
        canonical_norm=not args.no_canonical,
    )



if __name__ == '__main__':
    main()
