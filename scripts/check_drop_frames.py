#!/usr/bin/env python3
"""
Check how many frames to drop at START and END of each word to avoid rest pose.

Usage:
    python scripts/check_drop_frames.py \
        --data_dir /mnt/ngan/vsl_data/sequences \
        --sample 300
"""

import numpy as np
import argparse
from pathlib import Path
import random

LEFT_WRIST_Y  = 15 * 3 + 1
RIGHT_WRIST_Y = 16 * 3 + 1


def wrist_motion(data_flat):
    lw = data_flat[:, LEFT_WRIST_Y]
    rw = data_flat[:, RIGHT_WRIST_Y]
    lv = np.abs(np.diff(lw))
    rv = np.abs(np.diff(rw))
    return (lv + rv) / 2


def analyze_file(path):
    try:
        d = np.load(str(path)).astype(np.float32)
        if d.ndim == 3:
            d = d.reshape(d.shape[0], -1)
        if d.shape[0] < 25:
            return None
        motion = wrist_motion(d)  # (frames-1,)

        last_motions  = {}
        first_motions = {}
        max_drop = min(20, len(motion) - 5)

        for drop in range(0, max_drop + 1):
            # Last frame anchor after dropping `drop` frames from end
            last_idx = -drop - 1 if drop > 0 else -1
            last_motions[drop] = float(motion[last_idx])

            # First frame anchor after dropping `drop` frames from start
            first_idx = drop
            first_motions[drop] = float(motion[min(first_idx, len(motion)-1)])

        return last_motions, first_motions
    except Exception:
        return None


def print_analysis(drop_data, label, threshold):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"{'Drop':>6} | {'Mean motion':>12} | {'% still (<{:.3f})'.format(threshold):>20}")
    print("-" * 45)

    best_drop = 0
    best_pct  = 100.0
    for drop in range(0, 21):
        vals = drop_data.get(drop, [])
        if not vals:
            continue
        mean_m   = np.mean(vals)
        pct_still = np.mean([v < threshold for v in vals]) * 100
        marker = " ← best" if pct_still < best_pct else ""
        if pct_still < best_pct:
            best_pct  = pct_still
            best_drop = drop
        print(f"{drop:>6} | {mean_m:>12.5f} | {pct_still:>18.1f}%{marker}")

    print(f"\n→ Recommended: {best_drop} frames  ({best_pct:.1f}% rest at anchor)")
    return best_drop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/ngan/vsl_data/sequences')
    parser.add_argument('--sample', type=int, default=300)
    parser.add_argument('--motion_threshold', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_files = list(Path(args.data_dir).glob('*/*.npy'))
    sample = random.sample(all_files, min(args.sample, len(all_files)))
    print(f"Analyzing {len(sample)} files...\n")

    last_data  = {d: [] for d in range(0, 21)}
    first_data = {d: [] for d in range(0, 21)}

    for f in sample:
        res = analyze_file(f)
        if res is None:
            continue
        last_motions, first_motions = res
        for drop, motion in last_motions.items():
            last_data[drop].append(motion)
        for drop, motion in first_motions.items():
            first_data[drop].append(motion)

    rec_last  = print_analysis(last_data,  "DROP LAST FRAMES (end rest pose)",   args.motion_threshold)
    rec_first = print_analysis(first_data, "DROP FIRST FRAMES (start rest pose)", args.motion_threshold)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  --drop_last_frames  {rec_last}")
    print(f"  --drop_first_frames {rec_first}")


if __name__ == '__main__':
    main()
