#!/usr/bin/env python3
"""
Check how many frames to drop at end of each word to avoid rest pose.

Analyzes wrist motion in last N frames across many word files to find
the optimal drop_last_frames value.

Usage:
    python scripts/check_drop_frames.py \
        --data_dir /mnt/ngan/vsl_data/sequences \
        --sample 200
"""

import numpy as np
import argparse
from pathlib import Path
import random

# MediaPipe pose: 33 kp × 3 (no visibility) = 99 features
# Left wrist = index 15, right wrist = 16
LEFT_WRIST_Y  = 15 * 3 + 1   # y-coord of left wrist
RIGHT_WRIST_Y = 16 * 3 + 1   # y-coord of right wrist

def wrist_motion(data_flat):
    """Per-frame wrist motion (velocity magnitude)."""
    lw = data_flat[:, LEFT_WRIST_Y]
    rw = data_flat[:, RIGHT_WRIST_Y]
    # Use just Y (up/down motion = most informative for rest pose)
    lv = np.abs(np.diff(lw))
    rv = np.abs(np.diff(rw))
    return (lv + rv) / 2  # (frames-1,)

def analyze_file(path):
    try:
        d = np.load(str(path)).astype(np.float32)
        if d.ndim == 3:
            d = d.reshape(d.shape[0], -1)
        if d.shape[0] < 20:
            return None
        motion = wrist_motion(d)  # (frames-1,)

        # For each drop count (1..20), check motion at frame [-drop-1]
        # (the frame that WOULD be the last frame after dropping)
        results = {}
        max_drop = min(20, len(motion) - 5)
        for drop in range(0, max_drop + 1):
            frame_idx = -drop - 1 if drop > 0 else -1
            motion_at_anchor = motion[frame_idx]
            results[drop] = float(motion_at_anchor)
        return results
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/ngan/vsl_data/sequences')
    parser.add_argument('--sample', type=int, default=300,
                        help='Number of files to sample')
    parser.add_argument('--motion_threshold', type=float, default=0.003,
                        help='Motion below this = rest pose')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    all_files = list(data_dir.glob('*/*.npy'))
    sample = random.sample(all_files, min(args.sample, len(all_files)))
    print(f"Analyzing {len(sample)} files from {data_dir}\n")

    # Collect motion at anchor frame for each drop count
    drop_motions = {d: [] for d in range(0, 21)}

    for f in sample:
        res = analyze_file(f)
        if res is None:
            continue
        for drop, motion in res.items():
            drop_motions[drop].append(motion)

    print(f"{'Drop':>6} | {'Mean motion':>12} | {'% still (< {:.3f})'.format(args.motion_threshold):>20} | {'Verdict':>10}")
    print("-" * 60)

    prev_pct = 0
    recommended = 0
    for drop in range(0, 21):
        vals = drop_motions[drop]
        if not vals:
            continue
        mean_m = np.mean(vals)
        pct_still = np.mean([v < args.motion_threshold for v in vals]) * 100
        delta = pct_still - prev_pct

        if pct_still < 10 and drop > recommended:
            recommended = drop

        verdict = ""
        if pct_still >= 90:
            verdict = "✓ safe"
        elif pct_still >= 70:
            verdict = "~ ok"
        elif pct_still <= 20:
            verdict = "✗ still rest"

        print(f"{drop:>6} | {mean_m:>12.5f} | {pct_still:>18.1f}% | {verdict}")
        prev_pct = pct_still

    print(f"\n→ Recommended drop_last_frames ≈ {recommended}")
    print(f"  (anchor frame has < 10% rest pose frames at this drop count)")


if __name__ == '__main__':
    main()
