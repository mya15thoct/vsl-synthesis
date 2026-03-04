#!/usr/bin/env python3
"""
MicT Inference Script — test best.pt checkpoint
arXiv 2508.04049

Usage:
    PYTHONPATH=. python -m src.methods.mict.infer_mict \
        --model_path models/mict/best.pt \
        --word_dirs /mnt/ngan/vsl_data/sequences/HELLO /mnt/ngan/vsl_data/sequences/CAM_ON \
        --output_npy output_transition.npy

    # Or test ngẫu nhiên từ data:
    PYTHONPATH=. python -m src.methods.mict.infer_mict \
        --model_path models/mict/best.pt \
        --sequences_dir /mnt/ngan/vsl_data/sequences \
        --num_words 3 \
        --output_npy output_transition.npy
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path

from .model_mict import MicTDiffusionModel, MicTDDPMScheduler

GLOBAL_MIN = -2.0
GLOBAL_MAX = 2.0


def normalize(x):
    return (np.clip(x, GLOBAL_MIN, GLOBAL_MAX) - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)


def denormalize(x):
    return x * (GLOBAL_MAX - GLOBAL_MIN) + GLOBAL_MIN


def load_npy(path):
    data = np.load(str(path)).astype(np.float32)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    return data


# Reuse canonical normalization logic from prepare_data_mict
IDX_L_HIP      = 23 * 3
IDX_R_HIP      = 24 * 3
IDX_L_SHOULDER = 11 * 3
IDX_R_SHOULDER = 12 * 3

def canonical_normalize_skeleton(data_flat):
    """Normalize scale/position: hip center → (0.5,0.5), torso height → 0.25."""
    lhip = data_flat[:, IDX_L_HIP:IDX_L_HIP+2]
    rhip = data_flat[:, IDX_R_HIP:IDX_R_HIP+2]
    lsho = data_flat[:, IDX_L_SHOULDER:IDX_L_SHOULDER+2]
    rsho = data_flat[:, IDX_R_SHOULDER:IDX_R_SHOULDER+2]
    hip_center = (lhip + rhip) / 2
    sho_center = (lsho + rsho) / 2
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
    """Trim rest frames bằng vị trí tay. START: smooth>0.4, END: smooth>0.7."""
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


def snap_hands_to_wrist(frames_raw):
    """
    Dịch chuyển toàn bộ hand landmarks để hand[0] khớp với pose wrist.
    frames_raw: (T, 1659) trong space denormalized
    
    Real data layout: pose(0:99) + face(99:1533) + lhand(1533:1596) + rhand(1596:1659)
    """
    out = np.array(frames_raw, dtype=np.float32)
    lw_s, lw_e = 45, 48      # pose kp15 (L_WRIST) x,y,z
    rw_s, rw_e = 48, 51      # pose kp16 (R_WRIST) x,y,z
    lh_s, lh_e = 1533, 1596  # lhand 21kp × 3
    rh_s, rh_e = 1596, 1659  # rhand 21kp × 3
    for t in range(len(out)):
        shift_l = out[t, lw_s:lw_e] - out[t, lh_s:lh_s+3]
        out[t, lh_s:lh_e] = out[t, lh_s:lh_e] + np.tile(shift_l.astype(np.float32), 21)
        shift_r = out[t, rw_s:rw_e] - out[t, rh_s:rh_s+3]
        out[t, rh_s:rh_e] = out[t, rh_s:rh_e] + np.tile(shift_r.astype(np.float32), 21)
    return out


def build_masked_seq(word_npys, transition_frames=10,
                     canonical_norm=True, hip_margin=0.05):
    """Build masked_sequence (words + zero transitions) for inference."""
    segments = []
    for p in word_npys:
        s = load_npy(p)
        if canonical_norm:
            s = canonical_normalize_skeleton(s)
        s = adaptive_trim(s, hip_margin=hip_margin)
        s = normalize(s)
        segments.append(s)
    feat_dim = segments[0].shape[1]
    zeros = np.zeros((transition_frames, feat_dim), dtype=np.float32)
    parts = []
    for i, seg in enumerate(segments):
        parts.append(seg)
        if i < len(segments) - 1:
            parts.append(zeros.copy())
    masked_seq = np.concatenate(parts, axis=0)
    return masked_seq, [len(s) for s in segments], segments


def run_inference(
    model_path: str,
    word_npys: list,
    transition_frames: int = 10,
    num_inference_steps: int = 50,
    device: str = None,
    drop_last_frames: int = 8,
    drop_first_frames: int = 3,
    canonical_norm: bool = True,
    hip_margin: float = 0.05,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location=device)
    cfg  = ckpt.get('config', {})
    model = MicTDiffusionModel(
        input_dim    = cfg.get('input_dim',     1659),
        hidden_dim   = cfg.get('hidden_dim',    512),
        num_heads    = cfg.get('num_heads',     8),
        enc_layers   = cfg.get('enc_layers',    4),
        dec_layers   = cfg.get('dec_layers',    6),
        num_timesteps= cfg.get('num_timesteps', 1000),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  Epoch: {ckpt.get('epoch', '?')}  |  Val loss: {ckpt.get('val_loss', '?'):.4f}")

    # Build observation condition
    masked_seq, word_lengths, original_segments_norm = build_masked_seq(
        word_npys, transition_frames, canonical_norm, hip_margin
    )
    print(f"\nInput:")
    print(f"  Words: {len(word_npys)} ({word_lengths} frames each)")
    print(f"  Transition frames: {transition_frames} per gap")
    print(f"  Total seq length: {masked_seq.shape[0]} frames")

    # Scheduler + sampling
    scheduler = MicTDDPMScheduler(num_timesteps=cfg.get('num_timesteps', 1000)).to(device)

    masked_tensor = torch.tensor(masked_seq, dtype=torch.float32, device=device).unsqueeze(0)

    print(f"\nRunning DDPM sampling ({num_inference_steps} steps)...")
    with torch.no_grad():
        generated = scheduler.sample(
            model, masked_tensor,
            num_inference_steps=num_inference_steps,
        )  # (1, T_total, 1659)

    generated = generated[0].cpu().numpy()  # (T_total, 1659)

    # Clamp trước khi denormalize — lệch nhỏ bị phóng to 4x nếu không clamp
    generated = np.clip(generated, 0.0, 1.0)

    # Denormalize: [0,1] → [-2,2]
    generated_raw = denormalize(generated)

    # Extract only transition frames
    transition_frames_out = []
    pos = 0
    for i, wlen in enumerate(word_lengths):
        pos += wlen
        if i < len(word_lengths) - 1:
            transition = generated_raw[pos:pos + transition_frames]
            transition_frames_out.append(transition)
            pos += transition_frames

    print(f"\nGenerated {len(transition_frames_out)} transition segment(s)")
    for i, t in enumerate(transition_frames_out):
        print(f"  Transition {i+1}: {t.shape} | range [{t.min():.3f}, {t.max():.3f}]")

    # Apply snap to each transition segment
    transition_frames_out = [snap_hands_to_wrist(t) for t in transition_frames_out]
    print("  Snapped hand landmarks to pose wrist positions")

    # Assemble final sequence using original word frames and generated transitions
    final_sequence_parts = []
    for i, seg_norm in enumerate(original_segments_norm):
        final_sequence_parts.append(denormalize(seg_norm)) # Denormalize original segments
        if i < len(original_segments_norm) - 1:
            final_sequence_parts.append(transition_frames_out[i])

    final_sequence = np.concatenate([np.asarray(p, dtype=np.float32) for p in final_sequence_parts], axis=0)

    return final_sequence, transition_frames_out


def main():
    parser = argparse.ArgumentParser(description='MicT Inference')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--word_dirs', nargs='+', default=None,
                        help='Folders containing .npy files for each word')
    parser.add_argument('--sequences_dir', default=None,
                        help='Root sequences dir (random pick if --word_dirs not given)')
    parser.add_argument('--num_words', type=int, default=3)
    parser.add_argument('--transition_frames', type=int, default=10)
    parser.add_argument('--inference_steps', type=int, default=50)
    parser.add_argument('--output_npy', default='mict_output.npy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drop_last_frames', type=int, default=8,
                        help='Bỏ N frame cuối mỗi word (default: 8)')
    parser.add_argument('--drop_first_frames', type=int, default=3,
                        help='Bỏ N frame đầu mỗi word (default: 3)')
    parser.add_argument('--no_canonical', action='store_true',
                        help='Tắt canonical normalization')
    parser.add_argument('--hip_margin', type=float, default=0.05,
                        help='Wrist phải cao hơn hông bao nhiêu để xem là active (default: 0.05)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve word npy files
    if args.word_dirs:
        word_npys = []
        for d in args.word_dirs:
            npys = sorted(Path(d).glob('*.npy'))
            if not npys:
                raise FileNotFoundError(f"No .npy found in {d}")
            word_npys.append(random.choice(npys))
        print(f"Words: {[str(p) for p in word_npys]}")
    elif args.sequences_dir:
        seq_dir = Path(args.sequences_dir)
        word_folders = [d for d in seq_dir.iterdir() if d.is_dir()]
        chosen = random.sample(word_folders, min(args.num_words, len(word_folders)))
        word_npys = []
        for folder in chosen:
            npys = sorted(folder.glob('*.npy'))
            if npys:
                word_npys.append(random.choice(npys))
        print(f"Random words: {[p.parent.name for p in word_npys]}")
    else:
        raise ValueError("Provide --word_dirs or --sequences_dir")

    generated, transitions = run_inference(
        model_path=args.model_path,
        word_npys=word_npys,
        transition_frames=args.transition_frames,
        num_inference_steps=args.inference_steps,
        canonical_norm=not args.no_canonical,
        hip_margin=args.hip_margin,
    )


    np.save(args.output_npy, generated)
    print(f"\nSaved full generated sequence → {args.output_npy}")
    print(f"Shape: {generated.shape}")


if __name__ == '__main__':
    main()
