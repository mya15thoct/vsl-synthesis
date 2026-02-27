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


def build_masked_seq(word_npys, transition_frames=10):
    """Build masked_sequence (words + zero transitions) for inference."""
    segments = [normalize(load_npy(p)) for p in word_npys]
    feat_dim = segments[0].shape[1]
    zeros = np.zeros((transition_frames, feat_dim), dtype=np.float32)

    parts = []
    for i, seg in enumerate(segments):
        parts.append(seg)
        if i < len(segments) - 1:
            parts.append(zeros.copy())

    masked_seq = np.concatenate(parts, axis=0)  # (T_total, 1659)
    return masked_seq, [len(s) for s in segments]


def run_inference(
    model_path: str,
    word_npys: list,
    transition_frames: int = 10,
    num_inference_steps: int = 50,
    device: str = None,
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
    masked_seq, word_lengths = build_masked_seq(word_npys, transition_frames)
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

    return generated_raw, transition_frames_out


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
    )

    np.save(args.output_npy, generated)
    print(f"\nSaved full generated sequence → {args.output_npy}")
    print(f"Shape: {generated.shape}")


if __name__ == '__main__':
    main()
