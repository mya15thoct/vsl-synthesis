#!/usr/bin/env python3
"""
Inference: MicT V2 — Latent Diffusion
Stage 2 inference: encode → DDPM latent diffusion → decode

Pipeline:
    1. Load word .npy files
    2. Canonical normalize + trim + normalize [0,1]
    3. Encode 1659D → 128D (frozen VAE)
    4. Build masked latent sequence (zeros at transitions)
    5. DDPM sampling in latent space (128D) → pred z0
    6. Decode 128D → 1659D
    7. Snap hands to wrist (post-process)
    8. Assemble final sequence

Usage:
    PYTHONPATH=. python -m src.methods.mict_v2.infer_mict_v2 \
        --ae_path models/mict_v2/autoencoder/best.pt \
        --model_path models/mict_v2/diffusion/best.pt \
        --word_dirs /mnt/ngan/vsl_data/sequences/Hello \
                    /mnt/ngan/vsl_data/sequences/Teacher \
        --output_npy mict_v2_output.npy
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path

from src.methods.mict_v2.autoencoder import PoseVAE
from src.methods.mict_v2.model_mict_v2 import MicTDiffusionModel, MicTDDPMScheduler
# Reuse preprocessing utilities from mict/
from src.methods.mict.infer_mict import (
    normalize, denormalize,
    load_npy,
    canonical_normalize_skeleton,
    adaptive_trim,
)

GLOBAL_MIN = -2.0
GLOBAL_MAX =  2.0


def build_masked_latent_seq(word_npys, vae, device,
                              transition_frames=10,
                              adaptive=True, hip_margin=0.05,
                              drop_first_frames=8, drop_last_frames=9):
    """
    Load words → canonical norm → trim → normalize → encode → build masked latent seq
    Returns:
        masked_latent: (T_total, latent_dim)
        word_lengths:  list of int
        orig_latents:  list of (T_wi, latent_dim) — original (not masked) latents
        orig_posesnorm: list of (T_wi, 1659) — for final assembly
    """
    segments_norm = []   # normalized 1659D
    for p in word_npys:
        s = load_npy(p)
        s = canonical_normalize_skeleton(s)
        if adaptive:
            s = adaptive_trim(s, hip_margin=hip_margin)
        else:
            if drop_first_frames > 0 and len(s) > drop_first_frames + drop_last_frames + 5:
                s = s[drop_first_frames:]
            if drop_last_frames > 0 and len(s) > drop_last_frames + 5:
                s = s[:-drop_last_frames]
        s = normalize(s)
        segments_norm.append(s)

    # Encode to latent
    word_lengths = [len(s) for s in segments_norm]
    latent_dim   = vae.latent_dim

    orig_latents  = []
    with torch.no_grad():
        for s in segments_norm:
            t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,1659)
            z = vae.encode(t, deterministic=True)                                   # (1,T,D)
            orig_latents.append(z.squeeze(0).cpu().numpy())                         # (T,D)

        # Compute "null latent" = VAE.encode(zero_pose) to match training distribution.
        # At training, masked_seqs has zero vectors at transition positions → encoded through VAE.
        # Using raw np.zeros at inference causes distribution mismatch → model outputs mean pose.
        zero_pose  = torch.zeros(1, 1, 1659, dtype=torch.float32, device=device)
        null_z     = vae.encode(zero_pose, deterministic=True)          # (1,1,D)
        null_latent = null_z.squeeze().cpu().numpy().astype(np.float32)  # (D,)

    # Build masked latent (null_latent at transitions, word latents at word positions)
    null_block = np.tile(null_latent, (transition_frames, 1))  # (tf, D)
    parts = []
    for i, z in enumerate(orig_latents):
        parts.append(np.array(z, dtype=np.float32))
        if i < len(orig_latents) - 1:
            parts.append(null_block.copy())
    masked_latent = np.concatenate(parts, axis=0)

    return masked_latent, word_lengths, orig_latents, segments_norm


def run_inference(ae_path, model_path, word_npys,
                   transition_frames=10,
                   num_inference_steps=50,
                   device=None,
                   adaptive=True, hip_margin=0.05,
                   drop_first_frames=8, drop_last_frames=9):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load VAE
    ae_ckpt = torch.load(ae_path, map_location=device)
    ae_cfg  = ae_ckpt['config']
    vae = PoseVAE(**ae_cfg).to(device)
    vae.load_state_dict(ae_ckpt['model_state_dict'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"Loaded VAE: latent_dim={ae_cfg['latent_dim']}, epoch={ae_ckpt['epoch']}")

    # Load diffusion model
    ckpt         = torch.load(model_path, map_location=device)
    cfg          = ckpt.get('config', {})
    use_velocity = ckpt.get('use_velocity', False)
    model = MicTDiffusionModel(
        input_dim    = ae_cfg['latent_dim'],
        hidden_dim   = cfg.get('hidden_dim',    512),
        num_heads    = cfg.get('num_heads',     8),
        enc_layers   = cfg.get('enc_layers',    4),
        dec_layers   = cfg.get('dec_layers',    6),
        num_timesteps= cfg.get('num_timesteps', 1000),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    mode_str = 'velocity+latent' if use_velocity else 'latent'
    print(f"Loaded {mode_str} diffusion model: epoch={ckpt.get('epoch','?')}, val={ckpt.get('val_loss','?'):.4f}")

    # Build inputs
    masked_latent, word_lengths, orig_latents, orig_segs_norm = build_masked_latent_seq(
        word_npys, vae, device,
        transition_frames, adaptive, hip_margin,
        drop_first_frames, drop_last_frames,
    )
    print(f"\nInput:")
    print(f"  Words: {len(word_npys)} ({word_lengths} frames each)")
    print(f"  Transition frames: {transition_frames} per gap")
    print(f"  Total latent seq: {masked_latent.shape[0]} frames × {masked_latent.shape[1]}D")

    # Convert masked latent to velocity if needed
    masked_np = masked_latent
    if use_velocity:
        ml_t = torch.tensor(masked_np, dtype=torch.float32, device=device).unsqueeze(0)
        prev  = torch.cat([torch.zeros_like(ml_t[:, :1, :]), ml_t[:, :-1, :]], dim=1)
        masked_np = (ml_t - prev).squeeze(0).cpu().numpy()

    # DDPM INPAINTING (RePaint-style):
    # Instead of soft conditioning, HARD PIN word frames at each denoising step.
    # After each step, overwrite word positions with noisy true latents → only transitions are free.
    scheduler = MicTDDPMScheduler(num_timesteps=cfg.get('num_timesteps', 1000)).to(device)
    masked_np  = np.array(masked_np, dtype=np.float32)
    masked_tensor = torch.tensor(masked_np, dtype=torch.float32, device=device).unsqueeze(0)

    T_total   = masked_tensor.shape[1]
    latent_dim = masked_tensor.shape[2]

    # Build known_mask (True = word frame) and known_x0 (true word latents)
    orig_latents = [np.array(z, dtype=np.float32) for z in orig_latents]
    known_mask = torch.zeros(T_total, dtype=torch.bool, device=device)
    known_x0   = torch.zeros(1, T_total, latent_dim, device=device)
    pos = 0
    for i, wlen in enumerate(word_lengths):
        known_mask[pos:pos + wlen] = True
        known_x0[0, pos:pos + wlen] = torch.tensor(
            orig_latents[i], dtype=torch.float32, device=device)
        pos += wlen
        if i < len(word_lengths) - 1:
            pos += transition_frames

    # RePaint denoising loop
    I_steps      = num_inference_steps
    N_steps      = scheduler.num_timesteps
    timestep_seq = [int(N_steps - 1 - (N_steps - 1) * i / (I_steps - 1)) for i in range(I_steps)]
    timestep_seq[-1] = 0

    print(f"\nRunning DDPM inpainting ({mode_str}, {num_inference_steps} steps)...")
    x_t = torch.randn(1, T_total, latent_dim, device=device)

    with torch.no_grad():
        for t_int in timestep_seq:
            x_t = scheduler.ddpm_step(model, x_t, t_int, masked_tensor, None)
            if t_int > 0:
                # Temporarily re-noise word positions at this timestep level
                ab    = scheduler.alphas_cumprod[t_int]
                noise = torch.randn_like(known_x0)
                noisy_known = ab.sqrt() * known_x0 + (1.0 - ab).sqrt() * noise
                x_t[:, known_mask, :] = noisy_known[:, known_mask, :]
        x_t = x_t.clamp(0.0, 1.0)

    generated = np.array(x_t[0].cpu().numpy(), dtype=np.float32)  # (T_total, latent_dim)


    # Extract transition latents and integrate velocity if needed
    transition_latents = []
    pos = 0
    for i, wlen in enumerate(word_lengths):
        pos += wlen
        if i < len(word_lengths) - 1:
            trans = generated[pos:pos + transition_frames]   # (tf, D)

            if use_velocity:
                # Integrate velocity anchored at last latent of word i
                anchor = np.array(orig_latents[i][-1], dtype=np.float32)   # (D,)
                trans  = np.array(trans, dtype=np.float32)
                trans_z = np.zeros_like(trans)
                trans_z[0] = anchor + trans[0]
                for t in range(1, len(trans)):
                    trans_z[t] = trans_z[t-1] + trans[t]
                trans = trans_z

            transition_latents.append(trans)
            pos += transition_frames

    print(f"\nGenerated {len(transition_latents)} transition segment(s)")

    # Decode transition latents → 1659D poses
    transition_poses = []
    with torch.no_grad():
        for trans_z in transition_latents:
            z_t = torch.tensor(trans_z, dtype=torch.float32, device=device).unsqueeze(0)
            pose_norm = vae.decode(z_t).squeeze(0).cpu().numpy()
            pose_raw  = denormalize(np.clip(pose_norm, 0.0, 1.0))
            transition_poses.append(pose_raw)

    # Post-process: snap hands
    transition_poses = [snap_hands_to_wrist(t) for t in transition_poses]
    print("  Snapped hand landmarks to pose wrist positions")

    # Assemble: original word frames + decoded transition frames
    # Apply snap to word frames too (not just transitions)
    final_parts = []
    for i, seg_norm in enumerate(orig_segs_norm):
        word_raw = denormalize(seg_norm)
        word_raw = snap_hands_to_wrist(word_raw)   # snap word frames too
        final_parts.append(word_raw)
        if i < len(orig_segs_norm) - 1:
            final_parts.append(transition_poses[i])

    final = np.concatenate([np.asarray(p, dtype=np.float32) for p in final_parts], axis=0)
    return final, transition_poses



def snap_hands_to_wrist(frames_raw):
    """Snap hand landmark 0 to pose wrist and shift all hand landmarks accordingly.
    
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


def main():
    parser = argparse.ArgumentParser(description='MicT V2 Inference (Latent Diffusion)')
    parser.add_argument('--ae_path',         required=True, help='Pose VAE checkpoint (Stage 1)')
    parser.add_argument('--model_path',      required=True, help='Diffusion checkpoint (Stage 2)')
    parser.add_argument('--word_dirs',       nargs='+', default=None)
    parser.add_argument('--sequences_dir',   default=None)
    parser.add_argument('--num_words',       type=int,   default=3)
    parser.add_argument('--transition_frames', type=int, default=10)
    parser.add_argument('--inference_steps', type=int,   default=50)
    parser.add_argument('--output_npy',      default='mict_v2_output.npy')
    parser.add_argument('--output_mp4',      default=None,
                        help='Nếu chỉ định, render skeleton video ra file .mp4 luôn')
    parser.add_argument('--fps',             type=int, default=30)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--hip_margin', type=float, default=0.05,
                        help='Wrist must be above hip by this margin to count as active (default: 0.05)')
    parser.add_argument('--no_adaptive',     action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.word_dirs:
        word_npys = []
        for d in args.word_dirs:
            npys = sorted(Path(d).glob('*.npy'))
            if not npys:
                raise FileNotFoundError(f"No .npy in {d}")
            word_npys.append(random.choice(npys))
        print(f"Words: {[str(p) for p in word_npys]}")
    elif args.sequences_dir:
        seq_dir = Path(args.sequences_dir)
        folders = [d for d in seq_dir.iterdir() if d.is_dir()]
        chosen  = random.sample(folders, min(args.num_words, len(folders)))
        word_npys = [random.choice(sorted(f.glob('*.npy'))) for f in chosen if list(f.glob('*.npy'))]
        print(f"Random words: {[p.parent.name for p in word_npys]}")
    else:
        raise ValueError("Provide --word_dirs or --sequences_dir")

    generated, transitions = run_inference(
        ae_path          = args.ae_path,
        model_path       = args.model_path,
        word_npys        = word_npys,
        transition_frames= args.transition_frames,
        num_inference_steps= args.inference_steps,
        adaptive         = not args.no_adaptive,
        hip_margin       = args.hip_margin,
    )

    np.save(args.output_npy, generated)
    print(f"\nSaved → {args.output_npy}  shape: {generated.shape}")

    # Render video nếu --output_mp4 được chỉ định
    if args.output_mp4:
        from src.core.render import render_skeleton_video
        # Flat (T, 1659) → (T, 553, 3)
        # Real layout: pose(0:99) + face(99:1533, 478kp) + lhand(1533:1596) + rhand(1596:1659)
        # Render expects: [pose(33), face(478), lhand(21), rhand(21)] = 553kp
        pose  = generated[:, 0:99].reshape(-1, 33, 3)
        face  = generated[:, 99:1533].reshape(-1, 478, 3)
        lhand = generated[:, 1533:1596].reshape(-1, 21, 3)
        rhand = generated[:, 1596:1659].reshape(-1, 21, 3)
        seq   = np.concatenate([pose, face, lhand, rhand], axis=1)  # (T, 553, 3)
        print(f"\nRendering video ({seq.shape[0]} frames @ {args.fps}fps)...")
        render_skeleton_video(seq, args.output_mp4, fps=args.fps)
        print(f"Video → {args.output_mp4}")


if __name__ == '__main__':
    main()
