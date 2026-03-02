"""
MicT V2 — Latent Diffusion version

Khác với mict/ (direct 1659D diffusion):
- Stage 1: Autoencoder 1659D → 128D latent
- Stage 2: Diffusion trên latent 128D → decode → pose

Files:
    autoencoder.py      : Pose VAE (Encoder + Decoder)
    train_autoencoder.py: Train Stage 1
    model_mict_v2.py    : MicT Diffusion model (latent space)
    train_mict_v2.py    : Train Stage 2
    infer_mict_v2.py    : Inference (encode → diffuse → decode)
    prepare_data.py     : Reuse prepare_data_mict.py từ mict/
"""
