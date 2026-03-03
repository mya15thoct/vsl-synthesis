#!/usr/bin/env python3
"""
MicT V2 — Diffusion Model trên Latent Space
Stage 2 của MicT V2 (Latent Diffusion)

Giống model_mict.py nhưng:
    input_dim = latent_dim (128) thay vì 1659
    Encoder/Decoder xử lý latent sequences thay vì raw pose

Architecture giống hệt MicT paper (arXiv 2508.04049).
"""

# Reuse hoàn toàn từ mict/ — chỉ thay input_dim khi khởi tạo
from src.methods.mict.model_mict import (
    MicTDiffusionModel,
    MicTDDPMScheduler,
    SinPosEncoding,
    ObservationEncoder,
    DenoiserBlock,
)

__all__ = [
    'MicTDiffusionModel',
    'MicTDDPMScheduler',
]
