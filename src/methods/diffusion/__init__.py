"""
Diffusion-based transition generation for VSL synthesis.

This package contains all code specific to the diffusion method:
- model.py      : VSLDiffusionModel architecture
- scheduler.py  : SimpleDDPMScheduler
- dataset.py    : VSLTransitionDataset for training
- losses.py     : MediaPipe perceptual loss
- adapter.py    : VSLDiffusionGenerator (inference)
- train.py      : Training loop
"""

from .adapter import VSLDiffusionGenerator

__all__ = ['VSLDiffusionGenerator']
