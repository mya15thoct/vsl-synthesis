"""
Synthesis methods for VSL video generation.

Available methods:
    - linear     : Linear interpolation
    - spline     : Cubic spline interpolation
    - bezier     : Bezier curve interpolation
    - diffusion  : Diffusion model (requires trained model)
"""

from .interpolation import (
    linear_interpolation,
    cubic_spline_interpolation,
    bezier_interpolation
)
from .diffusion import VSLDiffusionGenerator

__all__ = [
    'linear_interpolation',
    'cubic_spline_interpolation',
    'bezier_interpolation',
    'VSLDiffusionGenerator',
]
