"""
Synthesis methods for VSL video generation.

This module contains different methods for generating transitions
between sign language word sequences.
"""

from .interpolation import (
    linear_interpolation,
    cubic_spline_interpolation,
    bezier_interpolation
)

__all__ = [
    'linear_interpolation',
    'cubic_spline_interpolation',
    'bezier_interpolation',
]
