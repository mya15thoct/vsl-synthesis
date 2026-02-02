"""
Core processing modules for VSL synthesis.

This module contains core functionality for concatenation,
rendering, and evaluation.
"""

from .concatenation import (
    load_skeleton_sequence,
    concatenate_sequences,
    get_transition_boundaries
)
from .render import (
    render_skeleton_video,
)
from .evaluation import (
    calculate_jerk,
    evaluate_transition,
    compare_methods
)

__all__ = [
    # Concatenation
    'load_skeleton_sequence',
    'concatenate_sequences',
    'get_transition_boundaries',
    
    # Rendering
    'render_skeleton_video',
    
    # Evaluation
    'calculate_jerk',
    'evaluate_transition',
    'compare_methods',
]
