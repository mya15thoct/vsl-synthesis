"""
VSL Synthesis - Vietnamese Sign Language Video Synthesis

Main package for synthesizing continuous sign language videos
from individual word videos.
"""

from .pipeline import synthesize_sentence, batch_synthesize
from .methods import (
    linear_interpolation,
    cubic_spline_interpolation,
    bezier_interpolation
)
from .core import (
    load_skeleton_sequence,
    concatenate_sequences,
    get_transition_boundaries,
    render_skeleton_video,
    create_comparison_video,
    calculate_jerk,
    evaluate_transition,
    compare_methods
)

__all__ = [
    # Pipeline
    'synthesize_sentence',
    'batch_synthesize',
    
    # Methods
    'linear_interpolation',
    'cubic_spline_interpolation',
    'bezier_interpolation',
    
    # Core - Concatenation
    'load_skeleton_sequence',
    'concatenate_sequences',
    'get_transition_boundaries',
    
    # Core - Rendering
    'render_skeleton_video',
    'create_comparison_video',
    
    # Core - Evaluation
    'calculate_jerk',
    'evaluate_transition',
    'compare_methods',
]
