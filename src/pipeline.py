"""
Main synthesis pipeline for VSL video generation.

This module orchestrates the entire synthesis process:
1. Load skeleton sequences
2. Generate transitions
3. Concatenate sequences
4. Render to video
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict

from .core.concatenation import (
    load_skeleton_sequence,
    concatenate_sequences,
    get_transition_boundaries
)
from .methods.interpolation import (
    linear_interpolation,
    cubic_spline_interpolation,
    bezier_interpolation
)
from .core.render import render_skeleton_video


def synthesize_sentence(
    word_videos: List[Union[str, Path]],
    method: str = 'spline',
    output_path: str = 'output.mp4',
    transition_frames: int = 10,
    fps: int = 30,
    model_path: Optional[str] = None
) -> str:
    """
    Main synthesis pipeline: Generate continuous sentence video from word videos.
    
    Args:
        word_videos: List of paths to word skeleton files (.npy)
        method: Interpolation method - 'linear', 'spline', 'bezier', or 'diffusion'
        output_path: Where to save output video
        transition_frames: Number of frames for each transition
        fps: Frames per second for output video
        model_path: Path to diffusion model (required if method='diffusion')
        
    Returns:
        Path to generated video
        
    Example:
        >>> # Basic usage with spline interpolation
        >>> output = synthesize_sentence(
        ...     word_videos=['data/words/hello.npy', 'data/words/my.npy'],
        ...     method='spline',
        ...     output_path='outputs/baseline/sentence.mp4'
        ... )
        
        >>> # Using diffusion model (Phase 3)
        >>> output = synthesize_sentence(
        ...     word_videos=['data/words/hello.npy', 'data/words/my.npy'],
        ...     method='diffusion',
        ...     model_path='models/mdm/finetuned.pt',
        ...     output_path='outputs/diffusion/sentence.mp4'
        ... )
    """
    print(f" Starting synthesis pipeline...")
    print(f"   Method: {method}")
    print(f"   Words: {len(word_videos)}")
    print(f"   Transition frames: {transition_frames}")
    
    # Validate inputs
    if not word_videos:
        raise ValueError("word_videos cannot be empty")
    
    if method not in ['linear', 'spline', 'bezier', 'diffusion']:
        raise ValueError(f"Invalid method: {method}. Choose from: linear, spline, bezier, diffusion")
    
    if method == 'diffusion' and model_path is None:
        raise ValueError("model_path is required when method='diffusion'")
    
    # Step 1: Concatenate sequences with gaps for transitions
    print(f"\n Step 1: Concatenating {len(word_videos)} sequences...")
    concatenated = concatenate_sequences(word_videos, transition_frames)
    print(f"   Total frames: {concatenated.shape[0]}")
    
    # Step 2: Get transition boundaries
    print(f"\n Step 2: Identifying transition boundaries...")
    boundaries = get_transition_boundaries(word_videos, transition_frames)
    print(f"   Transitions to generate: {len(boundaries)}")
    
    # Step 3: Generate transitions
    print(f"\n Step 3: Generating transitions with {method} method...")
    
    if method == 'diffusion':
        # Phase 3: Use diffusion model
        # Try VSL-native diffusion first, fallback to MDM if not available
        try:
            from .methods.vsl_diffusion_adapter import VSLDiffusionGenerator
            vsl_diffusion = VSLDiffusionGenerator(model_path=model_path)
            vsl_diffusion.load_model()
            print("   Using VSL-native diffusion model")
        except Exception as e:
            raise RuntimeError(f"VSL diffusion model not available: {e}")
        
        for i, (start_idx, end_idx, start_pose_idx, end_pose_idx) in enumerate(boundaries):
            start_pose = concatenated[start_pose_idx]
            end_pose = concatenated[end_pose_idx]
            
            print(f"   Transition {i+1}/{len(boundaries)}: frames {start_idx}-{end_idx}")
            transition = vsl_diffusion.generate_transition(start_pose, end_pose, transition_frames)
            concatenated[start_idx:end_idx] = transition

    
    else:
        # Phase 2: Use interpolation methods
        interpolation_func = {
            'linear': linear_interpolation,
            'spline': cubic_spline_interpolation,
            'bezier': bezier_interpolation
        }[method]
        
        for i, (start_idx, end_idx, start_pose_idx, end_pose_idx) in enumerate(boundaries):
            start_pose = concatenated[start_pose_idx]
            end_pose = concatenated[end_pose_idx]
            
            print(f"   Transition {i+1}/{len(boundaries)}: frames {start_idx}-{end_idx}")
            transition = interpolation_func(start_pose, end_pose, transition_frames)
            concatenated[start_idx:end_idx] = transition
    
    # Step 4: Render to video
    print(f"\n Step 4: Rendering video...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    render_skeleton_video(
        concatenated,
        str(output_path),
        fps=fps
    )
    
    print(f"\n Synthesis complete!")
    print(f"   Output: {output_path}")
    print(f"   Duration: {concatenated.shape[0] / fps:.2f} seconds")
    
    return str(output_path)


def batch_synthesize(
    word_sequences: List[List[Union[str, Path]]],
    methods: List[str] = ['linear', 'spline', 'bezier'],
    output_dir: str = 'outputs',
    **kwargs
) -> Dict[str, List[str]]:
    """
    Batch synthesis for multiple sentences and methods.
    
    Args:
        word_sequences: List of word sequences, each is a list of .npy files
        methods: List of methods to try
        output_dir: Base output directory
        **kwargs: Additional arguments for synthesize_sentence
        
    Returns:
        Dictionary mapping method names to lists of output paths
        
    Example:
        >>> sequences = [
        ...     ['hello.npy', 'my.npy', 'name.npy'],
        ...     ['how.npy', 'are.npy', 'you.npy']
        ... ]
        >>> results = batch_synthesize(sequences, methods=['linear', 'spline'])
    """
    results = {method: [] for method in methods}
    
    for seq_idx, word_sequence in enumerate(word_sequences):
        for method in methods:
            output_path = Path(output_dir) / method / f"sentence_{seq_idx:03d}.mp4"
            
            try:
                output = synthesize_sentence(
                    word_sequence,
                    method=method,
                    output_path=str(output_path),
                    **kwargs
                )
                results[method].append(output)
            except Exception as e:
                print(f"Error processing sequence {seq_idx} with {method}: {e}")
                results[method].append(None)
    
    return results


if __name__ == "__main__":
    print("Pipeline module loaded successfully!")
    print("\n VSL Synthesis Pipeline")
    print("\nUsage:")
    print("  from src.pipeline import synthesize_sentence")
    print("  output = synthesize_sentence(")
    print("      word_videos=['hello.npy', 'my.npy', 'name.npy'],")
    print("      method='spline',")
    print("      output_path='output.mp4'")
    print("  )")
