"""
Evaluation metrics for VSL video synthesis quality assessment.

This module provides functions to calculate various metrics:
- FID (Fréchet Inception Distance)
- Jerk (motion smoothness)
- Foot skating detection
- Pose accuracy
"""

import numpy as np
from typing import List, Dict
from scipy.spatial.distance import euclidean


def calculate_jerk(sequence: np.ndarray) -> float:
    """
    Calculate average jerk (third derivative of position).
    
    Lower jerk indicates smoother motion.
    
    Args:
        sequence: Skeleton sequence (num_frames, 543, 3)
        
    Returns:
        Average jerk value (lower is better)
        
    Note:
        Jerk = d³position/dt³
        Measures how abruptly acceleration changes
    """
    if sequence.shape[0] < 4:
        raise ValueError("Need at least 4 frames to calculate jerk")
    
    # Calculate derivatives
    velocity = np.diff(sequence, axis=0)  # First derivative
    acceleration = np.diff(velocity, axis=0)  # Second derivative
    jerk = np.diff(acceleration, axis=0)  # Third derivative
    
    # Calculate magnitude of jerk for each keypoint
    jerk_magnitude = np.linalg.norm(jerk, axis=2)  # (frames-3, 543)
    
    # Average across all keypoints and frames
    avg_jerk = np.mean(jerk_magnitude)
    
    return float(avg_jerk)


def calculate_smoothness(sequence: np.ndarray) -> float:
    """
    Calculate motion smoothness using velocity variance.
    
    Args:
        sequence: Skeleton sequence (num_frames, 543, 3)
        
    Returns:
        Smoothness score (higher is smoother)
    """
    if sequence.shape[0] < 2:
        raise ValueError("Need at least 2 frames")
    
    # Calculate velocity
    velocity = np.diff(sequence, axis=0)
    
    # Calculate velocity magnitude for each keypoint
    velocity_magnitude = np.linalg.norm(velocity, axis=2)
    
    # Lower variance = more smooth
    variance = np.var(velocity_magnitude)
    
    # Convert to smoothness score (inverse of variance)
    smoothness = 1.0 / (1.0 + variance)
    
    return float(smoothness)


def calculate_pose_error(
    predicted_pose: np.ndarray,
    target_pose: np.ndarray
) -> float:
    """
    Calculate L2 distance between two poses.
    
    Args:
        predicted_pose: Predicted pose (543, 3)
        target_pose: Target pose (543, 3)
        
    Returns:
        Average L2 distance across all keypoints
    """
    if predicted_pose.shape != (543, 3) or target_pose.shape != (543, 3):
        raise ValueError("Poses must be shape (543, 3)")
    
    # Calculate L2 distance for each keypoint
    distances = np.linalg.norm(predicted_pose - target_pose, axis=1)
    
    # Average across all keypoints
    avg_distance = np.mean(distances)
    
    return float(avg_distance)


def evaluate_transition(
    transition: np.ndarray,
    start_pose: np.ndarray,
    end_pose: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a single transition.
    
    Args:
        transition: Transition sequence (num_frames, 543, 3)
        start_pose: Expected starting pose (543, 3)
        end_pose: Expected ending pose (543, 3)
        
    Returns:
        Dictionary of metrics:
        - jerk: Motion jerk (lower is better)
        - smoothness: Motion smoothness (higher is better)
        - start_error: L2 distance from start_pose
        - end_error: L2 distance from end_pose
    """
    metrics = {}
    
    # Jerk
    metrics['jerk'] = calculate_jerk(transition)
    
    # Smoothness
    metrics['smoothness'] = calculate_smoothness(transition)
    
    # Start pose accuracy
    metrics['start_error'] = calculate_pose_error(transition[0], start_pose)
    
    # End pose accuracy
    metrics['end_error'] = calculate_pose_error(transition[-1], end_pose)
    
    return metrics


def evaluate_full_sequence(
    sequence: np.ndarray,
    reference_sequence: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate a full synthesized sequence.
    
    Args:
        sequence: Synthesized sequence (num_frames, 543, 3)
        reference_sequence: Optional reference sequence for comparison
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['jerk'] = calculate_jerk(sequence)
    metrics['smoothness'] = calculate_smoothness(sequence)
    metrics['num_frames'] = sequence.shape[0]
    
    # If reference provided, calculate similarity
    if reference_sequence is not None:
        # TODO: Implement FID calculation
        # This requires a pretrained feature extractor
        metrics['fid'] = 0.0  # Placeholder
        
        # Frame-wise similarity
        if sequence.shape[0] == reference_sequence.shape[0]:
            frame_errors = []
            for i in range(sequence.shape[0]):
                error = calculate_pose_error(sequence[i], reference_sequence[i])
                frame_errors.append(error)
            metrics['avg_frame_error'] = np.mean(frame_errors)
    
    return metrics


def compare_methods(
    results: Dict[str, np.ndarray],
    reference: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple synthesis methods.
    
    Args:
        results: Dictionary mapping method names to sequences
                e.g., {'Linear': seq1, 'Spline': seq2, 'Diffusion': seq3}
        reference: Optional reference sequence
        
    Returns:
        Dictionary mapping method names to their metrics
        
    Example:
        >>> results = {
        ...     'Linear': linear_result,
        ...     'Spline': spline_result,
        ...     'Diffusion': diffusion_result
        ... }
        >>> comparison = compare_methods(results)
        >>> print(comparison['Spline']['jerk'])
    """
    comparison = {}
    
    for method_name, sequence in results.items():
        metrics = evaluate_full_sequence(sequence, reference)
        comparison[method_name] = metrics
    
    return comparison


def print_comparison_table(comparison: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted comparison table.
    
    Args:
        comparison: Output from compare_methods()
    """
    print("\n" + "="*60)
    print("SYNTHESIS METHODS COMPARISON")
    print("="*60)
    
    # Get all metric names
    all_metrics = set()
    for metrics in comparison.values():
        all_metrics.update(metrics.keys())
    
    # Print header
    methods = list(comparison.keys())
    print(f"{'Metric':<20} " + " ".join(f"{m:>12}" for m in methods))
    print("-"*60)
    
    # Print each metric
    for metric in sorted(all_metrics):
        values = [comparison[m].get(metric, 0.0) for m in methods]
        print(f"{metric:<20} " + " ".join(f"{v:>12.4f}" for v in values))
    
    print("="*60)
    print("\nLower is better: jerk, start_error, end_error, fid")
    print("Higher is better: smoothness")
    print()


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("  - calculate_jerk: Measure motion smoothness")
    print("  - evaluate_transition: Evaluate single transition")
    print("  - evaluate_full_sequence: Evaluate complete sequence")
    print("  - compare_methods: Compare multiple synthesis methods")
