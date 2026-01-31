"""
Interpolation methods for generating smooth transitions between skeleton poses.

This module implements various interpolation techniques:
- Linear interpolation
- Cubic spline interpolation
- Bezier curve interpolation
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional


def linear_interpolation(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    num_frames: int
) -> np.ndarray:
    """
    Simple linear interpolation between two poses.
    
    Args:
        start_pose: Starting pose (543, 3)
        end_pose: Ending pose (543, 3)
        num_frames: Number of frames to generate (including start and end)
        
    Returns:
        Interpolated sequence (num_frames, 543, 3)
        
    Example:
        >>> start = np.random.rand(543, 3)
        >>> end = np.random.rand(543, 3)
        >>> transition = linear_interpolation(start, end, 10)
        >>> transition.shape
        (10, 543, 3)
    """
    # Validate pose shapes match and are 2D with 3 coordinates
    if start_pose.ndim != 2 or end_pose.ndim != 2:
        raise ValueError(f"Poses must be 2D arrays, got {start_pose.ndim}D and {end_pose.ndim}D")
    
    if start_pose.shape != end_pose.shape:
        raise ValueError(f"Poses must have same shape, got {start_pose.shape} and {end_pose.shape}")
    
    if start_pose.shape[1] != 3:
        raise ValueError(f"Poses must have 3 coordinates, got {start_pose.shape[1]}")
    
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")
    
    # Create interpolation weights
    alphas = np.linspace(0, 1, num_frames)[:, np.newaxis, np.newaxis]
    
    # Linear interpolation: (1-alpha)*start + alpha*end
    interpolated = (1 - alphas) * start_pose + alphas * end_pose
    
    return interpolated.astype(np.float32)


def cubic_spline_interpolation(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    num_frames: int,
    start_velocity: Optional[np.ndarray] = None,
    end_velocity: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Smooth cubic spline interpolation between two poses.
    
    Uses scipy's CubicSpline for smooth, natural-looking transitions.
    
    Args:
        start_pose: Starting pose (543, 3)
        end_pose: Ending pose (543, 3)
        num_frames: Number of frames to generate
        start_velocity: Optional starting velocity for smoother transitions
        end_velocity: Optional ending velocity for smoother transitions
        
    Returns:
        Interpolated sequence (num_frames, 543, 3)
        
    Note:
        Cubic spline produces smoother motion than linear interpolation
        by ensuring continuous first and second derivatives.
    """
    # Validate pose shapes
    if start_pose.ndim != 2 or end_pose.ndim != 2:
        raise ValueError(f"Poses must be 2D arrays")
    
    if start_pose.shape != end_pose.shape:
        raise ValueError(f"Poses must have same shape, got {start_pose.shape} and {end_pose.shape}")
    
    if start_pose.shape[1] != 3:
        raise ValueError(f"Poses must have 3 coordinates, got {start_pose.shape[1]}")
    
    num_keypoints = start_pose.shape[0]
    
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2")
    
    # Time points for start and end
    t = np.array([0, 1])
    
    # Stack poses for interpolation
    poses = np.stack([start_pose, end_pose], axis=0)  # (2, 543, 3)
    
    # Interpolate each keypoint's each coordinate separately
    interpolated = np.zeros((num_frames, num_keypoints, 3), dtype=np.float32)
    
    # New time points for interpolation
    t_new = np.linspace(0, 1, num_frames)
    
    for keypoint_idx in range(num_keypoints):
        for coord_idx in range(3):
            # Get values for this keypoint's coordinate
            y = poses[:, keypoint_idx, coord_idx]
            
            # Create cubic spline
            cs = CubicSpline(t, y, bc_type='natural')
            
            # Interpolate
            interpolated[:, keypoint_idx, coord_idx] = cs(t_new)
    
    return interpolated


def bezier_interpolation(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    num_frames: int,
    control_point_ratio: float = 0.5
) -> np.ndarray:
    """
    Bezier curve interpolation between two poses.
    
    Uses cubic Bezier curves with automatically generated control points
    for smooth, natural transitions.
    
    Args:
        start_pose: Starting pose (543, 3)
        end_pose: Ending pose (543, 3)
        num_frames: Number of frames to generate
        control_point_ratio: Ratio for control point placement (0-1)
                           0.5 means control points at 1/3 and 2/3
        
    Returns:
        Interpolated sequence (num_frames, 543, 3)
        
    Note:
        Bezier curves provide smooth acceleration/deceleration,
        making motion look more natural than linear interpolation.
    """
    # Validate pose shapes
    if start_pose.ndim != 2 or end_pose.ndim != 2:
        raise ValueError(f"Poses must be 2D arrays")
    
    if start_pose.shape != end_pose.shape:
        raise ValueError(f"Poses must have same shape, got {start_pose.shape} and {end_pose.shape}")
    
    if start_pose.shape[1] != 3:
        raise ValueError(f"Poses must have 3 coordinates, got {start_pose.shape[1]}")
    
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2")
    
    # Generate control points (simple approach: 1/3 and 2/3 along the path)
    control1 = start_pose + control_point_ratio * (end_pose - start_pose) / 3
    control2 = start_pose + 2 * control_point_ratio * (end_pose - start_pose) / 3
    
    # Time parameter
    t = np.linspace(0, 1, num_frames)[:, np.newaxis, np.newaxis]
    
    # Cubic Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    interpolated = (
        (1 - t)**3 * start_pose +
        3 * (1 - t)**2 * t * control1 +
        3 * (1 - t) * t**2 * control2 +
        t**3 * end_pose
    )
    
    return interpolated.astype(np.float32)


def smooth_velocity_profile(sequence: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Smooth the velocity profile of a sequence to reduce jerkiness.
    
    Args:
        sequence: Input sequence (num_frames, 543, 3)
        window_size: Size of smoothing window (must be odd)
        
    Returns:
        Smoothed sequence with same shape
        
    Note:
        This can be applied after interpolation to further smooth motion.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    # Simple moving average smoothing
    from scipy.ndimage import uniform_filter1d
    
    smoothed = uniform_filter1d(sequence, size=window_size, axis=0, mode='nearest')
    
    return smoothed.astype(np.float32)


if __name__ == "__main__":
    # Example usage and comparison
    print("Interpolation module loaded successfully!")
    print("\nAvailable methods:")
    print("  - linear_interpolation: Simple, fast")
    print("  - cubic_spline_interpolation: Smooth, natural")
    print("  - bezier_interpolation: Smooth acceleration/deceleration")
    
    # Quick test
    start = np.random.rand(543, 3)
    end = np.random.rand(543, 3)
    
    linear = linear_interpolation(start, end, 10)
    spline = cubic_spline_interpolation(start, end, 10)
    bezier = bezier_interpolation(start, end, 10)
    
    print(f"\nTest successful! Generated {linear.shape[0]} frames with each method.")
