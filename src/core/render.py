"""
Video rendering module for skeleton sequences.

This module provides functions to render skeleton sequences to video files
using MediaPipe drawing utilities.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Dict, Tuple


# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles


def render_skeleton_video(
    skeleton_sequence: np.ndarray,
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720),
    draw_connections: bool = True,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> None:
    """
    Render skeleton sequence to video file.
    
    Args:
        skeleton_sequence: Skeleton data (num_frames, 543, 3)
        output_path: Path to save output video
        fps: Frames per second
        resolution: Video resolution (width, height)
        draw_connections: Whether to draw connections between keypoints
        background_color: Background color (B, G, R)
        
    Example:
        >>> skeleton = np.load('hello.npy')
        >>> render_skeleton_video(skeleton, 'output.mp4', fps=30)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    
    num_frames = skeleton_sequence.shape[0]
    
    for frame_idx in range(num_frames):
        # Create blank frame
        frame = np.full((resolution[1], resolution[0], 3), background_color, dtype=np.uint8)
        
        # Get skeleton for this frame
        skeleton = skeleton_sequence[frame_idx]  # (543, 3)
        
        # Draw skeleton on frame
        frame = draw_skeleton_on_frame(frame, skeleton, draw_connections)
        
        # Write frame
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")


def draw_skeleton_on_frame(
    frame: np.ndarray,
    skeleton: np.ndarray,
    draw_connections: bool = True
) -> np.ndarray:
    """
    Draw skeleton keypoints and connections on a frame.
    
    Args:
        frame: Image frame (H, W, 3)
        skeleton: Skeleton keypoints (543, 3) - normalized coordinates
        draw_connections: Whether to draw connections
        
    Returns:
        Frame with skeleton drawn
    """
    h, w = frame.shape[:2]
    
    # MediaPipe landmark indices
    # 543 = 33 (pose) + 468 (face) + 21 (left hand) + 21 (right hand)
    pose_landmarks = skeleton[:33]
    face_landmarks = skeleton[33:501]  # 468 face landmarks
    left_hand_landmarks = skeleton[501:522]  # 21 left hand
    right_hand_landmarks = skeleton[522:543]  # 21 right hand
    
    # Draw pose
    if draw_connections:
        _draw_landmarks_with_connections(
            frame, pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            (0, 255, 0)  # Green for pose
        )
    else:
        _draw_landmarks(frame, pose_landmarks, (0, 255, 0))
    
    # Draw hands
    if draw_connections:
        _draw_landmarks_with_connections(
            frame, left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            (255, 0, 0)  # Blue for left hand
        )
        _draw_landmarks_with_connections(
            frame, right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            (0, 0, 255)  # Red for right hand
        )
    else:
        _draw_landmarks(frame, left_hand_landmarks, (255, 0, 0))
        _draw_landmarks(frame, right_hand_landmarks, (0, 0, 255))
    
    # Draw face (just landmarks, no connections for clarity)
    _draw_landmarks(frame, face_landmarks, (128, 128, 128), radius=1)
    
    return frame


def _draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: Tuple[int, int, int],
    radius: int = 3
) -> None:
    """Draw landmarks as circles."""
    h, w = frame.shape[:2]
    
    for landmark in landmarks:
        x, y = int(landmark[0] * w), int(landmark[1] * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), radius, color, -1)


def _draw_landmarks_with_connections(
    frame: np.ndarray,
    landmarks: np.ndarray,
    connections,
    color: Tuple[int, int, int]
) -> None:
    """Draw landmarks with connections."""
    h, w = frame.shape[:2]
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_point = (int(start[0] * w), int(start[1] * h))
            end_point = (int(end[0] * w), int(end[1] * h))
            
            cv2.line(frame, start_point, end_point, color, 2)
    
    # Draw landmarks
    _draw_landmarks(frame, landmarks, color)


def create_comparison_video(
    sequences: Dict[str, np.ndarray],
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720)
) -> None:
    """
    Create side-by-side comparison video of multiple sequences.
    
    Args:
        sequences: Dictionary mapping method names to skeleton sequences
                  e.g., {'Linear': seq1, 'Spline': seq2, 'Diffusion': seq3}
        output_path: Path to save comparison video
        fps: Frames per second
        resolution: Resolution for each sub-video
        
    Example:
        >>> sequences = {
        ...     'Linear': linear_result,
        ...     'Spline': spline_result,
        ...     'Diffusion': diffusion_result
        ... }
        >>> create_comparison_video(sequences, 'comparison.mp4')
    """
    num_methods = len(sequences)
    
    if num_methods == 0:
        raise ValueError("sequences cannot be empty")
    
    # Calculate grid layout
    if num_methods <= 2:
        grid_rows, grid_cols = 1, num_methods
    elif num_methods <= 4:
        grid_rows, grid_cols = 2, 2
    else:
        grid_rows = int(np.ceil(np.sqrt(num_methods)))
        grid_cols = int(np.ceil(num_methods / grid_rows))
    
    # Calculate output resolution
    sub_w, sub_h = resolution
    output_w = sub_w * grid_cols
    output_h = sub_h * grid_rows
    
    # Get max frames
    max_frames = max(seq.shape[0] for seq in sequences.values())
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))
    
    method_names = list(sequences.keys())
    
    for frame_idx in range(max_frames):
        # Create blank output frame
        output_frame = np.full((output_h, output_w, 3), 255, dtype=np.uint8)
        
        for idx, method_name in enumerate(method_names):
            sequence = sequences[method_name]
            
            # Handle sequences of different lengths
            if frame_idx >= sequence.shape[0]:
                skeleton = sequence[-1]  # Use last frame
            else:
                skeleton = sequence[frame_idx]
            
            # Create sub-frame
            sub_frame = np.full((sub_h, sub_w, 3), 255, dtype=np.uint8)
            sub_frame = draw_skeleton_on_frame(sub_frame, skeleton)
            
            # Add method name label
            cv2.putText(sub_frame, method_name, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Calculate position in grid
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Place sub-frame in output
            y_start = row * sub_h
            y_end = y_start + sub_h
            x_start = col * sub_w
            x_end = x_start + sub_w
            
            output_frame[y_start:y_end, x_start:x_end] = sub_frame
        
        out.write(output_frame)
    
    out.release()
    print(f"Comparison video saved to: {output_path}")


if __name__ == "__main__":
    print("Render module loaded successfully!")
    print("\nAvailable functions:")
    print("  - render_skeleton_video: Render single sequence to video")
    print("  - create_comparison_video: Create side-by-side comparison")
