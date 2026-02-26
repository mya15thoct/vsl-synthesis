"""
Video rendering module for 553-keypoint skeleton sequences.

This module handles the specific format from vsl-recognition project:
- Pose: 33 landmarks × 4 (x,y,z,visibility) = 132 values → 44 keypoints
- Face: 468 landmarks × 3 (x,y,z) = 1404 values → 468 keypoints  
- Left Hand: 21 landmarks × 3 (x,y,z) = 63 values → 21 keypoints
- Right Hand: 21 landmarks × 3 (x,y,z) = 63 values → 21 keypoints
Total: 1659 values → 553 keypoints when reshaped to (frames, 553, 3)
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Tuple

# MediaPipe drawing utilities
mp_holistic = mp.solutions.holistic


def render_skeleton_video(
    skeleton_sequence: np.ndarray,
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720),
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> None:
    """
    Render skeleton sequence to video (auto-detects 553 or 543 keypoints).
    
    Args:
        skeleton_sequence: Skeleton data (num_frames, num_keypoints, 3)
        output_path: Path to save output video
        fps: Frames per second
        resolution: Video resolution (width, height)
        background_color: Background color (B, G, R)
    """
    num_keypoints = skeleton_sequence.shape[1]
    
    if num_keypoints == 553:
        render_skeleton_video_554(skeleton_sequence, output_path, fps, resolution, background_color)
    else:
        # Fallback: use 553 rendering (works for any keypoint count)
        render_skeleton_video_554(skeleton_sequence, output_path, fps, resolution, background_color)


def render_skeleton_video_554(
    skeleton_sequence: np.ndarray,
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720),
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> None:
    """
    Render 553-keypoint skeleton sequence to video.
    
    Args:
        skeleton_sequence: Skeleton data (num_frames, 553, 3)
        output_path: Path to save output video
        fps: Frames per second
        resolution: Video resolution (width, height)
        background_color: Background color (B, G, R)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-normalize using only NON-ZERO frames (actual word frames).
    # Transition frames from an undertrained model may have extreme values
    # that collapse the entire render to a single dot if included in normalization.
    if skeleton_sequence.size > 0:
        xy_coords = skeleton_sequence[:, :, :2]  # (frames, 553, 2)

        # Identify non-zero frames (frames with actual skeleton data)
        frame_sums = np.abs(skeleton_sequence).sum(axis=(1, 2))  # (frames,)
        nonzero_mask = frame_sums > 1e-6

        if nonzero_mask.any():
            valid_xy = xy_coords[nonzero_mask]  # only real frames
            min_val = valid_xy.min()
            max_val = valid_xy.max()
        else:
            min_val = xy_coords.min()
            max_val = xy_coords.max()

        print(f"  Coordinate range (non-zero frames): [{min_val:.4f}, {max_val:.4f}]")

        if max_val > min_val:
            skeleton_sequence[:, :, :2] = (xy_coords - min_val) / (max_val - min_val)
            # Clamp to [0, 1] to handle extreme transition values
            skeleton_sequence[:, :, :2] = np.clip(skeleton_sequence[:, :, :2], 0.0, 1.0)
            print(f"  Normalized to: [0.0, 1.0]")
        else:
            skeleton_sequence[:, :, :2] = 0.5
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    
    num_frames = skeleton_sequence.shape[0]
    
    for frame_idx in range(num_frames):
        # Create blank frame
        frame = np.full((resolution[1], resolution[0], 3), background_color, dtype=np.uint8)
        
        # Get skeleton for this frame (already normalized)
        skeleton = skeleton_sequence[frame_idx]  # (553, 3)
        
        # Draw skeleton on frame
        frame = draw_skeleton_554(frame, skeleton)
        
        # Write frame
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")


def draw_skeleton_554(
    frame: np.ndarray,
    skeleton: np.ndarray
) -> np.ndarray:
    """
    Draw 553-keypoint skeleton on frame.
    
    Args:
        frame: Image frame (H, W, 3)
        skeleton: Skeleton keypoints (553, 3) - 3D landmarks (x, y, z)
                  We only use x, y for 2D rendering
        
    Returns:
        Frame with skeleton drawn
    """
    h, w = frame.shape[:2]
    
    # Extract ONLY x, y coordinates (ignore z)
    # MediaPipe 3D landmarks have z in range [-1, 1] which causes distortion
    skeleton_2d = skeleton[:, :2]  # (553, 2)
    
    # Extract landmarks
    pose_kpts = skeleton_2d[:33]
    face_kpts = skeleton_2d[33:511]
    left_hand_kpts = skeleton_2d[511:532]
    right_hand_kpts = skeleton_2d[532:553]
    
    # Draw pose with connections
    pose_33 = skeleton_2d[:33]
    
    _draw_landmarks_with_connections(
        frame, pose_33,
        mp_holistic.POSE_CONNECTIONS,
        (0, 255, 0),  # Green for pose
        thickness=2
    )
    
    # Draw hands with connections
    _draw_landmarks_with_connections(
        frame, left_hand_kpts,
        mp_holistic.HAND_CONNECTIONS,
        (255, 0, 0),  # Blue for left hand
        thickness=2
    )
    
    _draw_landmarks_with_connections(
        frame, right_hand_kpts,
        mp_holistic.HAND_CONNECTIONS,
        (0, 0, 255),  # Red for right hand
        thickness=2
    )
    
    # Draw face landmarks
    _draw_landmarks(frame, face_kpts, (128, 128, 128), radius=1)
    
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
    color: Tuple[int, int, int],
    thickness: int = 2
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
            
            # Only draw if both points are visible (not all zeros)
            if not (np.allclose(start, 0) or np.allclose(end, 0)):
                cv2.line(frame, start_point, end_point, color, thickness)
    
    # Draw landmarks
    _draw_landmarks(frame, landmarks, color, radius=4)
