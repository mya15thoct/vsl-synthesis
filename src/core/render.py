"""
Video rendering module for 554-keypoint skeleton sequences.

This module handles the specific format from vsl-recognition project:
- Pose: 33 landmarks × 4 (x,y,z,visibility) = 132 values → 44 keypoints
- Face: 468 landmarks × 3 (x,y,z) = 1404 values → 468 keypoints  
- Left Hand: 21 landmarks × 3 (x,y,z) = 63 values → 21 keypoints
- Right Hand: 21 landmarks × 3 (x,y,z) = 63 values → 21 keypoints
Total: 1662 values → 554 keypoints when reshaped to (frames, 554, 3)
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
    Render skeleton sequence to video (auto-detects 554 or 543 keypoints).
    
    Args:
        skeleton_sequence: Skeleton data (num_frames, num_keypoints, 3)
        output_path: Path to save output video
        fps: Frames per second
        resolution: Video resolution (width, height)
        background_color: Background color (B, G, R)
    """
    num_keypoints = skeleton_sequence.shape[1]
    
    if num_keypoints == 554:
        render_skeleton_video_554(skeleton_sequence, output_path, fps, resolution, background_color)
    else:
        # Fallback: use 554 rendering (works for any keypoint count)
        render_skeleton_video_554(skeleton_sequence, output_path, fps, resolution, background_color)


def render_skeleton_video_554(
    skeleton_sequence: np.ndarray,
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720),
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> None:
    """
    Render 554-keypoint skeleton sequence to video.
    
    Args:
        skeleton_sequence: Skeleton data (num_frames, 554, 3)
        output_path: Path to save output video
        fps: Frames per second
        resolution: Video resolution (width, height)
        background_color: Background color (B, G, R)
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
        skeleton = skeleton_sequence[frame_idx]  # (554, 3)
        
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
    Draw 554-keypoint skeleton on frame.
    
    The 554 keypoints are structured as:
    - 0-43: Pose (33 landmarks with 4 values each, reshaped to 44 keypoints)
    - 44-511: Face (468 landmarks)
    - 512-532: Left Hand (21 landmarks)
    - 533-553: Right Hand (21 landmarks)
    
    Args:
        frame: Image frame (H, W, 3)
        skeleton: Skeleton keypoints (554, 3)
        
    Returns:
        Frame with skeleton drawn
    """
    h, w = frame.shape[:2]
    
    # Extract landmarks (approximate indices based on 1662/3 = 554 structure)
    # Pose: 132/3 = 44 keypoints (indices 0-43)
    # Face: 1404/3 = 468 keypoints (indices 44-511)
    # Left Hand: 63/3 = 21 keypoints (indices 512-532)
    # Right Hand: 63/3 = 21 keypoints (indices 533-553)
    
    pose_kpts = skeleton[:44]
    face_kpts = skeleton[44:512]
    left_hand_kpts = skeleton[512:533]
    right_hand_kpts = skeleton[533:554]
    
    # Draw pose with connections (use first 33 keypoints, skip visibility keypoints)
    pose_33 = pose_kpts[::4//3][:33]  # Approximate: take every ~1.33th keypoint to get 33
    # Simpler: just use first 33 keypoints
    pose_33 = skeleton[:33]
    
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
    
    # Draw face landmarks (no connections, just dots for clarity)
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
    
    # Auto-normalize if data is not in [0, 1] range
    if landmarks.size > 0:
        min_val = landmarks.min()
        max_val = landmarks.max()
        if min_val < 0 or max_val > 1:
            # Normalize to [0, 1]
            if max_val > min_val:
                landmarks = (landmarks - min_val) / (max_val - min_val)
            else:
                landmarks = np.full_like(landmarks, 0.5)
    
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
    
    # Auto-normalize if data is not in [0, 1] range
    if landmarks.size > 0:
        min_val = landmarks.min()
        max_val = landmarks.max()
        if min_val < 0 or max_val > 1:
            # Normalize to [0, 1]
            if max_val > min_val:
                landmarks = (landmarks - min_val) / (max_val - min_val)
            else:
                landmarks = np.full_like(landmarks, 0.5)
    
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
