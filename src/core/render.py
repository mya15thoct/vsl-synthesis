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
    Render skeleton sequence to video (auto-detects format).
    
    Args:
        skeleton_sequence: Skeleton data (num_frames, num_keypoints, 3) or (num_frames, 1662)
        output_path: Path to save output video
        fps: Frames per second
        resolution: Video resolution (width, height)
        background_color: Background color (B, G, R)
    """
    # Check if we need to handle raw 1662 format
    if skeleton_sequence.ndim == 2 and skeleton_sequence.shape[1] == 1662:
        # Raw 1662 format - render directly
        render_skeleton_video_1662(skeleton_sequence, output_path, fps, resolution, background_color)
    elif skeleton_sequence.ndim == 3 and skeleton_sequence.shape[1] == 554:
        # Already reshaped to 554 - convert back to 1662 for proper rendering
        # Reshape back: (frames, 554, 3) -> (frames, 1662)
        raw = skeleton_sequence.reshape(skeleton_sequence.shape[0], -1)
        render_skeleton_video_1662(raw, output_path, fps, resolution, background_color)
    elif skeleton_sequence.ndim == 3:
        # Other 3D format - try generic rendering
        render_skeleton_video_generic(skeleton_sequence, output_path, fps, resolution, background_color)
    else:
        raise ValueError(f"Unsupported skeleton format: {skeleton_sequence.shape}")


def render_skeleton_video_1662(
    skeleton_sequence: np.ndarray,
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720),
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> None:
    """
    Render skeleton from raw 1662 format (proper parsing of pose with visibility).
    
    1662 values breakdown:
    - Pose: 33 × 4 (x,y,z,visibility) = 132 values (indices 0-131)
    - Face: 468 × 3 (x,y,z) = 1404 values (indices 132-1535)
    - Left Hand: 21 × 3 (x,y,z) = 63 values (indices 1536-1598)
    - Right Hand: 21 × 3 (x,y,z) = 63 values (indices 1599-1661)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    
    h, w = resolution[1], resolution[0]
    num_frames = skeleton_sequence.shape[0]
    
    for frame_idx in range(num_frames):
        frame = np.full((h, w, 3), background_color, dtype=np.uint8)
        raw = skeleton_sequence[frame_idx]  # (1662,)
        
        # Parse pose: 33 landmarks × 4 values each
        pose_raw = raw[:132].reshape(33, 4)  # (33, 4) -> x, y, z, visibility
        pose_xyz = pose_raw[:, :3]  # (33, 3) -> just x, y, z
        
        # Parse face: 468 landmarks × 3 values each
        face_xyz = raw[132:1536].reshape(468, 3)
        
        # Parse hands: 21 landmarks × 3 values each
        left_hand_xyz = raw[1536:1599].reshape(21, 3)
        right_hand_xyz = raw[1599:1662].reshape(21, 3)
        
        # Draw pose with connections
        _draw_landmarks_with_connections(
            frame, pose_xyz,
            mp_holistic.POSE_CONNECTIONS,
            (0, 255, 0), thickness=2
        )
        
        # Draw hands
        _draw_landmarks_with_connections(
            frame, left_hand_xyz,
            mp_holistic.HAND_CONNECTIONS,
            (255, 0, 0), thickness=2  # Blue for left
        )
        _draw_landmarks_with_connections(
            frame, right_hand_xyz,
            mp_holistic.HAND_CONNECTIONS,
            (0, 0, 255), thickness=2  # Red for right
        )
        
        # Draw face (just dots)
        _draw_landmarks(frame, face_xyz, (128, 128, 128), radius=1)
        
        out.write(frame)
    
    out.release()
    print(f" Video saved to: {output_path}")


def render_skeleton_video_generic(
    skeleton_sequence: np.ndarray,
    output_path: str,
    fps: int = 30,
    resolution: Tuple[int, int] = (1280, 720),
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> None:
    """Generic renderer for unknown formats - draws points only."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    
    h, w = resolution[1], resolution[0]
    
    for frame_idx in range(skeleton_sequence.shape[0]):
        frame = np.full((h, w, 3), background_color, dtype=np.uint8)
        skeleton = skeleton_sequence[frame_idx]
        _draw_landmarks(frame, skeleton, (0, 128, 0), radius=2)
        out.write(frame)
    
    out.release()
    print(f" Video saved to: {output_path}")


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
    print(f" Video saved to: {output_path}")


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
    
    # Draw pose with connections
    # Note: For 554-keypoint format, pose data is 132 values (33 landmarks × 4: x,y,z,visibility)
    # When reshaped to (554, 3), pose occupies indices 0-43 (132/3=44 keypoints)
    # But the actual pose landmarks are interleaved with visibility values
    # We need to extract every 4th value pattern correctly
    
    # Extract actual pose landmarks (33 points) from the 44 keypoints
    # The pattern in 554 format: [x0,y0,z0], [v0,x1,y1], [z1,v1,x2], [y2,z2,v2], ...
    # This is messy, so we reconstruct from the original 132 values concept
    
    # Simpler approach: Check if coordinates are normalized (0-1 range)
    # If pose points have invalid values (>1 or <0), skip drawing body connections
    
    # For 554-keypoint format, try to use pose keypoints 0-32 with stride
    # Actually the 554 format maps pose as: first 44 keypoints contain pose data
    # But connections expect indices 0-32 for MediaPipe pose
    
    pose_landmarks = skeleton[:33]  # First 33 keypoints for pose
    
    # Only draw pose connections if landmarks look valid (in 0-1 range)
    valid_pose = np.all((pose_landmarks[:, :2] >= 0) & (pose_landmarks[:, :2] <= 1.5))
    
    if valid_pose:
        _draw_landmarks_with_connections(
            frame, pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            (0, 255, 0),  # Green for pose
            thickness=2
        )
    else:
        # Just draw pose points without connections
        _draw_landmarks(frame, pose_landmarks, (0, 255, 0), radius=3)
    
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
