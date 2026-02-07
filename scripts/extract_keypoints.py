#!/usr/bin/env python3
"""
Extract MediaPipe Holistic keypoints from videos.

This script processes video files and extracts skeleton keypoints using MediaPipe Holistic.
Saves keypoints as .npy files in standard MediaPipe format.

Usage:
    python scripts/extract_keypoints.py --input_dir /mnt/ngan/vsl_data --output_dir /mnt/ngan/vsl_data/sequences
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_keypoints_from_video(video_path: Path, holistic) -> np.ndarray:
    """
    Extract MediaPipe Holistic keypoints from a video file.
    
    Args:
        video_path: Path to video file
        holistic: MediaPipe Holistic instance
        
    Returns:
        Keypoints array of shape (frames, 554, 3)
        - 554 keypoints: 33 pose + 468 face + 21 left_hand + 21 right_hand + 11 padding
        - 3 coordinates: x, y, z (normalized)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames_keypoints = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = holistic.process(frame_rgb)
        
        # Extract keypoints in standard MediaPipe order
        keypoints = []
        
        # 1. Pose landmarks (33 keypoints)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            # Fill with zeros if not detected
            keypoints.extend([0.0] * (33 * 3))
        
        # 2. Face landmarks (468 keypoints)
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * (468 * 3))
        
        # 3. Left hand landmarks (21 keypoints)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * (21 * 3))
        
        # 4. Right hand landmarks (21 keypoints)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * (21 * 3))
        
        # 5. Padding to reach 554 keypoints (11 * 3 = 33 values)
        # 33 + 468 + 21 + 21 = 543, need 11 more to reach 554
        keypoints.extend([0.0] * (11 * 3))
        
        frames_keypoints.append(keypoints)
    
    cap.release()
    
    # Convert to numpy array and reshape
    keypoints_array = np.array(frames_keypoints, dtype=np.float32)  # (frames, 1662)
    keypoints_array = keypoints_array.reshape(-1, 554, 3)  # (frames, 554, 3)
    
    return keypoints_array


def process_videos(input_dir: Path, output_dir: Path, video_extensions: list = None):
    """
    Process all videos in input directory and save keypoints.
    
    Args:
        input_dir: Directory containing word folders with videos
        output_dir: Directory to save extracted keypoints
        video_extensions: List of video file extensions to process
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI']
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print(f"\n🎥 Scanning for videos in: {input_dir}")
    
    # Find all word folders
    word_folders = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(word_folders)} word folders")
    
    total_videos = 0
    total_processed = 0
    total_failed = 0
    
    for word_folder in tqdm(word_folders, desc="Processing words"):
        # Create output folder for this word
        output_word_dir = output_dir / word_folder.name
        output_word_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(word_folder.glob(f"*{ext}"))
        
        total_videos += len(video_files)
        
        for video_file in video_files:
            try:
                # Extract keypoints
                keypoints = extract_keypoints_from_video(video_file, holistic)
                
                # Save as .npy file
                output_file = output_word_dir / f"{video_file.stem}.npy"
                
                # Save in 2D format (frames, 1662) for compatibility
                keypoints_flat = keypoints.reshape(keypoints.shape[0], -1)
                np.save(output_file, keypoints_flat)
                
                total_processed += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {video_file}: {e}")
                total_failed += 1
    
    holistic.close()
    
    print(f"\n✅ Extraction complete!")
    print(f"  Total videos found: {total_videos}")
    print(f"  Successfully processed: {total_processed}")
    print(f"  Failed: {total_failed}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract MediaPipe Holistic keypoints from videos'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing word folders with videos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save extracted keypoints'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI'],
        help='Video file extensions to process'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        return
    
    process_videos(input_dir, output_dir, args.extensions)


if __name__ == "__main__":
    main()
