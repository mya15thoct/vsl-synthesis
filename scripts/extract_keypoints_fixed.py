#!/usr/bin/env python3
"""
FIXED: Extract MediaPipe Holistic keypoints with CORRECT format (1662 features)

Format: 33×4 pose (with visibility) + 468×3 face + 21×3 left_hand + 21×3 right_hand = 1662

Usage:
    python scripts/extract_keypoints_fixed.py --input_dir /mnt/ngan/vsl_data --output_dir /mnt/ngan/vsl_data/sequences
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_keypoints_from_video(video_path: Path, holistic) -> np.ndarray:
    """
    Extract MediaPipe Holistic keypoints in CORRECT format.
    
    Returns:
        Keypoints array of shape (frames, 554, 3)
        - 554 keypoints total
        - Format: 44 pose (33×4 with visibility) + 468 face + 21 left + 21 right
        - Flattened to (frames, 1662) when saved
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
        
        # Extract keypoints in CORRECT order
        keypoints = []
        
        # 1. Pose landmarks (33 keypoints × 4: x, y, z, visibility)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility  # ✅ INCLUDE VISIBILITY
                ])
        else:
            keypoints.extend([0.0] * (33 * 4))  # 132 features
        
        # 2. Face landmarks (468 keypoints × 3: x, y, z)
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * (468 * 3))  # 1404 features
        
        # 3. Left hand landmarks (21 keypoints × 3: x, y, z)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * (21 * 3))  # 63 features
        
        # 4. Right hand landmarks (21 keypoints × 3: x, y, z)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            keypoints.extend([0.0] * (21 * 3))  # 63 features
        
        # Total: 132 + 1404 + 63 + 63 = 1662 features ✅
        frames_keypoints.append(keypoints)
    
    cap.release()
    
    # Convert to numpy array
    keypoints_array = np.array(frames_keypoints, dtype=np.float32)
    
    # Verify size
    expected_features = 1662
    actual_features = len(frames_keypoints[0]) if frames_keypoints else 0
    
    if actual_features != expected_features:
        raise ValueError(
            f"Feature count mismatch!\n"
            f"  Expected: {expected_features} features (33×4 + 468×3 + 21×3 + 21×3)\n"
            f"  Actual: {actual_features} features\n"
            f"  Breakdown:\n"
            f"    - Pose: 33 × 4 = 132 (with visibility)\n"
            f"    - Face: 468 × 3 = 1404\n"
            f"    - Left hand: 21 × 3 = 63\n"
            f"    - Right hand: 21 × 3 = 63"
        )
    
    # Reshape to (frames, 554, 3) - treating visibility as 3rd coordinate for pose
    # Actually keep as (frames, 1662) for now
    # Will be reshaped to (frames, 554, 3) when loaded
    
    return keypoints_array


def process_videos(input_dir: Path, output_dir: Path, video_extensions: list = None):
    """
    Process all videos and save keypoints in CORRECT format.
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI']
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=True,  # ✅ This gives 468 face landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print(f"\n🎥 Scanning for videos in: {input_dir}")
    print(f"✅ Output format: 1662 features (33×4 pose + 468×3 face + 21×3 hands)")
    
    # Find all word folders
    word_folders = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(word_folders)} word folders")
    
    total_videos = 0
    total_processed = 0
    total_failed = 0
    
    for word_folder in tqdm(word_folders, desc="Processing words"):
        # Create output folder
        output_word_dir = output_dir / word_folder.name
        output_word_dir.mkdir(parents=True, exist_ok=True)
        
        # Find videos
        video_files = []
        for ext in video_extensions:
            video_files.extend(word_folder.glob(f"*{ext}"))
        
        total_videos += len(video_files)
        
        for video_file in video_files:
            try:
                # Extract keypoints
                keypoints = extract_keypoints_from_video(video_file, holistic)
                
                # Verify shape
                if keypoints.shape[1] != 1662:
                    raise ValueError(f"Wrong shape: {keypoints.shape}, expected (frames, 1662)")
                
                # Save as .npy
                output_file = output_word_dir / f"{video_file.stem}.npy"
                np.save(output_file, keypoints)
                
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
    print(f"\n📊 Output format: (frames, 1662)")
    print(f"  - Pose: 132 features (33 landmarks × 4 with visibility)")
    print(f"  - Face: 1404 features (468 landmarks × 3)")
    print(f"  - Left hand: 63 features (21 landmarks × 3)")
    print(f"  - Right hand: 63 features (21 landmarks × 3)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract MediaPipe keypoints in CORRECT format (1662 features)'
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
        help='Video file extensions'
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
