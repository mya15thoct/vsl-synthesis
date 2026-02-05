"""
MediaPipe-based Perceptual Loss

Uses pretrained MediaPipe Holistic model to evaluate skeleton quality.
MediaPipe provides learned features for pose, hands, and face validation.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
from typing import Optional


class MediaPipePerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained MediaPipe Holistic model.
    
    MediaPipe evaluates skeleton quality based on learned human pose priors.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,  # Highest quality
            enable_segmentation=False,
            refine_face_landmarks=True
        )
        
        # Feature extractor (frozen)
        self.register_buffer('initialized', torch.tensor(True))
    
    def skeleton_to_image(self, skeleton_flat):
        """
        Convert skeleton to image for MediaPipe processing.
        
        Args:
            skeleton_flat: (1662,) flattened skeleton
            
        Returns:
            image: (H, W, 3) RGB image with skeleton drawn
        """
        # Create blank image
        img_size = 512
        image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Extract keypoints
        pose_raw = skeleton_flat[:132].reshape(33, 4)
        pose_xyz = pose_raw[:, :3]  # (33, 3)
        
        # Scale to image coordinates
        pose_xy = pose_xyz[:, :2]  # Take x, y only
        pose_xy = (pose_xy * img_size).astype(np.int32)
        
        # Clip to valid range
        pose_xy = np.clip(pose_xy, 0, img_size - 1)
        
        # Draw pose keypoints
        for i, (x, y) in enumerate(pose_xy):
            # Ensure int type for OpenCV
            x, y = int(x), int(y)
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        
        # Draw connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(pose_xy) and end_idx < len(pose_xy):
                x1, y1 = int(pose_xy[start_idx][0]), int(pose_xy[start_idx][1])
                x2, y2 = int(pose_xy[end_idx][0]), int(pose_xy[end_idx][1])
                if (0 <= x1 < img_size and 0 <= y1 < img_size and
                    0 <= x2 < img_size and 0 <= y2 < img_size):
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return image
    
    def extract_mediapipe_features(self, skeleton_flat):
        """
        Extract features using MediaPipe pretrained model.
        
        Args:
            skeleton_flat: (1662,) numpy array
            
        Returns:
            features: dict of MediaPipe landmarks or None if detection fails
        """
        # Convert to image
        image = self.skeleton_to_image(skeleton_flat)
        
        # Process with MediaPipe
        results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks is None:
            return None
        
        # Extract landmark features
        features = {
            'pose': np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]),
            'left_hand': np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else None,
            'right_hand': np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else None,
        }
        
        return features
    
    def compute_feature_distance(self, features1, features2):
        """
        Compute distance between MediaPipe features.
        
        Args:
            features1, features2: MediaPipe feature dicts
            
        Returns:
            distance: scalar
        """
        if features1 is None or features2 is None:
            return 1.0  # High penalty for invalid skeletons
        
        total_dist = 0.0
        count = 0
        
        # Pose distance
        if features1['pose'] is not None and features2['pose'] is not None:
            pose_dist = np.mean((features1['pose'] - features2['pose']) ** 2)
            total_dist += pose_dist
            count += 1
        
        # Hand distances
        for hand in ['left_hand', 'right_hand']:
            if features1[hand] is not None and features2[hand] is not None:
                hand_dist = np.mean((features1[hand] - features2[hand]) ** 2)
                total_dist += hand_dist
                count += 1
        
        return total_dist / max(count, 1)
    
    def forward(self, predicted_skeleton, target_skeleton=None):
        """
        Compute perceptual loss using MediaPipe.
        
        Args:
            predicted_skeleton: (batch, num_frames, 1662) predicted skeleton
            target_skeleton: (batch, num_frames, 1662) optional target skeleton
            
        Returns:
            loss: scalar tensor
        """
        batch_size, num_frames, _ = predicted_skeleton.shape
        
        # Convert to numpy
        pred_np = predicted_skeleton.detach().cpu().numpy()
        
        total_loss = 0.0
        valid_count = 0
        
        # Process each frame
        for b in range(min(batch_size, 2)):  # Limit to 2 samples for speed
            for f in range(0, num_frames, max(1, num_frames // 3)):  # Sample 3 frames
                skeleton = pred_np[b, f]
                
                # Extract MediaPipe features
                features = self.extract_mediapipe_features(skeleton)
                
                # Penalize if MediaPipe can't detect pose (invalid skeleton)
                if features is None:
                    total_loss += 0.01  # Invalid skeleton (scaled down from 1.0)
                else:
                    # Reward valid detection (MediaPipe recognizes it as human pose)
                    # Lower loss for valid skeletons
                    total_loss += 0.001  # Valid skeleton (scaled down from 0.1)
                
                valid_count += 1
        
        if valid_count == 0:
            return torch.tensor(0.01, device=predicted_skeleton.device)
        
        avg_loss = total_loss / valid_count
        return torch.tensor(avg_loss, device=predicted_skeleton.device)
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'holistic'):
            self.holistic.close()


# Global instance (singleton)
_mediapipe_loss = None

def get_mediapipe_perceptual_loss(device='cuda'):
    """Get or create MediaPipe perceptual loss instance."""
    global _mediapipe_loss
    if _mediapipe_loss is None:
        _mediapipe_loss = MediaPipePerceptualLoss(device=device)
    return _mediapipe_loss


def mediapipe_perceptual_loss(predicted_skeleton, target_skeleton=None):
    """
    Compute MediaPipe-based perceptual loss.
    
    Args:
        predicted_skeleton: (batch, num_frames, 1662)
        target_skeleton: (batch, num_frames, 1662) optional
        
    Returns:
        loss: scalar tensor
    """
    mp_loss = get_mediapipe_perceptual_loss(device=predicted_skeleton.device)
    return mp_loss(predicted_skeleton, target_skeleton)
