"""
Physical Constraints for VSL Skeleton

This module implements constraint losses to ensure generated skeletons
are physically realistic (correct bone lengths, smooth motion, valid poses).
"""

import torch
import torch.nn.functional as F


# VSL Skeleton structure (1662 features)
# Pose: 0-131 (33 keypoints × 4: x,y,z,visibility)
# Face: 132-1535 (468 keypoints × 3)
# Left Hand: 1536-1598 (21 keypoints × 3)
# Right Hand: 1599-1661 (21 keypoints × 3)

# MediaPipe Pose connections (indices for 33 pose landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Face
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
]

# Hand connections (21 landmarks each)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]


def extract_pose_keypoints(skeleton_flat):
    """
    Extract pose keypoints from flattened 1662 format.
    
    Args:
        skeleton_flat: (batch, num_frames, 1662) or (batch, 1662)
        
    Returns:
        pose_xyz: (batch, num_frames, 33, 3) or (batch, 33, 3)
    """
    if skeleton_flat.ndim == 3:
        batch, frames, _ = skeleton_flat.shape
        # Extract pose (first 132 values)
        pose_raw = skeleton_flat[:, :, :132].reshape(batch, frames, 33, 4)
        pose_xyz = pose_raw[:, :, :, :3]  # Take only x,y,z
    else:
        batch, _ = skeleton_flat.shape
        pose_raw = skeleton_flat[:, :132].reshape(batch, 33, 4)
        pose_xyz = pose_raw[:, :, :3]
    
    return pose_xyz


def extract_hand_keypoints(skeleton_flat, hand='left'):
    """
    Extract hand keypoints from flattened 1662 format.
    
    Args:
        skeleton_flat: (batch, num_frames, 1662) or (batch, 1662)
        hand: 'left' or 'right'
        
    Returns:
        hand_xyz: (batch, num_frames, 21, 3) or (batch, 21, 3)
    """
    if hand == 'left':
        start, end = 1536, 1599
    else:  # right
        start, end = 1599, 1662
    
    if skeleton_flat.ndim == 3:
        batch, frames, _ = skeleton_flat.shape
        hand_flat = skeleton_flat[:, :, start:end]
        hand_xyz = hand_flat.reshape(batch, frames, 21, 3)
    else:
        batch, _ = skeleton_flat.shape
        hand_flat = skeleton_flat[:, start:end]
        hand_xyz = hand_flat.reshape(batch, 21, 3)
    
    return hand_xyz


def bone_length_loss(skeleton_flat, reference_skeleton=None):
    """
    Penalize changes in bone lengths (bones should maintain constant length).
    
    Args:
        skeleton_flat: (batch, num_frames, 1662) predicted skeleton
        reference_skeleton: (batch, num_frames, 1662) optional reference for bone lengths
        
    Returns:
        loss: scalar tensor
    """
    batch, frames, _ = skeleton_flat.shape
    
    # Extract pose keypoints
    pose = extract_pose_keypoints(skeleton_flat)  # (batch, frames, 33, 3)
    
    # Compute bone lengths for each connection
    total_loss = 0.0
    num_bones = 0
    
    for start_idx, end_idx in POSE_CONNECTIONS:
        # Get bone vectors
        start_points = pose[:, :, start_idx, :]  # (batch, frames, 3)
        end_points = pose[:, :, end_idx, :]
        
        bone_vectors = end_points - start_points
        bone_lengths = torch.norm(bone_vectors, dim=-1)  # (batch, frames)
        
        # Bone length should be constant across frames
        # Penalize variance in bone length over time
        bone_length_variance = bone_lengths.var(dim=1).mean()
        total_loss += bone_length_variance
        num_bones += 1
    
    # Also check hands
    for hand in ['left', 'right']:
        hand_kpts = extract_hand_keypoints(skeleton_flat, hand)  # (batch, frames, 21, 3)
        
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_points = hand_kpts[:, :, start_idx, :]
            end_points = hand_kpts[:, :, end_idx, :]
            
            bone_vectors = end_points - start_points
            bone_lengths = torch.norm(bone_vectors, dim=-1)
            
            bone_length_variance = bone_lengths.var(dim=1).mean()
            total_loss += bone_length_variance
            num_bones += 1
    
    return total_loss / num_bones


def temporal_smoothness_loss(skeleton_flat):
    """
    Penalize jerky motion (encourage smooth transitions).
    
    Args:
        skeleton_flat: (batch, num_frames, 1662)
        
    Returns:
        loss: scalar tensor
    """
    # Compute velocity (first derivative)
    velocity = skeleton_flat[:, 1:, :] - skeleton_flat[:, :-1, :]
    
    # Compute acceleration (second derivative)
    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
    
    # Penalize large accelerations (jerk)
    smoothness_loss = (acceleration ** 2).mean()
    
    return smoothness_loss


def symmetry_loss(skeleton_flat):
    """
    Encourage left-right symmetry in hands.
    
    Args:
        skeleton_flat: (batch, num_frames, 1662)
        
    Returns:
        loss: scalar tensor
    """
    left_hand = extract_hand_keypoints(skeleton_flat, 'left')
    right_hand = extract_hand_keypoints(skeleton_flat, 'right')
    
    # Mirror right hand (flip x coordinate)
    right_hand_mirrored = right_hand.clone()
    right_hand_mirrored[:, :, :, 0] = 1.0 - right_hand_mirrored[:, :, :, 0]
    
    # Compute difference (should be similar when mirrored)
    sym_loss = F.mse_loss(left_hand, right_hand_mirrored)
    
    return sym_loss


def coordinate_range_loss(skeleton_flat):
    """
    Penalize coordinates outside valid range [0, 1].
    
    Args:
        skeleton_flat: (batch, num_frames, 1662)
        
    Returns:
        loss: scalar tensor
    """
    # Penalize values < 0 or > 1
    below_zero = F.relu(-skeleton_flat)
    above_one = F.relu(skeleton_flat - 1.0)
    
    range_loss = (below_zero ** 2).mean() + (above_one ** 2).mean()
    
    return range_loss


def pose_validity_loss(skeleton_flat):
    """
    Penalize invalid pose configurations (e.g., impossible joint angles).
    
    Args:
        skeleton_flat: (batch, num_frames, 1662)
        
    Returns:
        loss: scalar tensor
    """
    pose = extract_pose_keypoints(skeleton_flat)  # (batch, frames, 33, 3)
    
    # Check shoulder-elbow-wrist angles (should be reasonable)
    # Left arm: shoulder(11), elbow(13), wrist(15)
    # Right arm: shoulder(12), elbow(14), wrist(16)
    
    total_loss = 0.0
    
    for shoulder_idx, elbow_idx, wrist_idx in [(11, 13, 15), (12, 14, 16)]:
        shoulder = pose[:, :, shoulder_idx, :]
        elbow = pose[:, :, elbow_idx, :]
        wrist = pose[:, :, wrist_idx, :]
        
        # Vectors
        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        
        # Normalize
        upper_arm_norm = F.normalize(upper_arm, dim=-1)
        forearm_norm = F.normalize(forearm, dim=-1)
        
        # Dot product (cosine of angle)
        cos_angle = (upper_arm_norm * forearm_norm).sum(dim=-1)
        
        # Penalize extreme angles (< 30° or > 170°)
        # cos(30°) ≈ 0.866, cos(170°) ≈ -0.985
        too_straight = F.relu(cos_angle - 0.866)  # > 30° is OK
        too_bent = F.relu(-0.985 - cos_angle)  # < 170° is OK
        
        total_loss += (too_straight ** 2).mean() + (too_bent ** 2).mean()
    
    return total_loss


def hand_coherence_loss(skeleton_flat):
    """
    Penalize incoherent hand shapes (fingers should move together).
    
    Args:
        skeleton_flat: (batch, num_frames, 1662)
        
    Returns:
        loss: scalar tensor
    """
    left_hand = extract_hand_keypoints(skeleton_flat, 'left')
    right_hand = extract_hand_keypoints(skeleton_flat, 'right')
    
    total_loss = 0.0
    
    for hand in [left_hand, right_hand]:
        # Check finger lengths (should be consistent)
        # Each finger: 4 joints (0->1->2->3->4 for thumb, etc.)
        
        for finger_start in [1, 5, 9, 13, 17]:  # Start of each finger
            finger_joints = []
            for i in range(4):
                if finger_start + i < 21:
                    finger_joints.append(hand[:, :, finger_start + i, :])
            
            # Compute finger length (sum of bone lengths)
            finger_length = 0
            for i in range(len(finger_joints) - 1):
                bone = finger_joints[i+1] - finger_joints[i]
                bone_length = torch.norm(bone, dim=-1)
                finger_length += bone_length
            
            # Finger length should be consistent across frames
            length_variance = finger_length.var(dim=1).mean()
            total_loss += length_variance
    
    return total_loss / 10  # 10 fingers total


def motion_naturalness_loss(skeleton_flat):
    """
    Penalize unnatural motion patterns (sudden jumps, jitter).
    
    Args:
        skeleton_flat: (batch, num_frames, 1662)
        
    Returns:
        loss: scalar tensor
    """
    # Compute velocity
    velocity = skeleton_flat[:, 1:, :] - skeleton_flat[:, :-1, :]
    
    # Velocity should be smooth (not jump around)
    velocity_change = velocity[:, 1:, :] - velocity[:, :-1, :]
    
    # Penalize large velocity changes (jerk)
    jerk_loss = (velocity_change ** 2).mean()
    
    # Also penalize very high velocities (sudden jumps)
    velocity_magnitude = torch.norm(velocity, dim=-1)
    high_velocity = F.relu(velocity_magnitude - 0.1)  # Threshold: 0.1
    velocity_loss = (high_velocity ** 2).mean()
    
    return jerk_loss + velocity_loss


def perceptual_loss(skeleton_flat):
    """
    Combined perceptual loss using MediaPipe pretrained model + custom rules.
    
    Args:
        skeleton_flat: (batch, num_frames, 1662)
        
    Returns:
        loss: scalar tensor
    """
    # Import MediaPipe loss
    try:
        from models.mediapipe_loss import mediapipe_perceptual_loss
        
        # Use MediaPipe pretrained model (70% weight)
        mp_loss = mediapipe_perceptual_loss(skeleton_flat)
        
        # Add custom rule-based losses (30% weight)
        pose_loss = pose_validity_loss(skeleton_flat)
        hand_loss = hand_coherence_loss(skeleton_flat)
        motion_loss = motion_naturalness_loss(skeleton_flat)
        
        custom_loss = 0.3 * pose_loss + 0.3 * hand_loss + 0.4 * motion_loss
        
        # Combine: 70% MediaPipe + 30% custom
        total = 0.7 * mp_loss + 0.3 * custom_loss
        
    except Exception as e:
        # Fallback to custom loss if MediaPipe fails
        # Suppress warning after first occurrence
        if not hasattr(perceptual_loss, '_warned'):
            print(f"Note: Using custom perceptual loss (MediaPipe: {e})")
            perceptual_loss._warned = True
        
        pose_loss = pose_validity_loss(skeleton_flat)
        hand_loss = hand_coherence_loss(skeleton_flat)
        motion_loss = motion_naturalness_loss(skeleton_flat)
        total = 0.3 * pose_loss + 0.3 * hand_loss + 0.4 * motion_loss
    
    return total


def combined_constraint_loss(
    skeleton_flat,
    mse_loss,
    bone_weight=0.1,
    smooth_weight=0.05,
    symmetry_weight=0.02,
    range_weight=0.1,
    perceptual_weight=0.15
):
    """
    Combine all constraint losses including perceptual loss.
    
    Args:
        skeleton_flat: (batch, num_frames, 1662) predicted skeleton
        mse_loss: MSE loss (already computed)
        bone_weight: Weight for bone length constraint
        smooth_weight: Weight for smoothness constraint
        symmetry_weight: Weight for symmetry constraint
        range_weight: Weight for coordinate range constraint
        perceptual_weight: Weight for perceptual loss
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual losses for logging
    """
    # Compute constraint losses
    bone_loss = bone_length_loss(skeleton_flat)
    smooth_loss = temporal_smoothness_loss(skeleton_flat)
    sym_loss = symmetry_loss(skeleton_flat)
    range_loss = coordinate_range_loss(skeleton_flat)
    percept_loss = perceptual_loss(skeleton_flat)
    
    # Combine
    total_loss = (
        mse_loss +
        bone_weight * bone_loss +
        smooth_weight * smooth_loss +
        symmetry_weight * sym_loss +
        range_weight * range_loss +
        perceptual_weight * percept_loss
    )
    
    loss_dict = {
        'mse': mse_loss.item(),
        'bone': bone_loss.item(),
        'smooth': smooth_loss.item(),
        'symmetry': sym_loss.item(),
        'range': range_loss.item(),
        'perceptual': percept_loss.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict
