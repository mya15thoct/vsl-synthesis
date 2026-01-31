"""
Skeleton sequence concatenation module.

This module provides functions to concatenate multiple skeleton sequences
from individual word videos, preparing them for transition generation.
"""

import numpy as np
from typing import List, Union
from pathlib import Path


def resolve_word_path(word_path: Union[str, Path]) -> Path:
    """
    Resolve word path to actual .npy file.
    
    Handles two cases:
    1. Direct file: word.npy
    2. Folder with multiple files: word/sample1.npy, word/sample2.npy, ...
       -> Returns first .npy file in folder
    
    Args:
        word_path: Path to word (can be file or folder)
        
    Returns:
        Path to .npy file
        
    Raises:
        FileNotFoundError: If path doesn't exist or no .npy files found
    """
    word_path = Path(word_path)
    
    # Case 1: Direct .npy file
    if word_path.is_file() and word_path.suffix == '.npy':
        return word_path
    
    # Case 2: Folder containing .npy files
    if word_path.is_dir():
        npy_files = sorted(word_path.glob("*.npy"))
        if npy_files:
            return npy_files[0]  # Return first file
        else:
            raise FileNotFoundError(f"No .npy files found in folder: {word_path}")
    
    # Case 3: Path doesn't exist
    raise FileNotFoundError(f"Path not found: {word_path}")


def load_skeleton_sequence(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load skeleton sequence from .npy file.
    
    Args:
        file_path: Path to .npy file containing skeleton data
        
    Returns:
        Skeleton sequence of shape (frames, 543, 3)
        - frames: number of frames in the video
        - 543: number of keypoints (MediaPipe holistic)
        - 3: x, y, z coordinates
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    # Resolve path (handles both files and folders)
    file_path = resolve_word_path(file_path)
    
    skeleton = np.load(file_path)
    
    # Handle different data formats
    if skeleton.ndim == 2:
        # 2D array: (frames, keypoints*3) -> reshape to (frames, keypoints, 3)
        num_frames = skeleton.shape[0]
        num_features = skeleton.shape[1]
        
        # Detect number of keypoints
        if num_features % 3 != 0:
            raise ValueError(f"Invalid shape {skeleton.shape}: second dimension must be divisible by 3")
        
        num_keypoints = num_features // 3
        
        # Reshape to 3D
        skeleton = skeleton.reshape(num_frames, num_keypoints, 3)
        print(f"  Reshaped from {(num_frames, num_features)} to {skeleton.shape}")
    
    # Validate shape
    if skeleton.ndim != 3:
        raise ValueError(f"Expected 3D array after reshaping, got shape {skeleton.shape}")
    
    if skeleton.shape[2] != 3:
        raise ValueError(f"Expected 3 coordinates (x,y,z), got {skeleton.shape[2]}")
    
    # Note: We don't enforce 543 keypoints anymore, as different datasets may have different numbers
    num_keypoints = skeleton.shape[1]
    if num_keypoints != 543:
        print(f"  Warning: Found {num_keypoints} keypoints (expected 543). Continuing anyway...")
    
    return skeleton


def concatenate_sequences(
    skeleton_files: List[Union[str, Path]],
    transition_frames: int = 10,
    remove_duplicates: bool = True
) -> np.ndarray:
    """
    Concatenate multiple skeleton sequences with placeholder gaps for transitions.
    
    Args:
        skeleton_files: List of paths to .npy skeleton files
        transition_frames: Number of frames to reserve for transitions between words
        remove_duplicates: If True, remove duplicate frames at word boundaries
        
    Returns:
        Concatenated skeleton sequence with gaps for transitions
        Shape: (total_frames, 543, 3)
        - Gaps are filled with zeros and will be replaced by interpolation
        
    Example:
        >>> files = ['hello.npy', 'my.npy', 'name.npy']
        >>> result = concatenate_sequences(files, transition_frames=10)
        >>> # Result contains: hello + 10 gap + my + 10 gap + name
    """
    if not skeleton_files:
        raise ValueError("skeleton_files cannot be empty")
    
    sequences = []
    
    for file_path in skeleton_files:
        skeleton = load_skeleton_sequence(file_path)
        sequences.append(skeleton)
    
    # Calculate total frames needed
    total_frames = sum(seq.shape[0] for seq in sequences)
    total_frames += transition_frames * (len(sequences) - 1)  # Add transition gaps
    
    # Initialize output array
    concatenated = np.zeros((total_frames, 543, 3), dtype=np.float32)
    
    current_idx = 0
    
    for i, sequence in enumerate(sequences):
        seq_len = sequence.shape[0]
        
        # Copy sequence
        concatenated[current_idx:current_idx + seq_len] = sequence
        current_idx += seq_len
        
        # Add transition gap (except after last sequence)
        if i < len(sequences) - 1:
            # Gap is already zeros, just skip ahead
            current_idx += transition_frames
    
    return concatenated


def get_transition_boundaries(
    skeleton_files: List[Union[str, Path]],
    transition_frames: int = 10
) -> List[tuple]:
    """
    Get the frame indices where transitions should be inserted.
    
    Args:
        skeleton_files: List of paths to .npy skeleton files
        transition_frames: Number of frames for each transition
        
    Returns:
        List of tuples (start_idx, end_idx, start_pose_idx, end_pose_idx)
        - start_idx: Start of transition gap
        - end_idx: End of transition gap
        - start_pose_idx: Index of last frame of previous word
        - end_pose_idx: Index of first frame of next word
        
    Example:
        >>> boundaries = get_transition_boundaries(['hello.npy', 'my.npy'])
        >>> # [(30, 40, 29, 40)] means:
        >>> # - Transition from frame 30 to 40
        >>> # - Start pose at frame 29 (last of 'hello')
        >>> # - End pose at frame 40 (first of 'my')
    """
    sequences = [load_skeleton_sequence(f) for f in skeleton_files]
    
    boundaries = []
    current_idx = 0
    
    for i in range(len(sequences) - 1):
        seq_len = sequences[i].shape[0]
        
        # Last frame of current word
        start_pose_idx = current_idx + seq_len - 1
        
        # Start of transition gap
        transition_start = current_idx + seq_len
        
        # End of transition gap
        transition_end = transition_start + transition_frames
        
        # First frame of next word (after gap)
        end_pose_idx = transition_end
        
        boundaries.append((
            transition_start,
            transition_end,
            start_pose_idx,
            end_pose_idx
        ))
        
        current_idx += seq_len + transition_frames
    
    return boundaries


if __name__ == "__main__":
    # Example usage
    print("Concatenation module loaded successfully!")
    print("\nExample usage:")
    print("  from src.synthesis.concatenation import concatenate_sequences")
    print("  result = concatenate_sequences(['hello.npy', 'my.npy'], transition_frames=10)")
