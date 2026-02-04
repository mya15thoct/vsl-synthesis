"""
PyTorch Dataset for VSL Diffusion Training

Loads preprocessed transition examples for training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
from typing import Tuple


class VSLTransitionDataset(Dataset):
    """
    Dataset for VSL transition training.
    
    Each example contains:
        - start_pose: (1662,) starting pose
        - end_pose: (1662,) ending pose
        - ground_truth: (num_frames, 1662) ground truth transition
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Args:
            data_dir: Root directory containing train/val folders
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir) / split
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Find all transition files
        self.transition_files = sorted(self.data_dir.glob("transition_*.npz"))
        
        if len(self.transition_files) == 0:
            raise ValueError(f"No transition files found in {self.data_dir}")
        
        print(f"Loaded {len(self.transition_files)} transitions from {split} split")
    
    def __len__(self) -> int:
        return len(self.transition_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            start_pose: (1662,)
            end_pose: (1662,)
            ground_truth: (num_frames, 1662)
        """
        # Load transition
        data = np.load(self.transition_files[idx])
        
        # Convert directly to torch tensors (bypass numpy compatibility issue)
        # Using torch.tensor instead of torch.from_numpy to avoid numpy 1.23.5 issue
        start_pose = torch.tensor(data['start_pose'], dtype=torch.float32)
        end_pose = torch.tensor(data['end_pose'], dtype=torch.float32)
        ground_truth = torch.tensor(data['ground_truth'], dtype=torch.float32)
        
        return start_pose, end_pose, ground_truth


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    Pads sequences to the same length within a batch.
    """
    start_poses, end_poses, ground_truths = zip(*batch)
    
    # Stack start/end poses (all same size)
    start_poses = torch.stack(start_poses)
    end_poses = torch.stack(end_poses)
    
    # Find max length in batch
    max_len = max(gt.shape[0] for gt in ground_truths)
    
    # Pad ground truths
    padded_gts = []
    masks = []
    
    for gt in ground_truths:
        seq_len = gt.shape[0]
        
        # Pad if needed
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, gt.shape[1])
            padded_gt = torch.cat([gt, padding], dim=0)
            mask = torch.cat([
                torch.ones(seq_len),
                torch.zeros(max_len - seq_len)
            ])
        else:
            padded_gt = gt
            mask = torch.ones(seq_len)
        
        padded_gts.append(padded_gt)
        masks.append(mask)
    
    padded_gts = torch.stack(padded_gts)
    masks = torch.stack(masks)
    
    return start_poses, end_poses, padded_gts, masks


if __name__ == "__main__":
    # Test dataset
    print("Testing VSL Transition Dataset...")
    
    # This will fail if data not prepared yet
    try:
        dataset = VSLTransitionDataset("data/diffusion", split="train")
        
        # Get first example
        start, end, gt = dataset[0]
        
        print(f"Dataset test passed!")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Start pose shape: {start.shape}")
        print(f"  End pose shape: {end.shape}")
        print(f"  Ground truth shape: {gt.shape}")
    except Exception as e:
        print(f"Warning: Dataset not ready: {e}")
        print("  Run prepare_diffusion_data.py first!")
