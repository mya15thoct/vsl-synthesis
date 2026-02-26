#!/usr/bin/env python3
"""
MicT Dataset
arXiv 2508.04049 — Continuous Motion Transition Generation

Mỗi sample chứa:
    full_sequence:    (T, 1659) — câu hoàn chỉnh (signs + transition)
    masked_sequence:  (T, 1659) — transition frames = 0 (observation condition m)
    frame_mask:       (T,)      — 1.0 = transition frame, 0.0 = observed
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MicTDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', max_len: int = 300):
        """
        Args:
            data_dir: thư mục chứa train/ và val/ folders
            split: 'train' hoặc 'val'
            max_len: độ dài tối đa của sequence (padding/truncate)
        """
        self.data_dir = Path(data_dir) / split
        self.max_len = max_len
        self.files = sorted(self.data_dir.glob("*.npz"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        print(f"MicTDataset [{split}]: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        full_seq = data['full_sequence'].astype(np.float32)    # (T, 1659)
        masked_seq = data['masked_sequence'].astype(np.float32) # (T, 1659)
        frame_mask = data['frame_mask'].astype(np.float32)      # (T,)

        T = len(full_seq)

        # Truncate nếu quá dài
        if T > self.max_len:
            full_seq = full_seq[:self.max_len]
            masked_seq = masked_seq[:self.max_len]
            frame_mask = frame_mask[:self.max_len]
            T = self.max_len

        return {
            'full_sequence': torch.from_numpy(full_seq),    # (T, 1659)
            'masked_sequence': torch.from_numpy(masked_seq), # (T, 1659)
            'frame_mask': torch.from_numpy(frame_mask),      # (T,)
            'length': T,
        }


def collate_fn_mict(batch):
    """Pad sequences trong batch về cùng độ dài."""
    max_len = max(item['length'] for item in batch)
    feat_dim = batch[0]['full_sequence'].shape[1]

    full_seqs = torch.zeros(len(batch), max_len, feat_dim)
    masked_seqs = torch.zeros(len(batch), max_len, feat_dim)
    masks = torch.zeros(len(batch), max_len)        # transition mask
    padding_masks = torch.zeros(len(batch), max_len) # padding mask

    for i, item in enumerate(batch):
        T = item['length']
        full_seqs[i, :T] = item['full_sequence']
        masked_seqs[i, :T] = item['masked_sequence']
        masks[i, :T] = item['frame_mask']
        padding_masks[i, :T] = 1.0  # valid frames

    return full_seqs, masked_seqs, masks, padding_masks
