#!/usr/bin/env python3
"""
MicT Dataset
arXiv 2508.04049 — Continuous Motion Transition Generation

Mỗi sample chứa:
    full_sequence:    (T_words, 1659) — chỉ word frames, không zeros (ground truth)
    masked_sequence:  (T_total, 1659) — word frames + zero transitions (inference input)
    frame_mask:       (T_total,)      — 1.0 = transition frame, 0.0 = word frame

Lưu ý: T_words ≤ T_total (T_total = T_words + N_transitions * transition_frames)
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
            max_len: độ dài tối đa của full_sequence (word frames only)
        """
        self.data_dir = Path(data_dir) / split
        self.max_len = max_len
        all_files = sorted(self.data_dir.glob("*.npz"))

        if len(all_files) == 0:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        # Filter out corrupt files (BadZipFile, incomplete writes)
        self.files = []
        for f in all_files:
            try:
                import zipfile
                with zipfile.ZipFile(f, 'r'):
                    pass
                self.files.append(f)
            except Exception:
                pass

        n_bad = len(all_files) - len(self.files)
        print(f"MicTDataset [{split}]: {len(self.files)} samples ({n_bad} corrupt skipped)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        full_seq   = data['full_sequence'].astype(np.float32)    # (T_words, 1659)
        masked_seq = data['masked_sequence'].astype(np.float32)  # (T_total, 1659)
        frame_mask = data['frame_mask'].astype(np.float32)       # (T_total,)

        T_words = len(full_seq)
        T_total = len(masked_seq)

        # Truncate full_seq nếu quá dài
        if T_words > self.max_len:
            full_seq = full_seq[:self.max_len]
            T_words  = self.max_len

        # Truncate masked_seq proportionally nếu cần
        max_masked = int(self.max_len * T_total / max(len(data['full_sequence']), 1))
        if T_total > max_masked:
            masked_seq = masked_seq[:max_masked]
            frame_mask = frame_mask[:max_masked]
            T_total    = max_masked

        return {
            'full_sequence':   torch.tensor(full_seq,   dtype=torch.float32),
            'masked_sequence': torch.tensor(masked_seq, dtype=torch.float32),
            'frame_mask':      torch.tensor(frame_mask, dtype=torch.float32),
            'full_length':     T_words,
            'masked_length':   T_total,
        }


def collate_fn_mict(batch):
    """Pad full_sequence và masked_sequence về cùng độ dài (riêng biệt)."""
    max_full   = max(item['full_length']   for item in batch)
    max_masked = max(item['masked_length'] for item in batch)
    feat_dim   = batch[0]['full_sequence'].shape[1]

    full_seqs    = torch.zeros(len(batch), max_full,   feat_dim)
    masked_seqs  = torch.zeros(len(batch), max_masked, feat_dim)
    trans_masks  = torch.zeros(len(batch), max_masked)          # transition mask
    pad_masks    = torch.zeros(len(batch), max_full)            # valid word frames

    for i, item in enumerate(batch):
        Tf = item['full_length']
        Tm = item['masked_length']
        full_seqs[i,   :Tf] = item['full_sequence']
        masked_seqs[i, :Tm] = item['masked_sequence']
        trans_masks[i, :Tm] = item['frame_mask']
        pad_masks[i,   :Tf] = 1.0   # 1 = valid word frame

    return full_seqs, masked_seqs, trans_masks, pad_masks
