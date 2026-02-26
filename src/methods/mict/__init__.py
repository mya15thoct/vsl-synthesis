"""
MicT - Motion is the Choreographer
Continuous Motion Transition Generation
Based on arXiv 2508.04049
"""

from .model_mict import MicTDiffusionModel
from .dataset_mict import MicTDataset, collate_fn_mict
from .train_mict import main as train

__all__ = ['MicTDiffusionModel', 'MicTDataset', 'collate_fn_mict', 'train']
