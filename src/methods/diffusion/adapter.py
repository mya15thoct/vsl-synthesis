"""
VSL Diffusion Adapter for Transition Generation (Inference)

This module provides VSLDiffusionGenerator for using the trained diffusion
model in the synthesis pipeline.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional

from .model import VSLDiffusionModel
from .scheduler import SimpleDDPMScheduler


class VSLDiffusionGenerator:
    """Generate transitions using VSL-native diffusion model."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.scheduler = None
        self._model_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self):
        """Load trained diffusion model."""
        if self._model_loaded:
            return
        
        if self.model_path is None:
            raise RuntimeError("Model path not specified.")
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise RuntimeError(f"Model not found: {model_path}")
        
        print(f"Loading VSL diffusion model from {model_path}...")
        
        self.model = VSLDiffusionModel.load(str(model_path), device=str(self.device))
        self.model.eval()
        
        self.scheduler = SimpleDDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2"
        )
        self.scheduler.set_timesteps(50)
        
        self._model_loaded = True
        print(f"VSL diffusion model loaded successfully on {self.device}!")
    
    def generate_transition(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        num_frames: int = 10,
        **kwargs
    ) -> np.ndarray:
        """
        Generate smooth transition between two poses.
        
        Args:
            start_pose: (1659,) or (553, 3) starting pose
            end_pose: (1659,) or (553, 3) ending pose
            num_frames: Number of frames to generate
            
        Returns:
            Transition sequence (num_frames, 553, 3)
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Flatten inputs
        start_pose_flat = start_pose.flatten() if start_pose.ndim > 1 else start_pose
        end_pose_flat = end_pose.flatten() if end_pose.ndim > 1 else end_pose
        
        if len(start_pose_flat) != 1659:
            raise ValueError(f"Expected 1659 features, got {len(start_pose_flat)}")
        
        # Normalize inputs to [0, 1] to match training data
        # Training uses global fixed range: [-2, 2] -> [0, 1]
        def normalize_pose(pose):
            pose_clipped = np.clip(pose, -2.0, 2.0)
            return (pose_clipped + 2.0) / 4.0
        
        start_pose_norm = normalize_pose(start_pose_flat)
        end_pose_norm = normalize_pose(end_pose_flat)
        
        print(f"\n[DEBUG] Input normalization:")
        print(f"  Start: [{start_pose_flat.min():.4f}, {start_pose_flat.max():.4f}] -> [{start_pose_norm.min():.4f}, {start_pose_norm.max():.4f}]")
        print(f"  End:   [{end_pose_flat.min():.4f}, {end_pose_flat.max():.4f}] -> [{end_pose_norm.min():.4f}, {end_pose_norm.max():.4f}]")
        
        # Convert to tensors
        start_tensor = torch.tensor(start_pose_norm, dtype=torch.float32, device=self.device)
        end_tensor = torch.tensor(end_pose_norm, dtype=torch.float32, device=self.device)
        condition = torch.cat([start_tensor, end_tensor], dim=-1).unsqueeze(0)  # (1, 3318)
        
        # Start with uniform noise in [0, 1] (matches training data distribution)
        x_t = torch.rand(1, num_frames, 1659, device=self.device)
        target_length = torch.tensor([num_frames], dtype=torch.long, device=self.device)
        
        # Denoising loop
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t_tensor = torch.tensor([t], device=self.device).long()
                noise_pred = self.model(x_t, t_tensor, condition, target_length)
                # Tanh in model already bounds to [-1, 1], but clamp for safety
                noise_pred = torch.clamp(noise_pred, -1.0, 1.0)
                x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample
        
        # Convert to numpy
        transition = x_t[0].cpu().numpy()  # (num_frames, 1659)
        
        print(f"\n[DEBUG] Raw model output: [{transition.min():.4f}, {transition.max():.4f}]")
        
        # Denormalize: [0, 1] -> [-2, 2] (reverse of normalization)
        transition_denorm = transition * 4.0 - 2.0
        
        print(f"[DEBUG] Denormalized: [{transition_denorm.min():.4f}, {transition_denorm.max():.4f}]")
        
        # Reshape to (num_frames, 553, 3)
        return transition_denorm.reshape(num_frames, 553, 3)
