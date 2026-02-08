"""
VSL Diffusion Adapter for Transition Generation

This module provides an adapter for using the trained VSL diffusion model
in the synthesis pipeline, replacing the MDM adapter.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional
from diffusers import DDPMScheduler
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.vsl_diffusion import VSLDiffusionModel


class VSLDiffusionGenerator:
    """
    Generate transitions using VSL-native diffusion model.
    
    This class provides the same API as MDMTransitionGenerator but uses
    the VSL-native diffusion model instead.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize VSL diffusion generator.
        
        Args:
            model_path: Path to trained model checkpoint (e.g., 'models/vsl_diffusion/best.pt')
        """
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
            raise RuntimeError("Model path not specified. Please provide path to trained model.")
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise RuntimeError(f"Model not found: {model_path}")
        
        print(f"Loading VSL diffusion model from {model_path}...")
        
        # Load model
        self.model = VSLDiffusionModel.load(str(model_path), device=str(self.device))
        self.model.eval()
        
        # Create diffusion scheduler (use custom implementation)
        from src.models.custom_scheduler import SimpleDDPMScheduler
        
        self.scheduler = SimpleDDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2"
        )
        
        # Set to 50 inference steps (faster than 1000)
        self.scheduler.set_timesteps(50)
        
        self._model_loaded = True
        print(f"VSL diffusion model loaded successfully on {self.device}!")
    
    def generate_transition(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        num_frames: int = 10,
        use_mdm: bool = True  # Keep for API compatibility
    ) -> np.ndarray:
        """
        Generate smooth transition between two poses.
        
        Args:
            start_pose: Starting pose in VSL format (1659,) or (553, 3)
            end_pose: Ending pose in VSL format
            num_frames: Number of frames to generate
            use_mdm: Ignored (kept for API compatibility)
            
        Returns:
            Transition sequence (num_frames, 553, 3)
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        # Flatten inputs if needed
        if start_pose.ndim > 1:
            start_pose_flat = start_pose.flatten()
        else:
            start_pose_flat = start_pose
        
        if end_pose.ndim > 1:
            end_pose_flat = end_pose.flatten()
        else:
            end_pose_flat = end_pose
        
        # Ensure correct size
        if len(start_pose_flat) != 1659:
            raise ValueError(f"Expected 1659 features, got {len(start_pose_flat)}")
        
        # CRITICAL: Normalize inputs to [0, 1] to match training data
        # Training data was normalized using render.py's normalization
        # Original MediaPipe range is approximately [-1.5, 1.5]
        # We use the same normalization as in render.py
        def normalize_pose(pose):
            # Clip to reasonable range first
            pose_clipped = np.clip(pose, -2.0, 2.0)
            # Normalize to [0, 1]
            pose_norm = (pose_clipped + 2.0) / 4.0  # [-2, 2] -> [0, 1]
            return pose_norm
        
        start_pose_norm = normalize_pose(start_pose_flat)
        end_pose_norm = normalize_pose(end_pose_flat)
        
        print(f"\n[DEBUG] Input normalization:")
        print(f"  Start pose: [{start_pose_flat.min():.4f}, {start_pose_flat.max():.4f}] -> [{start_pose_norm.min():.4f}, {start_pose_norm.max():.4f}]")
        print(f"  End pose: [{end_pose_flat.min():.4f}, {end_pose_flat.max():.4f}] -> [{end_pose_norm.min():.4f}, {end_pose_norm.max():.4f}]")
        
        # Convert to tensors
        start_tensor = torch.tensor(start_pose_norm, dtype=torch.float32, device=self.device)
        end_tensor = torch.tensor(end_pose_norm, dtype=torch.float32, device=self.device)
        
        # Create condition (concatenate start + end)
        condition = torch.cat([start_tensor, end_tensor], dim=-1).unsqueeze(0)  # (1, 3318)
        
        # CRITICAL: Start with noise in [0, 1] range to match training data
        # Training data is normalized to [0, 1], so initial noise should be too
        # Use uniform noise [0, 1] or clipped Gaussian
        x_t = torch.rand(1, num_frames, 1659, device=self.device)  # Uniform [0, 1]
        
        # Alternative: Gaussian clipped to [0, 1]
        # x_t = torch.randn(1, num_frames, 1659, device=self.device) * 0.3 + 0.5
        # x_t = torch.clamp(x_t, 0.0, 1.0)
        
        # Create target_length tensor for length conditioning
        target_length = torch.tensor([num_frames], dtype=torch.long, device=self.device)
        
        # Denoising loop
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                # Predict noise (with length conditioning)
                t_tensor = torch.tensor([t], device=self.device).long()
                noise_pred = self.model(x_t, t_tensor, condition, target_length)
                
                # Denoise step
                x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample
        
        # Convert back to numpy
        transition = x_t[0].cpu().numpy()  # (num_frames, 1659)
        
        # Debug: Check raw model output
        print(f"\n[DEBUG] Raw model output:")
        print(f"  Shape: {transition.shape}")
        print(f"  Min: {transition.min():.4f}, Max: {transition.max():.4f}")
        print(f"  Mean: {transition.mean():.4f}, Std: {transition.std():.4f}")
        
        # Denormalize from [0, 1] back to MediaPipe range [-2, 2]
        # Model was trained on data normalized to [0, 1]
        # We need to reverse the normalization: x_norm = (x + 2) / 4
        # So: x = x_norm * 4 - 2
        
        transition_denorm = transition * 4.0 - 2.0
        
        print(f"\n[DEBUG] Output denormalization:")
        print(f"  Model output [0,1]: [{transition.min():.4f}, {transition.max():.4f}]")
        print(f"  Denormalized [-2,2]: [{transition_denorm.min():.4f}, {transition_denorm.max():.4f}]")
        print(f"  Mean: {transition_denorm.mean():.4f}, Std: {transition_denorm.std():.4f}")
        
        # Reshape to (num_frames, 553, 3)
        transition_denorm = transition_denorm.reshape(num_frames, 553, 3)
        
        return transition_denorm


# Convenience function for direct use
def generate_transition_vsl(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    num_frames: int = 10,
    model_path: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function to generate transition using VSL diffusion.
    
    Args:
        start_pose: Starting pose in VSL format
        end_pose: Ending pose in VSL format
        num_frames: Number of transition frames
        model_path: Path to trained model checkpoint
        
    Returns:
        Transition sequence (num_frames, 553, 3)
    """
    generator = VSLDiffusionGenerator(model_path=model_path)
    generator.load_model()
    return generator.generate_transition(start_pose, end_pose, num_frames)


if __name__ == "__main__":
    print("VSL Diffusion Adapter Module")
    print("=" * 50)
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    
    start_pose = np.random.rand(1659).astype(np.float32)
    end_pose = np.random.rand(1659).astype(np.float32)
    
    print(f"Start pose shape: {start_pose.shape}")
    print(f"End pose shape: {end_pose.shape}")
    
    # Note: This will fail if model not trained yet
    try:
        generator = VSLDiffusionGenerator(model_path="models/vsl_diffusion/best.pt")
        generator.load_model()
        
        transition = generator.generate_transition(start_pose, end_pose, num_frames=10)
        
        print(f"\nAdapter test passed!")
        print(f"  Transition shape: {transition.shape}")
    except Exception as e:
        print(f"\nWarning: Model not ready: {e}")
        print("  Train model first with: python scripts/train_diffusion.py")
