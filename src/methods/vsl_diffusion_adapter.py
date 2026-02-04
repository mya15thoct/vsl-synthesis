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
            start_pose: Starting pose in VSL format (1662,) or (554, 3)
            end_pose: Ending pose in VSL format
            num_frames: Number of frames to generate
            use_mdm: Ignored (kept for API compatibility)
            
        Returns:
            Transition sequence (num_frames, 554, 3)
            
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
        if len(start_pose_flat) != 1662:
            raise ValueError(f"Expected 1662 features, got {len(start_pose_flat)}")
        
        # Convert to tensors (use torch.tensor to avoid numpy 1.23.5 issue)
        start_tensor = torch.tensor(start_pose_flat, dtype=torch.float32, device=self.device)
        end_tensor = torch.tensor(end_pose_flat, dtype=torch.float32, device=self.device)
        
        # Create condition (concatenate start + end)
        condition = torch.cat([start_tensor, end_tensor], dim=-1).unsqueeze(0)  # (1, 3324)
        
        # Start with random noise
        x_t = torch.randn(1, num_frames, 1662, device=self.device)
        
        # Denoising loop
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                # Predict noise
                t_tensor = torch.tensor([t], device=self.device).long()
                noise_pred = self.model(x_t, t_tensor, condition)
                
                # Denoise step
                x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample
        
        # Convert back to numpy
        transition = x_t[0].cpu().numpy()  # (num_frames, 1662)
        
        # Normalize to 0-1 range using min-max scaling (preserve structure better than clipping)
        # This ensures coordinates are in valid range while maintaining relative positions
        min_val = transition.min()
        max_val = transition.max()
        if max_val > min_val:
            transition = (transition - min_val) / (max_val - min_val)
        else:
            # If all values are the same, just clip to 0-1
            transition = np.clip(transition, 0.0, 1.0)
        
        # Reshape to (num_frames, 554, 3)
        transition = transition.reshape(num_frames, 554, 3)
        
        return transition


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
        Transition sequence (num_frames, 554, 3)
    """
    generator = VSLDiffusionGenerator(model_path=model_path)
    generator.load_model()
    return generator.generate_transition(start_pose, end_pose, num_frames)


if __name__ == "__main__":
    print("VSL Diffusion Adapter Module")
    print("=" * 50)
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    
    start_pose = np.random.rand(1662).astype(np.float32)
    end_pose = np.random.rand(1662).astype(np.float32)
    
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
