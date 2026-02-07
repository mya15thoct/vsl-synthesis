"""
Custom DDPM Scheduler to bypass numpy version conflict

This is a minimal implementation that avoids the numpy compatibility issue
in diffusers 0.36.0 with numpy 1.23.5
"""

import torch
import numpy as np
from typing import Union


class SimpleDDPMScheduler:
    """
    Simplified DDPM scheduler that works with older numpy versions.
    
    Based on the DDPM paper: https://arxiv.org/abs/2006.11239
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        self.num_train_timesteps = num_train_timesteps
        
        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "squaredcos_cap_v2":
            # Cosine schedule from improved DDPM
            self.betas = self._betas_for_alpha_bar(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # For adding noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For denoising
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Config for compatibility
        self.config = type('Config', (), {
            'num_train_timesteps': num_train_timesteps,
            'beta_start': beta_start,
            'beta_end': beta_end,
            'beta_schedule': beta_schedule
        })()
    
    def _betas_for_alpha_bar(self, num_timesteps: int, max_beta: float = 0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function.
        Uses cosine schedule from improved DDPM.
        """
        def alpha_bar(time_step):
            return np.cos((time_step + 0.008) / 1.008 * np.pi / 2) ** 2
        
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        
        return torch.tensor(betas, dtype=torch.float32)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples at given timesteps.
        
        Args:
            original_samples: (batch, ...)
            noise: (batch, ...)
            timesteps: (batch,) timestep indices
            
        Returns:
            noisy_samples: (batch, ...)
        """
        # Move timesteps to CPU for indexing (fix device mismatch)
        timesteps_cpu = timesteps.cpu()
        
        # Move coefficients to same device as samples
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps_cpu].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps_cpu].to(original_samples.device)
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor
    ):
        """
        Predict the sample at the previous timestep.
        
        Args:
            model_output: predicted noise
            timestep: current timestep
            sample: current sample
            
        Returns:
            Object with prev_sample attribute
        """
        # Convert timestep to int if tensor (fix device issues)
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()
        
        # Get coefficients
        beta_t = self.betas[timestep].to(sample.device)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[timestep].to(sample.device)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep].to(sample.device)
        
        # Reshape for broadcasting
        while len(beta_t.shape) < len(sample.shape):
            beta_t = beta_t.unsqueeze(-1)
            sqrt_recip_alpha_t = sqrt_recip_alpha_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)
        
        # Predict previous sample mean
        pred_prev_sample = sqrt_recip_alpha_t * (sample - beta_t * model_output / sqrt_one_minus_alpha_cumprod_t)
        
        # Add noise if not final step
        if timestep > 0:
            variance = self.posterior_variance[timestep].to(sample.device)
            while len(variance.shape) < len(sample.shape):
                variance = variance.unsqueeze(-1)
            
            noise = torch.randn_like(sample)
            pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise
        
        # Return object with prev_sample attribute for compatibility
        return type('StepOutput', (), {'prev_sample': pred_prev_sample})()
    
    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference."""
        # Use evenly spaced timesteps
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long
        )


if __name__ == "__main__":
    # Test scheduler
    print("Testing SimpleDDPMScheduler...")
    
    scheduler = SimpleDDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    
    # Test add_noise
    batch_size = 4
    sample = torch.randn(batch_size, 10, 1659)
    noise = torch.randn_like(sample)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    noisy = scheduler.add_noise(sample, noise, timesteps)
    
    print(f"Sample shape: {sample.shape}")
    print(f"Noisy shape: {noisy.shape}")
    print(f"Scheduler test passed!")
