#!/usr/bin/env python3
"""
Test diffusers compatibility and suggest fix
"""

import sys

print("Testing diffusers compatibility...")

try:
    import numpy as np
    print(f"numpy version: {np.__version__}")
except ImportError as e:
    print(f" numpy not found: {e}")
    sys.exit(1)

try:
    import diffusers
    print(f"diffusers version: {diffusers.__version__}")
except ImportError as e:
    print(f"diffusers not found: {e}")
    sys.exit(1)

# Test DDPMScheduler
try:
    from diffusers import DDPMScheduler
    
    # Try direct initialization
    print("\nTesting direct initialization...")
    try:
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
        print("Direct initialization works!")
    except TypeError as e:
        print(f"✗ Direct initialization failed: {e}")
        print("\nTrying from_config method...")
        
        # Try from_config
        config = {
            "num_train_timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
            "clip_sample": False,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        }
        scheduler = DDPMScheduler.from_config(config)
        print("from_config works!")
        
except Exception as e:
    print(f"✗ DDPMScheduler test failed: {e}")
    print("\n" + "="*50)
    print("SOLUTION: Upgrade diffusers")
    print("  pip install --upgrade diffusers")
    print("="*50)
    sys.exit(1)

print("\nAll tests passed!")
print("\nRecommended versions:")
print("  numpy>=1.24.0")
print("  diffusers>=0.25.0")
