"""
Configuration for VSL Synthesis project.
Auto-detects server vs local environment.
"""

import os
from pathlib import Path

# Detect if running on server
IS_SERVER = os.path.exists("/mnt/ngan/vsl_data")

if IS_SERVER:
    # Server paths
    DATA_DIR = Path("/mnt/ngan/vsl_data/sequences")
    OUTPUT_DIR = Path("/mnt/ngan/vsl_synthesis_outputs")
else:
    # Local paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data" / "words"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "baseline").mkdir(exist_ok=True)
(OUTPUT_DIR / "diffusion").mkdir(exist_ok=True)

# Settings
FPS = 30
TRANSITION_FRAMES = 10
NUM_KEYPOINTS = 543  # MediaPipe Holistic (will auto-detect if different)
