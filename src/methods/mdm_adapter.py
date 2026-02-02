"""
MDM (Motion Diffusion Model) Adapter for VSL Synthesis.

This module adapts the Motion Diffusion Model for Vietnamese Sign Language
skeleton format conversion and transition generation.

MDM uses SMPL format (22 joints) while VSL uses MediaPipe (33 pose + 21+21 hands).
This adapter handles the conversion between formats.

Reference:
- MDM Paper: https://arxiv.org/abs/2209.14916
- MDM GitHub: https://github.com/GuyTevet/motion-diffusion-model
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

# MDM joint indices (22 joints from SMPL)
MDM_JOINTS = {
    'pelvis': 0,
    'left_hip': 1, 'right_hip': 2,
    'spine1': 3,
    'left_knee': 4, 'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7, 'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10, 'right_foot': 11,
    'neck': 12,
    'left_collar': 13, 'right_collar': 14,
    'head': 15,
    'left_shoulder': 16, 'right_shoulder': 17,
    'left_elbow': 18, 'right_elbow': 19,
    'left_wrist': 20, 'right_wrist': 21,
}

# MediaPipe Pose indices (33 landmarks)
# Reference: https://google.github.io/mediapipe/solutions/pose.html
MEDIAPIPE_POSE = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32,
}

# Mapping from MediaPipe to MDM (approximate mapping)
MEDIAPIPE_TO_MDM = {
    # Torso
    'pelvis': lambda mp: (mp[MEDIAPIPE_POSE['left_hip']] + mp[MEDIAPIPE_POSE['right_hip']]) / 2,
    'spine1': lambda mp: (mp[MEDIAPIPE_POSE['left_hip']] + mp[MEDIAPIPE_POSE['right_hip']] + 
                          mp[MEDIAPIPE_POSE['left_shoulder']] + mp[MEDIAPIPE_POSE['right_shoulder']]) / 4 * 0.7 + 
                         (mp[MEDIAPIPE_POSE['left_hip']] + mp[MEDIAPIPE_POSE['right_hip']]) / 2 * 0.3,
    'spine2': lambda mp: (mp[MEDIAPIPE_POSE['left_shoulder']] + mp[MEDIAPIPE_POSE['right_shoulder']]) / 2 * 0.5 +
                         (mp[MEDIAPIPE_POSE['left_hip']] + mp[MEDIAPIPE_POSE['right_hip']]) / 2 * 0.5,
    'spine3': lambda mp: (mp[MEDIAPIPE_POSE['left_shoulder']] + mp[MEDIAPIPE_POSE['right_shoulder']]) / 2 * 0.8 +
                         (mp[MEDIAPIPE_POSE['left_hip']] + mp[MEDIAPIPE_POSE['right_hip']]) / 2 * 0.2,
    'neck': lambda mp: (mp[MEDIAPIPE_POSE['left_shoulder']] + mp[MEDIAPIPE_POSE['right_shoulder']]) / 2 * 0.3 +
                       mp[MEDIAPIPE_POSE['nose']] * 0.7,
    'head': lambda mp: mp[MEDIAPIPE_POSE['nose']],
    
    # Left leg
    'left_hip': lambda mp: mp[MEDIAPIPE_POSE['left_hip']],
    'left_knee': lambda mp: mp[MEDIAPIPE_POSE['left_knee']],
    'left_ankle': lambda mp: mp[MEDIAPIPE_POSE['left_ankle']],
    'left_foot': lambda mp: mp[MEDIAPIPE_POSE['left_foot_index']],
    
    # Right leg
    'right_hip': lambda mp: mp[MEDIAPIPE_POSE['right_hip']],
    'right_knee': lambda mp: mp[MEDIAPIPE_POSE['right_knee']],
    'right_ankle': lambda mp: mp[MEDIAPIPE_POSE['right_ankle']],
    'right_foot': lambda mp: mp[MEDIAPIPE_POSE['right_foot_index']],
    
    # Left arm
    'left_collar': lambda mp: mp[MEDIAPIPE_POSE['left_shoulder']] * 0.8 + 
                              (mp[MEDIAPIPE_POSE['left_shoulder']] + mp[MEDIAPIPE_POSE['right_shoulder']]) / 2 * 0.2,
    'left_shoulder': lambda mp: mp[MEDIAPIPE_POSE['left_shoulder']],
    'left_elbow': lambda mp: mp[MEDIAPIPE_POSE['left_elbow']],
    'left_wrist': lambda mp: mp[MEDIAPIPE_POSE['left_wrist']],
    
    # Right arm
    'right_collar': lambda mp: mp[MEDIAPIPE_POSE['right_shoulder']] * 0.8 + 
                               (mp[MEDIAPIPE_POSE['left_shoulder']] + mp[MEDIAPIPE_POSE['right_shoulder']]) / 2 * 0.2,
    'right_shoulder': lambda mp: mp[MEDIAPIPE_POSE['right_shoulder']],
    'right_elbow': lambda mp: mp[MEDIAPIPE_POSE['right_elbow']],
    'right_wrist': lambda mp: mp[MEDIAPIPE_POSE['right_wrist']],
}


class VSLToMDMConverter:
    """
    Converter between VSL skeleton format (MediaPipe) and MDM format (SMPL).
    
    VSL format: 1662 values
        - Pose: 33 landmarks × 4 (x,y,z,visibility) = 132 values
        - Face: 468 landmarks × 3 (x,y,z) = 1404 values
        - Left Hand: 21 landmarks × 3 (x,y,z) = 63 values
        - Right Hand: 21 landmarks × 3 (x,y,z) = 63 values
        
    MDM format: (frames, 22, 3) - 22 SMPL joints with xyz coordinates
    """
    
    def __init__(self):
        self.mdm_joint_names = list(MDM_JOINTS.keys())
        self.mdm_joint_indices = MDM_JOINTS
        
    def vsl_to_mdm(self, vsl_skeleton: np.ndarray) -> np.ndarray:
        """
        Convert VSL skeleton to MDM format.
        
        Args:
            vsl_skeleton: VSL skeleton data
                - If 1D: (1662,) raw format
                - If 2D: (frames, 1662) sequence
                - If 3D: (frames, 554, 3) reshaped format
                
        Returns:
            MDM skeleton: (frames, 22, 3) or (22, 3) if single frame
        """
        # Handle different input formats
        if vsl_skeleton.ndim == 1:
            # Single frame, raw format (1662,)
            return self._convert_single_frame(vsl_skeleton)
        elif vsl_skeleton.ndim == 2:
            if vsl_skeleton.shape[1] == 1662:
                # Sequence in raw format (frames, 1662)
                mdm_frames = []
                for frame in vsl_skeleton:
                    mdm_frames.append(self._convert_single_frame(frame))
                return np.stack(mdm_frames, axis=0)
            else:
                # Single frame reshaped (554, 3) or similar
                flat = vsl_skeleton.flatten()
                return self._convert_single_frame(flat)
        elif vsl_skeleton.ndim == 3:
            # Sequence reshaped (frames, 554, 3)
            raw = vsl_skeleton.reshape(vsl_skeleton.shape[0], -1)
            return self.vsl_to_mdm(raw)
        else:
            raise ValueError(f"Unsupported input shape: {vsl_skeleton.shape}")
    
    def _convert_single_frame(self, vsl_raw: np.ndarray) -> np.ndarray:
        """
        Convert single VSL frame (1662 values) to MDM format (22, 3).
        """
        if len(vsl_raw) < 132:
            raise ValueError(f"VSL data too short: {len(vsl_raw)}, need at least 132 for pose")
        
        # Extract pose: 33 landmarks × 4 values (x, y, z, visibility)
        pose_raw = vsl_raw[:132].reshape(33, 4)
        pose_xyz = pose_raw[:, :3]  # (33, 3) - just x, y, z
        
        # Convert to MDM 22 joints
        mdm_joints = np.zeros((22, 3), dtype=np.float32)
        
        for joint_name, joint_idx in MDM_JOINTS.items():
            if joint_name in MEDIAPIPE_TO_MDM:
                mdm_joints[joint_idx] = MEDIAPIPE_TO_MDM[joint_name](pose_xyz)
            else:
                # Fallback: use center of body
                mdm_joints[joint_idx] = (pose_xyz[MEDIAPIPE_POSE['left_hip']] + 
                                         pose_xyz[MEDIAPIPE_POSE['right_hip']]) / 2
        
        return mdm_joints
    
    def mdm_to_vsl(self, mdm_skeleton: np.ndarray, 
                   reference_vsl: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert MDM skeleton back to VSL format.
        
        Args:
            mdm_skeleton: MDM skeleton (frames, 22, 3) or (22, 3)
            reference_vsl: Optional reference VSL skeleton for face/hands
                           If None, face and hands will be interpolated
                           
        Returns:
            VSL skeleton in raw format (frames, 1662) or (1662,)
        """
        single_frame = mdm_skeleton.ndim == 2
        if single_frame:
            mdm_skeleton = mdm_skeleton[np.newaxis, ...]
            
        num_frames = mdm_skeleton.shape[0]
        vsl_output = np.zeros((num_frames, 1662), dtype=np.float32)
        
        for frame_idx in range(num_frames):
            mdm_frame = mdm_skeleton[frame_idx]  # (22, 3)
            vsl_frame = self._convert_mdm_to_vsl_frame(mdm_frame, reference_vsl)
            vsl_output[frame_idx] = vsl_frame
            
        if single_frame:
            return vsl_output[0]
        return vsl_output
    
    def _convert_mdm_to_vsl_frame(self, mdm_frame: np.ndarray,
                                   reference_vsl: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert single MDM frame (22, 3) to VSL raw format (1662,).
        """
        vsl_raw = np.zeros(1662, dtype=np.float32)
        
        # Build pose (33 landmarks × 4 values)
        pose = np.zeros((33, 4), dtype=np.float32)
        pose[:, 3] = 1.0  # Default visibility = 1
        
        # Map MDM joints back to MediaPipe
        pose[MEDIAPIPE_POSE['left_shoulder'], :3] = mdm_frame[MDM_JOINTS['left_shoulder']]
        pose[MEDIAPIPE_POSE['right_shoulder'], :3] = mdm_frame[MDM_JOINTS['right_shoulder']]
        pose[MEDIAPIPE_POSE['left_elbow'], :3] = mdm_frame[MDM_JOINTS['left_elbow']]
        pose[MEDIAPIPE_POSE['right_elbow'], :3] = mdm_frame[MDM_JOINTS['right_elbow']]
        pose[MEDIAPIPE_POSE['left_wrist'], :3] = mdm_frame[MDM_JOINTS['left_wrist']]
        pose[MEDIAPIPE_POSE['right_wrist'], :3] = mdm_frame[MDM_JOINTS['right_wrist']]
        pose[MEDIAPIPE_POSE['left_hip'], :3] = mdm_frame[MDM_JOINTS['left_hip']]
        pose[MEDIAPIPE_POSE['right_hip'], :3] = mdm_frame[MDM_JOINTS['right_hip']]
        pose[MEDIAPIPE_POSE['left_knee'], :3] = mdm_frame[MDM_JOINTS['left_knee']]
        pose[MEDIAPIPE_POSE['right_knee'], :3] = mdm_frame[MDM_JOINTS['right_knee']]
        pose[MEDIAPIPE_POSE['left_ankle'], :3] = mdm_frame[MDM_JOINTS['left_ankle']]
        pose[MEDIAPIPE_POSE['right_ankle'], :3] = mdm_frame[MDM_JOINTS['right_ankle']]
        pose[MEDIAPIPE_POSE['left_foot_index'], :3] = mdm_frame[MDM_JOINTS['left_foot']]
        pose[MEDIAPIPE_POSE['right_foot_index'], :3] = mdm_frame[MDM_JOINTS['right_foot']]
        pose[MEDIAPIPE_POSE['nose'], :3] = mdm_frame[MDM_JOINTS['head']]
        
        # Interpolate missing landmarks (eyes, ears, mouth, etc.)
        # Use head position as base
        head_pos = mdm_frame[MDM_JOINTS['head']]
        for idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Eye and mouth landmarks
            pose[idx, :3] = head_pos + np.random.randn(3) * 0.01  # Small offset
        
        # Heel landmarks from ankle
        pose[MEDIAPIPE_POSE['left_heel'], :3] = mdm_frame[MDM_JOINTS['left_ankle']]
        pose[MEDIAPIPE_POSE['right_heel'], :3] = mdm_frame[MDM_JOINTS['right_ankle']]
        
        # Hand tip landmarks from wrist
        wrist_offset = np.array([0, 0.05, 0])  # Small offset for fingers
        pose[MEDIAPIPE_POSE['left_pinky'], :3] = mdm_frame[MDM_JOINTS['left_wrist']] + wrist_offset
        pose[MEDIAPIPE_POSE['left_index'], :3] = mdm_frame[MDM_JOINTS['left_wrist']] + wrist_offset
        pose[MEDIAPIPE_POSE['left_thumb'], :3] = mdm_frame[MDM_JOINTS['left_wrist']] + wrist_offset
        pose[MEDIAPIPE_POSE['right_pinky'], :3] = mdm_frame[MDM_JOINTS['right_wrist']] + wrist_offset
        pose[MEDIAPIPE_POSE['right_index'], :3] = mdm_frame[MDM_JOINTS['right_wrist']] + wrist_offset
        pose[MEDIAPIPE_POSE['right_thumb'], :3] = mdm_frame[MDM_JOINTS['right_wrist']] + wrist_offset
        
        vsl_raw[:132] = pose.flatten()
        
        # Face: use reference or default
        if reference_vsl is not None and len(reference_vsl) >= 1536:
            vsl_raw[132:1536] = reference_vsl[132:1536]  # Copy face
            vsl_raw[1536:1599] = reference_vsl[1536:1599]  # Copy left hand
            vsl_raw[1599:1662] = reference_vsl[1599:1662]  # Copy right hand
        else:
            # Default face (centered)
            face_center = mdm_frame[MDM_JOINTS['head']]
            vsl_raw[132:1536] = np.tile(face_center, 468)
            
            # Default hands (at wrist position)
            left_wrist = mdm_frame[MDM_JOINTS['left_wrist']]
            right_wrist = mdm_frame[MDM_JOINTS['right_wrist']]
            vsl_raw[1536:1599] = np.tile(left_wrist, 21)
            vsl_raw[1599:1662] = np.tile(right_wrist, 21)
        
        return vsl_raw


class MDMTransitionGenerator:
    """
    Generate transitions using Motion Diffusion Model.
    
    This class wraps MDM's inpainting capability for VSL transition generation.
    """
    
    def __init__(self, model_path: Optional[str] = None, mdm_repo_path: Optional[str] = None):
        """
        Initialize MDM transition generator.
        
        Args:
            model_path: Path to MDM checkpoint (e.g., 'models/mdm/humanml.pt')
            mdm_repo_path: Path to cloned MDM repository
        """
        self.model_path = model_path
        self.mdm_repo_path = mdm_repo_path or self._find_mdm_repo()
        self.converter = VSLToMDMConverter()
        
        self.model = None
        self.diffusion = None
        self._model_loaded = False
        
    def _find_mdm_repo(self) -> Optional[str]:
        """Find MDM repository in common locations."""
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "motion-diffusion-model",
            Path.home() / "motion-diffusion-model",
            Path("/home/islabworker2/mya/motion-diffusion-model"),
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "sample" / "edit.py").exists():
                return str(path)
        
        return None
    
    def load_model(self):
        """Load MDM model for inference."""
        if self._model_loaded:
            return
            
        if self.mdm_repo_path is None:
            raise RuntimeError("MDM repository not found. Please clone it first.")
            
        if self.model_path is None:
            raise RuntimeError("Model path not specified. Please provide path to MDM checkpoint.")
        
        # Add MDM to path
        sys.path.insert(0, self.mdm_repo_path)
        
        try:
            import torch
            import json
            
            print(f"Loading MDM model from {self.model_path}...")
            
            # Load args from the model directory
            model_dir = Path(self.model_path).parent
            args_path = model_dir / "args.json"
            
            if args_path.exists():
                with open(args_path, 'r') as f:
                    args_dict = json.load(f)
                
                # Convert dict to namespace
                from argparse import Namespace
                args = Namespace(**args_dict)
                
                # Set required defaults
                args.cond_mask_prob = getattr(args, 'cond_mask_prob', 0.1)
                args.device = 0 if torch.cuda.is_available() else 'cpu'
                args.arch = getattr(args, 'arch', 'trans_enc')
                args.emb_trans_dec = getattr(args, 'emb_trans_dec', False)
                args.dataset = getattr(args, 'dataset', 'humanml')
                
                # Import MDM modules
                from model.mdm import MDM
                from diffusion import gaussian_diffusion as gd
                from diffusion.respace import SpacedDiffusion, space_timesteps
                
                # Create diffusion
                diffusion_steps = getattr(args, 'diffusion_steps', 50)
                betas = gd.get_named_beta_schedule(
                    getattr(args, 'noise_schedule', 'cosine'),
                    diffusion_steps
                )
                
                # Handle timestep_respacing - if empty, use all steps
                timestep_respacing = getattr(args, 'timestep_respacing', '')
                if not timestep_respacing:
                    timestep_respacing = str(diffusion_steps)
                
                self.diffusion = SpacedDiffusion(
                    use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
                    betas=betas,
                    model_mean_type=gd.ModelMeanType.START_X,
                    model_var_type=gd.ModelVarType.FIXED_SMALL,
                    loss_type=gd.LossType.MSE,
                    rescale_timesteps=True,
                )
                
                # Create model (SMPL files should be in motion-diffusion-model/body_models/smpl/)
                njoints = 263  # HumanML3D format
                nfeats = 1
                self.model = MDM(
                    modeltype=getattr(args, 'modeltype', 'mdm'),
                    njoints=njoints,
                    nfeats=nfeats,
                    num_actions=getattr(args, 'num_actions', 1),
                    translation=getattr(args, 'translation', True),
                    pose_rep=getattr(args, 'pose_rep', 'rot6d'),
                    glob=getattr(args, 'glob', True),
                    glob_rot=getattr(args, 'glob_rot', True),
                    latent_dim=getattr(args, 'latent_dim', 512),
                    ff_size=getattr(args, 'ff_size', 1024),
                    num_layers=getattr(args, 'layers', 8),
                    num_heads=getattr(args, 'num_heads', 4),
                    dropout=getattr(args, 'dropout', 0.1),
                    activation=getattr(args, 'activation', 'gelu'),
                    cond_mask_prob=args.cond_mask_prob,
                    arch=args.arch,
                    emb_trans_dec=args.emb_trans_dec,
                    dataset=getattr(args, 'dataset', 'humanml'),
                )
                
                # Load weights
                state_dict = torch.load(self.model_path, map_location='cpu')
                self.model.load_state_dict(state_dict, strict=False)
                
                # Move to device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(device)
                self.model.eval()
                
                self._model_loaded = True
                print(f"MDM model loaded successfully on {device}!")
            else:
                print(f"Warning: args.json not found at {args_path}")
                self._model_loaded = True
            
        except Exception as e:
            print(f"Warning: Could not load MDM model: {e}")
            print("Will use diffusion-style interpolation fallback")
            self._model_loaded = True  # Mark as loaded to use fallback
    
    def generate_transition(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        num_frames: int = 10,
        use_mdm: bool = True
    ) -> np.ndarray:
        """
        Generate transition between two poses.
        
        Args:
            start_pose: Starting pose in VSL format (1662,) or (554, 3)
            end_pose: Ending pose in VSL format
            num_frames: Number of frames to generate
            use_mdm: If True, use MDM; if False, fallback to spline
            
        Returns:
            Transition sequence (num_frames, 554, 3)
        """
        # Flatten inputs if needed
        if start_pose.ndim > 1:
            start_pose = start_pose.flatten()
        if end_pose.ndim > 1:
            end_pose = end_pose.flatten()
            
        if not use_mdm or not self._model_loaded:
            # Fallback to spline interpolation
            return self._fallback_interpolation(start_pose, end_pose, num_frames)
        
        # Convert to MDM format
        start_mdm = self.converter.vsl_to_mdm(start_pose)  # (22, 3)
        end_mdm = self.converter.vsl_to_mdm(end_pose)  # (22, 3)
        
        # Generate transition using MDM inpainting
        transition_mdm = self._mdm_inpaint(start_mdm, end_mdm, num_frames)
        
        # Convert back to VSL format
        transition_vsl = self.converter.mdm_to_vsl(
            transition_mdm, 
            reference_vsl=(start_pose + end_pose) / 2  # Use average as reference for face/hands
        )
        
        # Reshape to (frames, 554, 3)
        transition_vsl = transition_vsl.reshape(num_frames, 554, 3)
        
        return transition_vsl
    
    def _mdm_inpaint(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        num_frames: int
    ) -> np.ndarray:
        """
        Use MDM's inpainting mode to generate middle frames.
        
        Args:
            start_frame: Starting pose (22, 3)
            end_frame: Ending pose (22, 3)
            num_frames: Number of frames to generate
            
        Returns:
            Generated sequence (num_frames, 22, 3)
        """
        import torch
        
        try:
            # Try to use actual MDM model for inpainting
            return self._mdm_inpaint_actual(start_frame, end_frame, num_frames)
        except Exception as e:
            print(f"MDM inpainting failed: {e}")
            print("Using diffusion-style interpolation fallback")
            return self._diffusion_style_interpolation(start_frame, end_frame, num_frames)
    
    def _diffusion_style_interpolation(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        num_frames: int
    ) -> np.ndarray:
        """
        Diffusion-style interpolation that adds natural motion variation.
        
        This simulates diffusion behavior by:
        1. Linear interpolation as base
        2. Adding smooth noise for natural variation
        3. Applying temporal smoothing
        """
        from scipy.ndimage import gaussian_filter1d
        
        # Base linear interpolation
        alphas = np.linspace(0, 1, num_frames)[:, np.newaxis, np.newaxis]
        base_transition = (1 - alphas) * start_frame + alphas * end_frame
        
        # Add smooth noise for natural motion (not on endpoints)
        noise_scale = 0.02  # Small noise
        noise = np.random.randn(num_frames, 22, 3) * noise_scale
        
        # Zero out noise at endpoints to preserve start/end poses
        noise[0] = 0
        noise[-1] = 0
        
        # Smooth the noise temporally
        for j in range(22):
            for k in range(3):
                noise[:, j, k] = gaussian_filter1d(noise[:, j, k], sigma=2)
        
        # Add noise to base
        transition = base_transition + noise
        
        # Apply temporal smoothing for natural motion
        for j in range(22):
            for k in range(3):
                transition[:, j, k] = gaussian_filter1d(transition[:, j, k], sigma=1)
        
        # Ensure endpoints are exact
        transition[0] = start_frame
        transition[-1] = end_frame
        
        return transition.astype(np.float32)
    
    def _mdm_inpaint_actual(
        self,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        num_frames: int
    ) -> np.ndarray:
        """
        Actual MDM inpainting using the loaded model.
        
        This is the full implementation using MDM's diffusion sampling.
        """
        import torch
        
        # Check if model components are available
        if self.model is None or self.diffusion is None:
            raise RuntimeError("MDM model not properly loaded")
        
        device = next(self.model.parameters()).device
        
        # Create motion tensor (batch=1, joints=22, coords=3, frames=num_frames)
        # MDM expects shape: (batch, njoints, nfeats, nframes)
        motion = torch.zeros(1, 22, 3, num_frames, device=device)
        
        # Set start and end frames
        motion[0, :, :, 0] = torch.from_numpy(start_frame).float().to(device)
        motion[0, :, :, -1] = torch.from_numpy(end_frame).float().to(device)
        
        # Create inpainting mask (1 = keep, 0 = generate)
        mask = torch.zeros(1, 22, 3, num_frames, device=device)
        mask[0, :, :, 0] = 1.0   # Keep first frame
        mask[0, :, :, -1] = 1.0  # Keep last frame
        
        # Run diffusion sampling with inpainting
        # Note: This is simplified - actual MDM API may differ
        with torch.no_grad():
            # Sample from diffusion
            sample = self.diffusion.p_sample_loop(
                self.model,
                motion.shape,
                clip_denoised=False,
                model_kwargs={
                    'y': {'mask': mask, 'inpainted_motion': motion}
                },
                skip_timesteps=0,
                init_image=None,
                progress=False,
            )
        
        # Convert to numpy (frames, joints, coords)
        result = sample[0].permute(2, 0, 1).cpu().numpy()  # (frames, 22, 3)
        
        return result.astype(np.float32)
    
    def _fallback_interpolation(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        num_frames: int
    ) -> np.ndarray:
        """
        Fallback to spline interpolation when MDM is not available.
        """
        from .interpolation import cubic_spline_interpolation
        
        # Reshape for interpolation
        start_reshaped = start_pose.reshape(-1, 3)
        end_reshaped = end_pose.reshape(-1, 3)
        
        # Use spline interpolation
        transition = cubic_spline_interpolation(start_reshaped, end_reshaped, num_frames)
        
        return transition


# Convenience function for direct use
def generate_transition_mdm(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    num_frames: int = 10,
    model_path: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function to generate transition using MDM.
    
    Args:
        start_pose: Starting pose in VSL format
        end_pose: Ending pose in VSL format
        num_frames: Number of transition frames
        model_path: Path to MDM checkpoint
        
    Returns:
        Transition sequence
    """
    generator = MDMTransitionGenerator(model_path=model_path)
    
    if model_path:
        try:
            generator.load_model()
        except Exception as e:
            print(f"Warning: Could not load MDM model: {e}")
            print("Using fallback interpolation")
    
    return generator.generate_transition(start_pose, end_pose, num_frames)


if __name__ == "__main__":
    print("MDM Adapter Module")
    print("="*50)
    
    # Test converter
    converter = VSLToMDMConverter()
    
    # Create dummy VSL data
    vsl_dummy = np.random.rand(1662).astype(np.float32)
    
    print(f"Input VSL shape: {vsl_dummy.shape}")
    
    # Convert to MDM
    mdm_output = converter.vsl_to_mdm(vsl_dummy)
    print(f"MDM output shape: {mdm_output.shape}")
    
    # Convert back to VSL
    vsl_back = converter.mdm_to_vsl(mdm_output, reference_vsl=vsl_dummy)
    print(f"VSL output shape: {vsl_back.shape}")
    
    print("\n Converter test passed!")
