"""
Trajectory Sampler Module
- Diffusion-inspired stochastic trajectory sampling
- NOT a pixel-based diffusion model, but trajectory-based
- Generates K plausible future trajectories from latent state
- Uses small denoising network
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FutureTrajectory:
    """Single future trajectory sample."""
    track_id: int
    sample_idx: int
    future_frames: int
    trajectory: np.ndarray  # (future_frames, 2) - (x, y) positions
    velocities: np.ndarray  # (future_frames, 2) - velocity vectors
    log_probability: float = 0.0  # log prob of this sample


class DenoisingNetwork(nn.Module):
    """Small neural network for denoising noisy trajectory samples."""
    
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, 
                 output_dim: int = 2, num_layers: int = 2):
        """
        Initialize denoising network.
        
        Args:
            latent_dim: Dimension of input latent vector
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (usually 2 for dx, dy)
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        layers = []
        in_dim = latent_dim + output_dim  # Concatenate with noisy sample
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, latent: torch.Tensor, noisy_sample: torch.Tensor) -> torch.Tensor:
        """
        Denoise a noisy trajectory sample.
        
        Args:
            latent: Latent motion state (batch_size, latent_dim)
            noisy_sample: Noisy trajectory sample (batch_size, output_dim)
            
        Returns:
            torch.Tensor: Denoised sample (batch_size, output_dim)
        """
        x = torch.cat([latent, noisy_sample], dim=-1)
        denoised = self.network(x)
        return denoised


class TrajectoryDiffusionSampler:
    """
    Diffusion-inspired trajectory sampler.
    
    Process:
    1. Start with random noise
    2. Iteratively add latent motion information
    3. Denoise using small network
    4. Produce plausible future trajectories
    """
    
    def __init__(self, latent_dim: int = 32, device: str = 'cpu',
                 num_denoising_steps: int = 10, future_frames: int = 30):
        """
        Initialize trajectory sampler.
        
        Args:
            latent_dim: Latent motion state dimension
            device: Device to run on
            num_denoising_steps: Number of denoising iterations
            future_frames: Number of frames to predict into future
        """
        self.latent_dim = latent_dim
        self.device = device
        self.num_denoising_steps = num_denoising_steps
        self.future_frames = future_frames
        
        # Initialize denoising network
        self.denoiser = DenoisingNetwork(
            latent_dim=latent_dim,
            hidden_dim=64,
            output_dim=2,  # (dx, dy)
            num_layers=2
        ).to(device)
        
        self.denoiser.eval()
        
        logger.info(f"TrajectoryDiffusionSampler initialized with {num_denoising_steps} steps")
    
    def _create_noise_schedule(self, num_steps: int) -> np.ndarray:
        """
        Create noise schedule for diffusion process.
        Standard linear schedule from 1.0 to 0.0.
        
        Args:
            num_steps: Number of diffusion steps
            
        Returns:
            np.ndarray: Noise levels [1.0, ..., 0.0]
        """
        return np.linspace(1.0, 0.0, num_steps)
    
    def _sample_noisy_trajectory(self, current_pos: np.ndarray, 
                                 noise_level: float, fps: float = 10.0) -> np.ndarray:
        """
        Sample noisy trajectory increments.
        
        Args:
            current_pos: Current position (2,)
            noise_level: Current noise level [0, 1]
            fps: Frames per second for velocity scaling
            
        Returns:
            np.ndarray: Trajectory increments (future_frames, 2)
        """
        # Generate random motion with noise
        noise_scale = noise_level * 5.0  # Scale noise to reasonable motion range
        
        # Random walk with exponential decay
        trajectory_increments = []
        current_vel = np.random.randn(2) * 2.0  # Initial random velocity
        
        for _ in range(self.future_frames):
            # Add random perturbation
            random_acc = np.random.randn(2) * noise_scale
            current_vel = current_vel * 0.95 + random_acc  # Velocity decay
            
            # Compute increment
            increment = current_vel / fps
            trajectory_increments.append(increment)
        
        return np.array(trajectory_increments)
    
    def _trajectory_to_positions(self, start_pos: np.ndarray, 
                                increments: np.ndarray) -> np.ndarray:
        """
        Convert incremental trajectory to absolute positions.
        
        Args:
            start_pos: Starting position (2,)
            increments: Velocity increments (future_frames, 2)
            
        Returns:
            np.ndarray: Absolute positions (future_frames, 2)
        """
        positions = [start_pos]
        for inc in increments:
            positions.append(positions[-1] + inc)
        return np.array(positions[1:])  # Exclude start position
    
    def sample_trajectories(self, latent_vector: np.ndarray,
                           start_pos: np.ndarray,
                           num_samples: int = 5) -> List[FutureTrajectory]:
        """
        Generate K plausible future trajectories from latent state.
        
        Args:
            latent_vector: Latent motion state (latent_dim,)
            start_pos: Starting position (2,)
            num_samples: Number of trajectory samples
            
        Returns:
            list: FutureTrajectory objects
        """
        samples = []
        
        # Create noise schedule
        noise_schedule = self._create_noise_schedule(self.num_denoising_steps)
        
        # Convert latent to tensor
        latent_tensor = torch.from_numpy(latent_vector.astype(np.float32)).unsqueeze(0).to(self.device)
        
        for sample_idx in range(num_samples):
            trajectory_increments = []
            
            # Iterative diffusion process: gradually add information
            for step, noise_level in enumerate(noise_schedule):
                # Sample noisy trajectory
                noisy_traj = self._sample_noisy_trajectory(start_pos, noise_level)
                
                # Denoise using network
                if step % max(1, self.num_denoising_steps // 3) == 0:
                    # Every few steps, apply denoising
                    for frame_idx in range(self.future_frames):
                        noisy_sample = torch.from_numpy(
                            noisy_traj[frame_idx:frame_idx+1].astype(np.float32)
                        ).to(self.device)
                        
                        with torch.no_grad():
                            denoised = self.denoiser(latent_tensor, noisy_sample)
                        
                        noisy_traj[frame_idx] = denoised.squeeze(0).cpu().numpy()
                
                trajectory_increments = noisy_traj
            
            # Convert to absolute positions
            positions = self._trajectory_to_positions(start_pos, trajectory_increments)
            
            # Compute velocities
            velocities = np.zeros_like(positions)
            velocities[0] = positions[0] - start_pos
            velocities[1:] = np.diff(positions, axis=0)
            
            # Create trajectory object
            future_traj = FutureTrajectory(
                track_id=-1,  # Will be set by caller
                sample_idx=sample_idx,
                future_frames=self.future_frames,
                trajectory=positions,
                velocities=velocities,
                log_probability=0.0
            )
            
            samples.append(future_traj)
        
        return samples
    
    def batch_sample_trajectories(self, latent_vectors: Dict[int, np.ndarray],
                                 current_positions: Dict[int, np.ndarray],
                                 num_samples: int = 5) -> Dict[int, List[FutureTrajectory]]:
        """
        Generate trajectory samples for multiple agents.
        
        Args:
            latent_vectors: Dict of track_id -> latent_vector
            current_positions: Dict of track_id -> position (2,)
            num_samples: Number of samples per agent
            
        Returns:
            dict: track_id -> list of FutureTrajectory
        """
        all_samples = {}
        
        for track_id in latent_vectors.keys():
            if track_id not in current_positions:
                continue
            
            latent = latent_vectors[track_id]
            start_pos = current_positions[track_id]
            
            samples = self.sample_trajectories(latent, start_pos, num_samples)
            
            # Set track_id
            for sample in samples:
                sample.track_id = track_id
            
            all_samples[track_id] = samples
        
        logger.info(f"Sampled {num_samples} trajectories for {len(all_samples)} agents")
        return all_samples
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained denoising network."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.denoiser.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained denoiser from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load denoiser: {e}")
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save denoising network checkpoint."""
        torch.save(self.denoiser.state_dict(), checkpoint_path)
        logger.info(f"Saved denoiser checkpoint to {checkpoint_path}")


class SimpleTrajectoryPredictor:
    """
    Simplified predictor: projects motion linearly with stochastic perturbations.
    Useful for baseline or when full diffusion sampling is not needed.
    """
    
    def __init__(self, future_frames: int = 30, device: str = 'cpu'):
        """
        Initialize simple predictor.
        
        Args:
            future_frames: Number of frames to predict
            device: Device (not really used here, for compatibility)
        """
        self.future_frames = future_frames
        self.device = device
    
    def predict_trajectory(self, current_pos: np.ndarray,
                          current_vel: np.ndarray,
                          num_samples: int = 5,
                          vel_noise_scale: float = 0.5) -> List[FutureTrajectory]:
        """
        Simple linear motion prediction with noise.
        
        Args:
            current_pos: Starting position (2,)
            current_vel: Current velocity (2,)
            num_samples: Number of stochastic samples
            vel_noise_scale: Scale of velocity noise
            
        Returns:
            list: FutureTrajectory objects
        """
        samples = []
        
        for sample_idx in range(num_samples):
            # Add noise to velocity
            noise = np.random.randn(2) * vel_noise_scale
            vel_sample = current_vel + noise
            
            # Exponential decay in velocity
            positions = []
            pos = current_pos.copy()
            vel = vel_sample.copy()
            
            for _ in range(self.future_frames):
                pos = pos + vel
                positions.append(pos.copy())
                vel = vel * 0.98  # Decay velocity
            
            positions = np.array(positions)
            
            # Compute velocities
            velocities = np.zeros_like(positions)
            velocities[0] = positions[0] - current_pos
            velocities[1:] = np.diff(positions, axis=0)
            
            sample = FutureTrajectory(
                track_id=-1,
                sample_idx=sample_idx,
                future_frames=self.future_frames,
                trajectory=positions,
                velocities=velocities
            )
            
            samples.append(sample)
        
        return samples
