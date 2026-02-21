from __future__ import annotations

"""
State Encoder Module
- Encodes observed trajectory history using LSTM
- Produces latent motion representation for each agent
- Sliding window of T_obs frames
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LatentMotionState:
    """Encoded latent motion state for an agent."""
    track_id: int
    frame_idx: int
    latent_vector: np.ndarray  # (latent_dim,)
    input_sequence_length: int
    confidence: float = 1.0  # How much context was available


class TemporalMotionEncoder(nn.Module):
    """LSTM-based temporal encoder for motion sequences."""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, 
                 num_layers: int = 2, latent_dim: int = 32, dropout: float = 0.2):
        """
        Initialize motion encoder.
        
        Args:
            input_dim: Dimension of input features per frame (e.g., cx, cy, vx, vy, ax, ay)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            latent_dim: Dimension of output latent vector
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Project LSTM output to latent space
        self.fc_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode motion sequence to latent vector.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Latent vectors of shape (batch_size, latent_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state
        final_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        # Project to latent space
        latent = self.fc_latent(final_hidden)  # (batch_size, latent_dim)
        
        return latent


class StateEncoder:
    """Encodes agent trajectories to latent motion states."""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64,
                 latent_dim: int = 32, device: str = 'cpu',
                 window_size: int = 20):
        """
        Initialize state encoder.
        
        Args:
            input_dim: Dimension of input features per frame
            hidden_dim: LSTM hidden dimension
            latent_dim: Dimension of latent vector
            device: Device to run on ('cpu' or 'cuda:0', etc.)
            window_size: Number of frames to look back (sliding window)
        """
        self.device = device
        self.window_size = window_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initialize encoder
        self.encoder = TemporalMotionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        ).to(device)
        
        self.encoder.eval()  # Inference mode
        logger.info(f"StateEncoder initialized on device {device} with latent_dim={latent_dim}")
    
    def _extract_motion_features(self, states: List['FrameState']) -> np.ndarray:
        """
        Extract motion features from a list of states.
        
        Args:
            states: List of FrameState objects
            
        Returns:
            np.ndarray: Features of shape (seq_len, input_dim)
        """
        features = []
        for state in states:
            feat = np.array([
                state.center_x,
                state.center_y,
                state.velocity_x,
                state.velocity_y,
                state.acceleration_x,
                state.acceleration_y,
            ], dtype=np.float32)
            features.append(feat)
        
        return np.stack(features, axis=0)  # (seq_len, input_dim)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features for stable encoding.
        Simple Z-score normalization per feature.
        
        Args:
            features: (seq_len, input_dim)
            
        Returns:
            np.ndarray: Normalized features
        """
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-6
        return (features - mean) / std
    
    def encode_trajectory_at_frame(self, trajectory: 'AgentTrajectory',
                                  frame_idx: int) -> Optional[LatentMotionState]:
        """
        Encode trajectory state at a specific frame using sliding window.
        
        Args:
            trajectory: AgentTrajectory object
            frame_idx: Frame index to encode at
            
        Returns:
            LatentMotionState or None if not enough history
        """
        # Get states in sliding window before frame_idx
        window_states = []
        for state in trajectory.states:
            if state.frame_idx <= frame_idx:
                window_states.append(state)
        
        if len(window_states) < 1:
            return None
        
        # Limit to window size
        if len(window_states) > self.window_size:
            window_states = window_states[-self.window_size:]
        
        # Extract and normalize features
        features = self._extract_motion_features(window_states)
        norm_features = self._normalize_features(features)
        
        # Convert to tensor
        x = torch.from_numpy(norm_features).unsqueeze(0).to(self.device)  # (1, seq_len, input_dim)
        
        # Encode
        with torch.no_grad():
            latent = self.encoder(x)
        
        latent_np = latent.squeeze(0).cpu().numpy()
        
        # Compute confidence based on available history
        # More history = higher confidence
        confidence = min(1.0, len(window_states) / self.window_size)
        
        return LatentMotionState(
            track_id=trajectory.track_id,
            frame_idx=frame_idx,
            latent_vector=latent_np,
            input_sequence_length=len(window_states),
            confidence=confidence
        )
    
    def encode_all_trajectories(self, trajectories: Dict[int, 'AgentTrajectory']) -> Dict[int, Dict[int, LatentMotionState]]:
        """
        Encode all trajectories at all time steps.
        
        Args:
            trajectories: Dict of track_id -> AgentTrajectory
            
        Returns:
            dict: track_id -> frame_idx -> LatentMotionState
        """
        all_encodings = {}
        
        for track_id, trajectory in trajectories.items():
            encodings = {}
            
            # Encode at each frame
            for state in trajectory.states:
                latent_state = self.encode_trajectory_at_frame(trajectory, state.frame_idx)
                if latent_state is not None:
                    encodings[state.frame_idx] = latent_state
            
            if encodings:
                all_encodings[track_id] = encodings
        
        logger.info(f"Encoded {len(all_encodings)} trajectories to latent space")
        return all_encodings
    
    def get_latest_encoding(self, trajectory: 'AgentTrajectory') -> Optional[LatentMotionState]:
        """Get encoding at the last frame of trajectory."""
        if not trajectory.states:
            return None
        
        last_frame = trajectory.states[-1].frame_idx
        return self.encode_trajectory_at_frame(trajectory, last_frame)
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained encoder weights."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained encoder from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load pretrained encoder: {e}")
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save encoder checkpoint."""
        torch.save(self.encoder.state_dict(), checkpoint_path)
        logger.info(f"Saved encoder checkpoint to {checkpoint_path}")


class PretrainedMotionEncoder:
    """
    Simple pretrained motion encoder (no training needed for research prototype).
    Uses a frozen LSTM to extract motion patterns.
    """
    
    def __init__(self, latent_dim: int = 32, device: str = 'cpu'):
        """Initialize pretrained encoder with random initialization."""
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = StateEncoder(latent_dim=latent_dim, device=device)
        
        # For research prototype, freeze weights (could load from pretrained)
        for param in self.encoder.encoder.parameters():
            param.requires_grad = False
        
        logger.info("Pretrained motion encoder ready (frozen)")
