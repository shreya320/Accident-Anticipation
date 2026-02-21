"""
Trajectory Extraction Module
- Extract per-frame center coordinates, velocity, acceleration, heading
- Maintain temporal sequences for each tracked agent
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FrameState:
    """State of an agent at a single frame."""
    frame_idx: int
    timestamp: float
    center_x: float
    center_y: float
    velocity_x: float = 0.0  # pixels/sec
    velocity_y: float = 0.0
    speed: float = 0.0  # magnitude of velocity
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    acceleration_mag: float = 0.0
    heading_angle: float = 0.0  # radians, 0=right, pi/2=down
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y])
    
    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.velocity_x, self.velocity_y])
    
    @property
    def acceleration(self) -> np.ndarray:
        return np.array([self.acceleration_x, self.acceleration_y])


@dataclass
class AgentTrajectory:
    """Complete trajectory for a single tracked agent."""
    track_id: int
    class_name: str
    class_id: int
    states: List[FrameState] = field(default_factory=list)
    
    def add_state(self, state: FrameState):
        """Add a frame state to trajectory."""
        self.states.append(state)
    
    def get_positions(self) -> np.ndarray:
        """Get all positions as (N, 2) array."""
        return np.array([[s.center_x, s.center_y] for s in self.states])
    
    def get_velocities(self) -> np.ndarray:
        """Get all velocities as (N, 2) array."""
        return np.array([[s.velocity_x, s.velocity_y] for s in self.states])
    
    def get_accelerations(self) -> np.ndarray:
        """Get all accelerations as (N, 2) array."""
        return np.array([[s.acceleration_x, s.acceleration_y] for s in self.states])
    
    def get_speeds(self) -> np.ndarray:
        """Get all speed magnitudes as (N,) array."""
        return np.array([s.speed for s in self.states])
    
    def get_headings(self) -> np.ndarray:
        """Get all heading angles as (N,) array."""
        return np.array([s.heading_angle for s in self.states])
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get min and max timestamps."""
        if not self.states:
            return (0, 0)
        return (self.states[0].timestamp, self.states[-1].timestamp)
    
    def get_frame_range(self) -> Tuple[int, int]:
        """Get min and max frame indices."""
        if not self.states:
            return (0, 0)
        return (self.states[0].frame_idx, self.states[-1].frame_idx)
    
    def duration_sec(self) -> float:
        """Duration in seconds."""
        t_min, t_max = self.get_time_range()
        return t_max - t_min
    
    def duration_frames(self) -> int:
        """Duration in frames."""
        f_min, f_max = self.get_frame_range()
        return f_max - f_min + 1
    
    def get_state_at_frame(self, frame_idx: int) -> Optional[FrameState]:
        """Get state at specific frame."""
        for state in self.states:
            if state.frame_idx == frame_idx:
                return state
        return None
    
    def get_states_in_frame_range(self, start_frame: int, end_frame: int) -> List[FrameState]:
        """Get states within frame range."""
        return [s for s in self.states if start_frame <= s.frame_idx <= end_frame]
    
    def smooth_velocities(self, window_size: int = 3):
        """Smooth velocities using moving average."""
        if len(self.states) < window_size:
            return
        
        # Extract velocities
        vels = np.array([[s.velocity_x, s.velocity_y] for s in self.states])
        
        # Apply moving average
        from scipy.ndimage import uniform_filter1d
        smooth_vels = np.zeros_like(vels)
        smooth_vels[:, 0] = uniform_filter1d(vels[:, 0], size=window_size, mode='nearest')
        smooth_vels[:, 1] = uniform_filter1d(vels[:, 1], size=window_size, mode='nearest')
        
        # Update states
        for i, state in enumerate(self.states):
            state.velocity_x = smooth_vels[i, 0]
            state.velocity_y = smooth_vels[i, 1]
            state.speed = np.linalg.norm([state.velocity_x, state.velocity_y])


class TrajectoryExtractor:
    """Extract trajectories from detection tracks."""
    
    def __init__(self, pixels_per_meter: float = 1.0):
        """
        Initialize trajectory extractor.
        
        Args:
            pixels_per_meter: Conversion factor for velocity/acceleration in m/s
                             (use 1.0 if keeping in pixel units)
        """
        self.pixels_per_meter = pixels_per_meter
        self.trajectories: Dict[int, AgentTrajectory] = {}
    
    def _calculate_heading(self, vx: float, vy: float) -> float:
        """
        Calculate heading angle from velocity.
        
        Args:
            vx, vy: Velocity components
            
        Returns:
            float: Heading in radians (0=right, pi/2=down, -pi=left, -pi/2=up)
        """
        return np.arctan2(vy, vx)
    
    def extract_trajectories(self, tracks_dict: Dict[int, 'Track']) -> Dict[int, AgentTrajectory]:
        """
        Extract trajectories from detection tracks.
        
        Args:
            tracks_dict: Dictionary of Track objects from DetectionTrackingPipeline
            
        Returns:
            dict: track_id -> AgentTrajectory
        """
        self.trajectories.clear()
        
        for track_id, track in tracks_dict.items():
            trajectory = AgentTrajectory(
                track_id=track_id,
                class_name=track.class_name,
                class_id=track.class_id
            )
            
            # Get center trajectory from detection track
            center_traj = track.get_center_trajectory()
            
            # Compute motion features for each frame
            prev_state = None
            
            for i, (frame_idx, (cx, cy)) in enumerate(center_traj):
                # Get timestamp from corresponding detection
                det = track.get_detection_at_frame(frame_idx)
                timestamp = 0.0  # Will be set by caller
                
                # Calculate velocity (finite difference)
                if prev_state is not None:
                    dt = frame_idx - prev_state.frame_idx
                    if dt > 0:
                        vx = (cx - prev_state.center_x) / dt
                        vy = (cy - prev_state.center_y) / dt
                    else:
                        vx, vy = 0.0, 0.0
                else:
                    vx, vy = 0.0, 0.0
                
                speed = np.sqrt(vx**2 + vy**2)
                heading = self._calculate_heading(vx, vy)
                
                # Calculate acceleration (second-order finite difference)
                if prev_state is not None and prev_state.velocity_x is not None:
                    dt = frame_idx - prev_state.frame_idx
                    if dt > 0:
                        ax = (vx - prev_state.velocity_x) / dt
                        ay = (vy - prev_state.velocity_y) / dt
                    else:
                        ax, ay = 0.0, 0.0
                else:
                    ax, ay = 0.0, 0.0
                
                accel_mag = np.sqrt(ax**2 + ay**2)
                
                state = FrameState(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    center_x=cx,
                    center_y=cy,
                    velocity_x=vx,
                    velocity_y=vy,
                    speed=speed,
                    acceleration_x=ax,
                    acceleration_y=ay,
                    acceleration_mag=accel_mag,
                    heading_angle=heading
                )
                
                trajectory.add_state(state)
                prev_state = state
            
            self.trajectories[track_id] = trajectory
        
        logger.info(f"Extracted {len(self.trajectories)} trajectories")
        return self.trajectories
    
    def update_timestamps(self, frame_times: List[float]):
        """
        Update trajectory frame timestamps.
        
        Args:
            frame_times: List of timestamps for each frame index
        """
        for track_id, trajectory in self.trajectories.items():
            for state in trajectory.states:
                if state.frame_idx < len(frame_times):
                    state.timestamp = frame_times[state.frame_idx]
    
    def get_trajectory(self, track_id: int) -> Optional[AgentTrajectory]:
        """Get trajectory by track ID."""
        return self.trajectories.get(track_id)
    
    def get_all_trajectories(self) -> Dict[int, AgentTrajectory]:
        """Get all trajectories."""
        return self.trajectories.copy()
    
    def filter_trajectories(self, min_duration_frames: int = 5) -> Dict[int, AgentTrajectory]:
        """
        Get trajectories meeting minimum duration.
        
        Args:
            min_duration_frames: Minimum frames for trajectory
            
        Returns:
            dict: Filtered trajectories
        """
        return {tid: traj for tid, traj in self.trajectories.items()
                if traj.duration_frames() >= min_duration_frames}
    
    def smooth_all_trajectories(self, window_size: int = 3):
        """Smooth all trajectory velocities."""
        for trajectory in self.trajectories.values():
            trajectory.smooth_velocities(window_size)
        logger.info(f"Smoothed {len(self.trajectories)} trajectories with window={window_size}")
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics of all trajectories."""
        if not self.trajectories:
            return {}
        
        durations = [t.duration_frames() for t in self.trajectories.values()]
        speeds = [t.get_speeds().mean() for t in self.trajectories.values() if len(t.states) > 0]
        
        return {
            'num_trajectories': len(self.trajectories),
            'avg_duration_frames': np.mean(durations) if durations else 0,
            'min_duration_frames': min(durations) if durations else 0,
            'max_duration_frames': max(durations) if durations else 0,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
        }
