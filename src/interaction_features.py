"""
Interaction Features Module
- Compute distance to nearest agents
- Relative velocity vectors
- Time-to-collision (TTC) approximation
- Collision/intersection conflict indicators (heuristic)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class InteractionFeatures:
    """Interaction features for one agent at one timestep."""
    track_id: int
    frame_idx: int
    
    # Proximity
    distance_to_nearest: float = np.inf  # pixels
    nearest_agent_id: int = -1
    num_nearby_agents: int = 0  # within 300px, e.g.
    
    # Relative motion
    relative_velocity_x: float = 0.0
    relative_velocity_y: float = 0.0
    relative_velocity_mag: float = 0.0
    relative_heading: float = 0.0  # radians
    
    # Collision/conflict
    ttc_nearest: float = np.inf  # Time-to-collision in seconds
    is_on_collision_course: bool = False
    collision_indicator: float = 0.0  # [0, 1] soft indicator
    
    # Conflict heuristic
    intersection_risk: float = 0.0  # Simple intersection detection
    
    # Number of concurrent agents
    concurrent_agents: int = 0


class InteractionFeatureComputer:
    """Compute interaction features between agents."""
    
    def __init__(self, proximity_threshold: float = 300.0, 
                 ttc_threshold: float = 2.0,
                 collision_distance_threshold: float = 50.0):
        """
        Initialize interaction feature computer.
        
        Args:
            proximity_threshold: Distance for "nearby" agents (pixels)
            ttc_threshold: Threshold for collision course (seconds)
            collision_distance_threshold: Distance threshold for near-miss (pixels)
        """
        self.proximity_threshold = proximity_threshold
        self.ttc_threshold = ttc_threshold
        self.collision_distance_threshold = collision_distance_threshold
    
    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(p1 - p2)
    
    def _calculate_ttc(self, pos1: np.ndarray, vel1: np.ndarray,
                      pos2: np.ndarray, vel2: np.ndarray) -> float:
        """
        Calculate approximate time-to-collision.
        
        Simplified model: assumes constant velocity.
        Uses relative motion to estimate collision time.
        
        Args:
            pos1, vel1: Position and velocity of agent 1
            pos2, vel2: Position and velocity of agent 2
            
        Returns:
            float: TTC in seconds (inf if not on collision course)
        """
        # Relative position and velocity
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1
        
        # Distance
        distance = np.linalg.norm(rel_pos)
        
        # Relative speed component in direction of relative position
        if distance < 1e-6:
            return np.inf
        
        rel_vel_norm = rel_vel / (np.linalg.norm(rel_vel) + 1e-6)
        closing_speed = -np.dot(rel_pos, rel_vel_norm)
        
        # If not closing, TTC is infinite
        if closing_speed <= 0:
            return np.inf
        
        # TTC = distance / closing_speed
        ttc = distance / (closing_speed + 1e-6)
        
        return max(0, ttc)
    
    def _collision_indicator_soft(self, ttc: float, 
                                  distance: float) -> float:
        """
        Soft collision indicator combining TTC and distance.
        
        Args:
            ttc: Time-to-collision (seconds)
            distance: Current distance (pixels)
            
        Returns:
            float: Risk indicator in [0, 1]
        """
        # Normalize TTC component: lower TTC -> higher risk
        ttc_risk = 1.0 / (1.0 + np.exp(ttc - 1.5))  # Sigmoid centered at 1.5s
        
        # Normalize distance component: closer -> higher risk
        dist_risk = 1.0 / (1.0 + np.exp((distance - 50) / 20))  # Sigmoid centered at 50px
        
        # Combine: take average
        indicator = 0.6 * ttc_risk + 0.4 * dist_risk
        
        return float(np.clip(indicator, 0, 1))
    
    def _check_intersection_heuristic(self, pos1: np.ndarray, vel1: np.ndarray,
                                     pos2: np.ndarray, vel2: np.ndarray,
                                     horizon_frames: int = 10) -> float:
        """
        Simple heuristic for intersection/cross path.
        Projects both agents forward and checks for spatial intersection.
        
        Args:
            pos1, vel1: Agent 1 position and velocity
            pos2, vel2: Agent 2 position and velocity
            horizon_frames: Number of frames to project forward
            
        Returns:
            float: Intersection risk in [0, 1]
        """
        # Assume 1 pixel per frame per unit velocity (adjust based on FPS)
        future_pos1 = pos1 + vel1 * horizon_frames
        future_pos2 = pos2 + vel2 * horizon_frames
        
        # Check if paths cross: use minimum distance along trajectory
        min_distance = np.inf
        for t in range(horizon_frames + 1):
            p1_t = pos1 + vel1 * t
            p2_t = pos2 + vel2 * t
            dist = np.linalg.norm(p1_t - p2_t)
            min_distance = min(min_distance, dist)
        
        # Risk increases as min_distance decreases
        intersection_risk = 1.0 / (1.0 + np.exp((min_distance - 50) / 20))
        
        return float(np.clip(intersection_risk, 0, 1))
    
    def compute_frame_interactions(self, frame_idx: int,
                                  frame_states: Dict[int, 'FrameState']) -> Dict[int, InteractionFeatures]:
        """
        Compute interaction features for all agents in a frame.
        
        Args:
            frame_idx: Frame index
            frame_states: Dict of track_id -> FrameState for this frame
            
        Returns:
            dict: track_id -> InteractionFeatures
        """
        interactions = {}
        
        agent_ids = list(frame_states.keys())
        num_agents = len(agent_ids)
        
        for track_id, state in frame_states.items():
            pos = np.array([state.center_x, state.center_y])
            vel = np.array([state.velocity_x, state.velocity_y])
            
            # Initialize features
            features = InteractionFeatures(
                track_id=track_id,
                frame_idx=frame_idx,
                concurrent_agents=num_agents - 1  # exclude self
            )
            
            # Compute pairwise interactions with other agents
            min_distance = np.inf
            best_agent_id = -1
            nearby_count = 0
            
            for other_id, other_state in frame_states.items():
                if other_id == track_id:
                    continue
                
                other_pos = np.array([other_state.center_x, other_state.center_y])
                other_vel = np.array([other_state.velocity_x, other_state.velocity_y])
                
                # Distance
                distance = self._euclidean_distance(pos, other_pos)
                
                # Track closest agent
                if distance < min_distance:
                    min_distance = distance
                    best_agent_id = other_id
                    features.nearest_agent_id = other_id
                
                # Count nearby agents
                if distance < self.proximity_threshold:
                    nearby_count += 1
                    
                    # Compute interaction features with this nearby agent
                    rel_vel = other_vel - vel
                    features.relative_velocity_x = rel_vel[0]
                    features.relative_velocity_y = rel_vel[1]
                    features.relative_velocity_mag = np.linalg.norm(rel_vel)
                    
                    # Heading difference
                    if np.linalg.norm(vel) > 1e-6 and np.linalg.norm(other_vel) > 1e-6:
                        heading_self = np.arctan2(vel[1], vel[0])
                        heading_other = np.arctan2(other_vel[1], other_vel[0])
                        features.relative_heading = heading_other - heading_self
                    
                    # TTC with closest agent
                    if distance < min(1000, self.proximity_threshold):
                        ttc = self._calculate_ttc(pos, vel, other_pos, other_vel)
                        if ttc < features.ttc_nearest:
                            features.ttc_nearest = ttc
                    
                    # Intersection heuristic
                    intersection = self._check_intersection_heuristic(
                        pos, vel, other_pos, other_vel, horizon_frames=10
                    )
                    features.intersection_risk = max(features.intersection_risk, intersection)
            
            # Set computed features
            if min_distance != np.inf:
                features.distance_to_nearest = min_distance
                features.num_nearby_agents = nearby_count
                
                # Collision indicators
                features.is_on_collision_course = (features.ttc_nearest < self.ttc_threshold
                                                   and features.ttc_nearest > 0)
                features.collision_indicator = self._collision_indicator_soft(
                    features.ttc_nearest,
                    features.distance_to_nearest
                )
            
            interactions[track_id] = features
        
        return interactions
    
    def compute_trajectory_interactions(self, trajectories: Dict[int, 'AgentTrajectory']) -> Dict[int, Dict[int, InteractionFeatures]]:
        """
        Compute interaction features for all trajectories.
        
        Args:
            trajectories: Dict of track_id -> AgentTrajectory
            
        Returns:
            dict: track_id -> frame_idx -> InteractionFeatures
        """
        all_interactions = {}
        
        # Collect all unique frame indices
        frame_indices = set()
        for trajectory in trajectories.values():
            for state in trajectory.states:
                frame_indices.add(state.frame_idx)
        
        frame_indices = sorted(frame_indices)
        
        # Process each frame
        for frame_idx in frame_indices:
            # Collect all agents present in this frame
            frame_states = {}
            for track_id, trajectory in trajectories.items():
                state = trajectory.get_state_at_frame(frame_idx)
                if state is not None:
                    frame_states[track_id] = state
            
            # Compute interactions for this frame
            frame_interactions = self.compute_frame_interactions(frame_idx, frame_states)
            
            # Store
            for track_id, interactions in frame_interactions.items():
                if track_id not in all_interactions:
                    all_interactions[track_id] = {}
                all_interactions[track_id][frame_idx] = interactions
        
        logger.info(f"Computed interactions for {len(all_interactions)} agents across {len(frame_indices)} frames")
        return all_interactions
    
    def get_interaction_stats(self, all_interactions: Dict[int, Dict[int, InteractionFeatures]]) -> dict:
        """Get summary statistics of interactions."""
        if not all_interactions:
            return {}
        
        all_ttc = []
        all_distances = []
        all_collision_indicators = []
        
        for track_interactions in all_interactions.values():
            for interaction in track_interactions.values():
                if interaction.ttc_nearest != np.inf:
                    all_ttc.append(interaction.ttc_nearest)
                if interaction.distance_to_nearest != np.inf:
                    all_distances.append(interaction.distance_to_nearest)
                all_collision_indicators.append(interaction.collision_indicator)
        
        return {
            'num_agents': len(all_interactions),
            'avg_ttc': np.mean(all_ttc) if all_ttc else np.inf,
            'min_ttc': np.min(all_ttc) if all_ttc else np.inf,
            'avg_distance': np.mean(all_distances) if all_distances else np.inf,
            'min_distance': np.min(all_distances) if all_distances else np.inf,
            'avg_collision_indicator': np.mean(all_collision_indicators) if all_collision_indicators else 0,
        }
