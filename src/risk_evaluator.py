"""
Risk Evaluation Module
- Evaluate safety/risk for each generated future trajectory
- Detect unsafe events: collision, near-miss, abrupt braking
- Output soft risk score [0, 1] per future
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollisionEvent:
    """Detected collision event."""
    frame_idx: int
    agent_id: int
    other_agent_id: int
    distance: float
    event_type: str  # 'collision', 'near_miss', 'abrupt_brake'
    severity: float  # [0, 1]


@dataclass
class FutureRiskScore:
    """Risk evaluation for a single future trajectory."""
    track_id: int
    sample_idx: int
    collision_risk: float = 0.0  # [0, 1]
    near_miss_risk: float = 0.0
    abrupt_brake_risk: float = 0.0
    composite_risk: float = 0.0  # Aggregate risk
    events: List[CollisionEvent] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
    
    def compute_composite(self, collision_weight: float = 0.85,
                     near_miss_weight: float = 0.15):

        self.composite_risk = (
            collision_weight * self.collision_risk +
            near_miss_weight * self.near_miss_risk
        )

        self.composite_risk = np.clip(self.composite_risk, 0, 1)


class RiskEvaluator:
    """Evaluates risk for predicted future trajectories."""
    
    def __init__(self, collision_distance: float = 60.0,
                 near_miss_distance: float = 150.0,
                 ttc_threshold: float = 1.5,
                 braking_threshold: float = 3.0):
        """
        Initialize risk evaluator.
        
        Args:
            collision_distance: Distance threshold for collision (pixels)
            near_miss_distance: Distance threshold for near-miss
            ttc_threshold: Time-to-collision threshold (seconds)
            braking_threshold: Acceleration magnitude for abrupt braking
        """
        self.collision_distance = collision_distance
        self.near_miss_distance = near_miss_distance
        self.ttc_threshold = ttc_threshold
        self.braking_threshold = braking_threshold
    
    def _compute_ttc_trajectory(self, traj1: np.ndarray, vel1: np.ndarray,
                               traj2: np.ndarray, vel2: np.ndarray,
                               fps: float = 10.0) -> Tuple[float, int]:
        """
        Compute minimum TTC along two trajectories.
        
        Args:
            traj1, traj2: Trajectory positions (N, 2)
            vel1, vel2: Velocity sequences (N, 2)
            fps: Frames per second
            
        Returns:
            tuple: (min_ttc, frame_of_min)
        """
        min_ttc = np.inf
        min_frame = -1
        
        min_len = min(len(traj1), len(traj2))
        
        for i in range(min_len):
            pos1 = traj1[i]
            pos2 = traj2[i]
            
            # Relative position and velocity
            rel_pos = pos2 - pos1
            rel_vel = vel2[i] - vel1[i]
            
            distance = np.linalg.norm(rel_pos)
            
            # Closing speed
            if np.linalg.norm(rel_vel) < 1e-6:
                continue
            
            rel_vel_norm = rel_vel / np.linalg.norm(rel_vel)
            closing_speed = -np.dot(rel_pos, rel_vel_norm)
            
            if closing_speed > 0:
                ttc = distance / (closing_speed + 1e-6) / fps
                if ttc < min_ttc:
                    min_ttc = ttc
                    min_frame = i
        
        return min_ttc, min_frame
    
    def _detect_collision_events(self, track_id: int,
                                trajectory: np.ndarray,
                                velocities: np.ndarray,
                                other_trajectories: Dict[int, Tuple[np.ndarray, np.ndarray]],
                                fps: float = 10.0) -> List[CollisionEvent]:
        """
        Detect collision events between ego trajectory and others.
        
        Args:
            track_id: Track ID of ego agent
            trajectory: Ego future trajectory (N, 2)
            velocities: Ego velocities (N, 2)
            other_trajectories: Dict of other_track_id -> (traj, vels)
            fps: Frames per second
            
        Returns:
            list: CollisionEvent objects
        """
        events = []
        
        for other_id, (other_traj, other_vels) in other_trajectories.items():
            # Check distance at each frame
            min_dist = np.inf
            collision_frames = []
            near_miss_frames = []
            
            min_len = min(len(trajectory), len(other_traj))
            
            for frame in range(min_len):
                dist = np.linalg.norm(trajectory[frame] - other_traj[frame])
                min_dist = min(min_dist, dist)
                
                # Collision
                if dist < self.collision_distance:
                    collision_frames.append(frame)
                # Near-miss
                elif dist < self.near_miss_distance:
                    near_miss_frames.append(frame)
            
            # Create events
            if collision_frames:
                # Find worst collision
                worst_dist = min(np.linalg.norm(trajectory[f] - other_traj[f]) 
                                for f in collision_frames)
                event = CollisionEvent(
                    frame_idx=collision_frames[0],
                    agent_id=track_id,
                    other_agent_id=other_id,
                    distance=worst_dist,
                    event_type='collision',
                    severity=1.0 - np.clip(worst_dist / self.collision_distance, 0, 1)
                )
                events.append(event)
            
            elif near_miss_frames:
                # Near-miss
                worst_dist = min(np.linalg.norm(trajectory[f] - other_traj[f])
                                for f in near_miss_frames)
                event = CollisionEvent(
                    frame_idx=near_miss_frames[0],
                    agent_id=track_id,
                    other_agent_id=other_id,
                    distance=worst_dist,
                    event_type='near_miss',
                    severity=1.0 - np.clip((worst_dist - self.collision_distance) / 
                                          (self.near_miss_distance - self.collision_distance), 0, 1)
                )
                events.append(event)
        
        return events
    
    def _detect_abrupt_braking(self, velocities: np.ndarray,
                              track_id: int, fps: float = 10.0) -> Tuple[float, List[int]]:
        """
        Detect abrupt braking in trajectory.
        
        Args:
            velocities: Velocity sequence (N, 2)
            track_id: Track ID
            fps: Frames per second
            
        Returns:
            tuple: (max_braking_risk, braking_frames)
        """
        speeds = np.linalg.norm(velocities, axis=1)
        accelerations = np.zeros(len(speeds))
        
        # Compute acceleration
        for i in range(1, len(speeds)):
            acc = (speeds[i] - speeds[i-1])
            accelerations[i] = acc
        
        # Find abrupt braking (negative acceleration)
        braking_frames = np.where(accelerations < -self.braking_threshold)[0]
        
        if len(braking_frames) == 0:
            return 0.0, []
        
        # Max braking severity
        max_braking = np.min(accelerations[braking_frames])
        braking_risk = np.clip(np.abs(max_braking) / self.braking_threshold, 0, 1)
        
        return braking_risk, braking_frames.tolist()
    
    def evaluate_trajectory(self, track_id: int, trajectory: np.ndarray,
                           velocities: np.ndarray,
                           other_trajectories: Dict[int, Tuple[np.ndarray, np.ndarray]],
                           fps: float = 10.0) -> FutureRiskScore:
        """
        Evaluate risk for a single future trajectory.
        
        Args:
            track_id: Track ID
            trajectory: Future positions (N, 2)
            velocities: Velocity sequence (N, 2)
            other_trajectories: Dict of other_id -> (traj, vels)
            fps: Frames per second
            
        Returns:
            FutureRiskScore
        """
        score = FutureRiskScore(track_id=track_id, sample_idx=0)
        
        # Detect collision events
        events = self._detect_collision_events(
            track_id, trajectory, velocities, other_trajectories, fps
        )
        score.events = events
        
        # Collision risk: max severity among collision events
        collision_events = [e for e in events if e.event_type == 'collision']
        if collision_events:
            score.collision_risk = max(e.severity for e in collision_events)
        
        # Near-miss risk
        near_miss_events = [e for e in events if e.event_type == 'near_miss']
        if near_miss_events:
            score.near_miss_risk = max(e.severity for e in near_miss_events)
        
        # Abrupt braking risk
        brake_risk, brake_frames = self._detect_abrupt_braking(velocities, track_id, fps)
        score.abrupt_brake_risk = brake_risk
        
        # Compute composite risk
        score.compute_composite()
        
        return score
    
    def batch_evaluate(self, future_trajectories: Dict[int, List],
                      other_trajectories: Dict[int, Tuple[np.ndarray, np.ndarray]] = None,
                      fps: float = 10.0) -> Dict[int, List[FutureRiskScore]]:
        """
        Evaluate risk for all trajectory samples.
        
        Args:
            future_trajectories: Dict of track_id -> list of FutureTrajectory
            other_trajectories: Dict of track_id -> (traj, vels) for reference
            fps: Frames per second
            
        Returns:
            dict: track_id -> list of FutureRiskScore
        """
        if other_trajectories is None:
            other_trajectories = {}
        
        all_scores = {}
        
        for track_id, samples in future_trajectories.items():
            scores = []
            
            for sample in samples:
                # Get other trajectories (exclude self)
                others = {oid: traj for oid, traj in other_trajectories.items()
                         if oid != track_id}
                
                # Evaluate
                score = self.evaluate_trajectory(
                    track_id, sample.trajectory, sample.velocities,
                    others, fps
                )
                score.sample_idx = sample.sample_idx
                
                scores.append(score)
            
            all_scores[track_id] = scores
        
        return all_scores


class FastRiskEvaluator:
    """
    Fast risk evaluator using simple heuristics (for real-time use).
    """
    
    def __init__(self, collision_distance: float = 30.0):
        """Initialize fast evaluator."""
        self.collision_distance = collision_distance
    
    def evaluate_trajectory(self, track_id: int, trajectory: np.ndarray) -> float:
        """
        Quick risk evaluation: check minimum distance to reference trajectory.
        
        Args:
            track_id: Track ID
            trajectory: Predicted future positions (N, 2)
            
        Returns:
            float: Risk score [0, 1]
        """
        # For now, return 0 (will be computed in context)
        return 0.0
