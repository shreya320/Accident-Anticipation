"""
Risk Aggregation & Smoothing Module
- Aggregate risk scores across K futures (mean, weighted mean)
- Temporal smoothing using exponential moving average
- Output final per-agent risk scores
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class AggregatedRiskScore:
    """Aggregated risk score for an agent at a timestep."""
    track_id: int
    frame_idx: int
    timestamp: float
    
    # Per-component risks
    collision_risk: float
    near_miss_risk: float
    abrupt_brake_risk: float
    
    # Aggregated risks
    mean_composite_risk: float = 0.0  # Mean across samples
    max_composite_risk: float = 0.0   # Max across samples
    percentile_95_risk: float = 0.0   # 95th percentile
    
    # Smoothed risk
    smoothed_risk: float = 0.0
    
    # Metadata
    num_samples: int = 0


class RiskAggregator:
    """Aggregates risk across multiple future samples."""
    
    def __init__(self, aggregation_method: str = 'mean'):
        """
        Initialize risk aggregator.
        
        Args:
            aggregation_method: 'mean', 'max', 'weighted_mean'
        """
        self.aggregation_method = aggregation_method
    
    def aggregate_frame_risks(self, frame_risks: List,
                             weights: Optional[List[float]] = None) -> Tuple[float, float, float]:
        """
        Aggregate risk scores from multiple samples.
        
        Args:
            frame_risks: List of risk scores [0, 1]
            weights: Optional weights for weighted mean
            
        Returns:
            tuple: (mean_risk, max_risk, p95_risk)
        """
        if not frame_risks:
            return 0.0, 0.0, 0.0
        
        risks = np.array(frame_risks)
        
        if self.aggregation_method == 'mean':
            mean_risk = np.mean(risks)
        elif self.aggregation_method == 'max':
            mean_risk = np.max(risks)
        elif self.aggregation_method == 'weighted_mean':
            if weights is None:
                weights = np.ones(len(risks)) / len(risks)
            else:
                weights = np.array(weights) / np.sum(weights)
            mean_risk = np.average(risks, weights=weights)
        else:
            mean_risk = np.mean(risks)
        
        max_risk = np.max(risks)
        p95_risk = np.percentile(risks, 95) if len(risks) > 1 else max_risk
        
        return mean_risk, max_risk, p95_risk
    
    def aggregate_all_risks(self, risk_scores_per_sample: Dict[int, List]) -> Dict[int, AggregatedRiskScore]:
        """
        Aggregate risks for all samples of an agent.
        
        Args:
            risk_scores_per_sample: Dict of sample_idx -> FutureRiskScore
            
        Returns:
            dict: Aggregated scores
        """
        aggregated = AggregatedRiskScore(
            track_id=-1,
            frame_idx=-1,
            timestamp=0.0,
            collision_risk=0.0,
            near_miss_risk=0.0,
            abrupt_brake_risk=0.0
        )
        
        # Extract per-component risks
        collision_risks = [r.collision_risk for r in risk_scores_per_sample]
        near_miss_risks = [r.near_miss_risk for r in risk_scores_per_sample]
        brake_risks = [r.abrupt_brake_risk for r in risk_scores_per_sample]
        composite_risks = [r.composite_risk for r in risk_scores_per_sample]
        
        # Aggregate each component
        agg_collision, _, _ = self.aggregate_frame_risks(collision_risks)
        agg_near_miss, _, _ = self.aggregate_frame_risks(near_miss_risks)
        agg_brake, _, _ = self.aggregate_frame_risks(brake_risks)
        agg_mean, agg_max, agg_p95 = self.aggregate_frame_risks(composite_risks)
        
        aggregated.collision_risk = agg_collision
        aggregated.near_miss_risk = agg_near_miss
        aggregated.abrupt_brake_risk = agg_brake
        aggregated.mean_composite_risk = agg_mean
        aggregated.max_composite_risk = agg_max
        aggregated.percentile_95_risk = agg_p95
        aggregated.num_samples = len(risk_scores_per_sample)
        
        return aggregated


class TemporalSmoother:
    """
    Applies temporal smoothing to risk scores.
    Uses exponential moving average to reduce flickering.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize temporal smoother.
        
        Args:
            alpha: Smoothing factor (0, 1]. Lower = more smoothing.
                  0.3 = consider 30% current + 70% history
        """
        self.alpha = alpha
        self.history: Dict[int, float] = defaultdict(float)
    
    def smooth(self, track_id: int, current_risk: float) -> float:
        """
        Apply exponential moving average smoothing.
        
        Args:
            track_id: Track ID
            current_risk: Current risk score
            
        Returns:
            float: Smoothed risk score
        """
        if track_id not in self.history:
            smoothed = current_risk
        else:
            prev_smoothed = self.history[track_id]
            smoothed = self.alpha * current_risk + (1 - self.alpha) * prev_smoothed
        
        self.history[track_id] = smoothed
        return smoothed
    
    def smooth_trajectory(self, track_id: int, 
                         risk_sequence: np.ndarray) -> np.ndarray:
        """
        Smooth entire risk trajectory.
        
        Args:
            track_id: Track ID
            risk_sequence: Risk scores over time (N,)
            
        Returns:
            np.ndarray: Smoothed risk scores
        """
        smoothed = np.zeros_like(risk_sequence)
        
        # Forward pass
        smooth_val = risk_sequence[0]
        smoothed[0] = smooth_val
        
        for t in range(1, len(risk_sequence)):
            smooth_val = self.alpha * risk_sequence[t] + (1 - self.alpha) * smooth_val
            smoothed[t] = smooth_val
        
        return smoothed
    
    def reset(self):
        """Reset smoothing history."""
        self.history.clear()


class RiskAggregationPipeline:
    """Complete pipeline for risk aggregation and smoothing."""
    
    def __init__(self, aggregation_method: str = 'mean',
                 temporal_alpha: float = 0.3):
        """
        Initialize risk aggregation pipeline.
        
        Args:
            aggregation_method: How to aggregate across samples
            temporal_alpha: Temporal smoothing factor
        """
        self.aggregator = RiskAggregator(aggregation_method)
        self.smoother = TemporalSmoother(temporal_alpha)
    
    def process_frame_risks(self, frame_idx: int, timestamp: float,
                           risk_scores_by_agent: Dict[int, List]) -> Dict[int, AggregatedRiskScore]:
        """
        Process risks for a single frame.
        
        Args:
            frame_idx: Frame index
            timestamp: Frame timestamp
            risk_scores_by_agent: Dict of track_id -> list of FutureRiskScore
            
        Returns:
            dict: track_id -> AggregatedRiskScore (not yet smoothed)
        """
        frame_aggregated = {}
        
        for track_id, risk_scores in risk_scores_by_agent.items():
            # Aggregate risks for this agent
            agg = self.aggregator.aggregate_all_risks(risk_scores)
            
            agg.track_id = track_id
            agg.frame_idx = frame_idx
            agg.timestamp = timestamp
            
            frame_aggregated[track_id] = agg
        
        return frame_aggregated
    
    def apply_temporal_smoothing(self, track_id: int, 
                                aggregated_risk: AggregatedRiskScore) -> AggregatedRiskScore:
        """
        Apply temporal smoothing to aggregated risk.
        
        Args:
            track_id: Track ID
            aggregated_risk: Aggregated risk score
            
        Returns:
            AggregatedRiskScore: Risk with smoothed composite
        """
        smoothed_composite = self.smoother.smooth(
            track_id, aggregated_risk.mean_composite_risk
        )
        
        aggregated_risk.smoothed_risk = smoothed_composite
        
        return aggregated_risk
    
    def process_trajectory_risks(self, all_frame_risks: Dict[int, Dict[int, List]]) -> Dict[int, List[AggregatedRiskScore]]:
        """
        Process entire trajectory of risks.
        
        Args:
            all_frame_risks: Dict of frame_idx -> (track_id -> list of FutureRiskScore)
            
        Returns:
            dict: track_id -> list of AggregatedRiskScore (smoothed)
        """
        trajectory_risks = defaultdict(list)
        
        for frame_idx in sorted(all_frame_risks.keys()):
            risk_by_agent = all_frame_risks[frame_idx]
            
            for track_id, risk_scores in risk_by_agent.items():
                # Aggregate
                agg = self.aggregator.aggregate_all_risks(risk_scores)
                agg.track_id = track_id
                agg.frame_idx = frame_idx
                
                # Apply smoothing
                agg = self.apply_temporal_smoothing(track_id, agg)
                
                trajectory_risks[track_id].append(agg)
        
        return dict(trajectory_risks)


class RiskScoreNormalizer:
    """Normalizes and clamps risk scores."""
    
    @staticmethod
    def normalize(risk_score: float, min_val: float = 0.0, 
                 max_val: float = 1.0) -> float:
        """
        Normalize and clamp risk score.
        
        Args:
            risk_score: Risk value to normalize
            min_val: Minimum clamped value
            max_val: Maximum clamped value
            
        Returns:
            float: Normalized risk in [min_val, max_val]
        """
        return np.clip(risk_score, min_val, max_val)
    
    @staticmethod
    def get_risk_category(risk_score: float) -> str:
        """
        Categorize risk score.
        
        Args:
            risk_score: Risk value [0, 1]
            
        Returns:
            str: Risk category
        """
        if risk_score < 0.1:
            return 'very_low'
        elif risk_score < 0.25:
            return 'low'
        elif risk_score < 0.5:
            return 'medium'
        elif risk_score < 0.75:
            return 'high'
        else:
            return 'very_high'


def compute_final_risk_per_agent(aggregated_trajectory: List[AggregatedRiskScore]) -> float:
    """
    Compute single risk value for an entire agent trajectory.
    
    Args:
        aggregated_trajectory: List of AggregatedRiskScore over time
        
    Returns:
        float: Overall risk for agent
    """
    if not aggregated_trajectory:
        return 0.0
    
    smoothed_risks = [r.smoothed_risk for r in aggregated_trajectory]
    
    # Overall risk = max over trajectory (worst case)
    overall_risk = max(smoothed_risks) if smoothed_risks else 0.0
    
    return overall_risk
