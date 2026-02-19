"""
Demo and Testing Script
Quick test of individual components
"""

import sys
from pathlib import Path
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from src.preprocessing import VideoPreprocessor
from src.detection_tracking import Detection, Track
from src.trajectory import TrajectoryExtractor, AgentTrajectory, FrameState
from src.interaction_features import InteractionFeatureComputer
from src.state_encoder import StateEncoder
from src.trajectory_sampler import TrajectoryDiffusionSampler, SimpleTrajectoryPredictor
from src.risk_evaluator import RiskEvaluator
from src.risk_aggregation import RiskAggregationPipeline, TemporalSmoother

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_preprocessing():
    """Test video preprocessing module."""
    logger.info("=" * 60)
    logger.info("Testing Preprocessing Module")
    logger.info("=" * 60)
    
    preprocessor = VideoPreprocessor(target_fps=10, target_width=640, target_height=480)
    logger.info(f"✓ VideoPreprocessor initialized")
    logger.info(f"  Target FPS: {preprocessor.target_fps}")
    logger.info(f"  Target size: {preprocessor.target_width}x{preprocessor.target_height}")


def test_trajectory():
    """Test trajectory module with synthetic data."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Trajectory Extraction Module")
    logger.info("=" * 60)
    
    # Create synthetic trajectory
    trajectory = AgentTrajectory(track_id=1, class_name='car', class_id=2)
    
    for i in range(10):
        state = FrameState(
            frame_idx=i,
            timestamp=i / 10.0,
            center_x=100.0 + i * 5.0,
            center_y=200.0 + i * 2.0,
            velocity_x=5.0,
            velocity_y=2.0,
            speed=np.sqrt(5.0**2 + 2.0**2)
        )
        trajectory.add_state(state)
    
    logger.info(f"✓ Created synthetic trajectory")
    logger.info(f"  Track ID: {trajectory.track_id}")
    logger.info(f"  Class: {trajectory.class_name}")
    logger.info(f"  Duration: {trajectory.duration_frames()} frames")
    logger.info(f"  Avg speed: {trajectory.get_speeds().mean():.2f} px/frame")


def test_interaction_features():
    """Test interaction feature computation."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Interaction Features Module")
    logger.info("=" * 60)
    
    computer = InteractionFeatureComputer(
        proximity_threshold=300.0,
        ttc_threshold=2.0
    )
    logger.info(f"✓ InteractionFeatureComputer initialized")
    logger.info(f"  Proximity threshold: {computer.proximity_threshold} px")
    logger.info(f"  TTC threshold: {computer.ttc_threshold} s")


def test_state_encoder():
    """Test state encoder."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing State Encoder Module")
    logger.info("=" * 60)
    
    device = 'cpu'
    encoder = StateEncoder(
        input_dim=6,
        hidden_dim=64,
        latent_dim=32,
        device=device,
        window_size=20
    )
    logger.info(f"✓ StateEncoder initialized")
    logger.info(f"  Device: {device}")
    logger.info(f"  Latent dim: {encoder.latent_dim}")
    logger.info(f"  Window size: {encoder.window_size}")


def test_trajectory_sampler():
    """Test trajectory sampling."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Trajectory Sampler Module")
    logger.info("=" * 60)
    
    device = 'cpu'
    sampler = TrajectoryDiffusionSampler(
        latent_dim=32,
        device=device,
        num_denoising_steps=10,
        future_frames=30
    )
    logger.info(f"✓ TrajectoryDiffusionSampler initialized")
    logger.info(f"  Denoising steps: {sampler.num_denoising_steps}")
    logger.info(f"  Future frames: {sampler.future_frames}")
    
    # Test sampling
    latent = np.random.randn(32).astype(np.float32)
    start_pos = np.array([100.0, 200.0])
    
    samples = sampler.sample_trajectories(latent, start_pos, num_samples=3)
    logger.info(f"✓ Generated {len(samples)} trajectory samples")
    for sample in samples:
        logger.info(f"  Sample {sample.sample_idx}: {sample.trajectory.shape}")
    
    # Test simple predictor
    logger.info("\nTesting SimpleTrajectoryPredictor...")
    simple_pred = SimpleTrajectoryPredictor(future_frames=30)
    current_vel = np.array([3.0, 1.0])
    simple_samples = simple_pred.predict_trajectory(start_pos, current_vel, num_samples=3)
    logger.info(f"✓ Generated {len(simple_samples)} simple trajectory samples")


def test_risk_evaluator():
    """Test risk evaluation."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Risk Evaluator Module")
    logger.info("=" * 60)
    
    evaluator = RiskEvaluator(
        collision_distance=30.0,
        near_miss_distance=100.0
    )
    logger.info(f"✓ RiskEvaluator initialized")
    logger.info(f"  Collision distance: {evaluator.collision_distance} px")
    logger.info(f"  Near-miss distance: {evaluator.near_miss_distance} px")


def test_risk_aggregation():
    """Test risk aggregation and smoothing."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Risk Aggregation & Smoothing Module")
    logger.info("=" * 60)
    
    aggregator = RiskAggregationPipeline(
        aggregation_method='mean',
        temporal_alpha=0.3
    )
    logger.info(f"✓ RiskAggregationPipeline initialized")
    logger.info(f"  Aggregation method: mean")
    logger.info(f"  Temporal alpha: 0.3")
    
    # Test smoothing
    smoother = TemporalSmoother(alpha=0.3)
    risks = [0.1, 0.2, 0.15, 0.3, 0.4, 0.35]
    
    smoothed = []
    for i, risk in enumerate(risks):
        smooth_risk = smoother.smooth(track_id=1, current_risk=risk)
        smoothed.append(smooth_risk)
    
    logger.info(f"✓ Applied temporal smoothing")
    logger.info(f"  Original: {risks}")
    logger.info(f"  Smoothed: {[f'{r:.3f}' for r in smoothed]}")


def test_pipeline():
    """Test complete pipeline configuration."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Complete Pipeline")
    logger.info("=" * 60)
    
    from src.pipeline import AccidentAnticipationPipeline
    
    config = AccidentAnticipationPipeline.get_default_config()
    logger.info(f"✓ Pipeline configuration loaded")
    logger.info(f"  Target FPS: {config['target_fps']}")
    logger.info(f"  YOLO model: {config['yolo_model']}")
    logger.info(f"  Latent dim: {config['latent_dim']}")
    logger.info(f"  Num samples: {config['num_trajectory_samples']}")
    logger.info(f"  Prediction horizon: {config['prediction_horizon']} frames")


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 58 + "║")
    logger.info("║" + "  ACCIDENT ANTICIPATION SYSTEM - COMPONENT TESTS".center(58) + "║")
    logger.info("║" + " " * 58 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    try:
        test_preprocessing()
        test_trajectory()
        test_interaction_features()
        test_state_encoder()
        test_trajectory_sampler()
        test_risk_evaluator()
        test_risk_aggregation()
        test_pipeline()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nYou can now run: python main.py --video <path> --output output")
        logger.info("\nFor help: python main.py --help")
        
        return 0
    
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
