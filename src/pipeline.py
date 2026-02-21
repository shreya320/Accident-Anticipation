"""
Integrated Pipeline Module
- Orchestrates the entire accident risk prediction pipeline
- Coordinates all modules from preprocessing to output
"""

import logging
from typing import Dict, Optional, List, Tuple
import numpy as np
import cv2
import json
from pathlib import Path
from datetime import datetime

# Import all modules
from .preprocessing import VideoPreprocessor
from .detection_tracking import DetectionTrackingPipeline
from .trajectory import TrajectoryExtractor, AgentTrajectory
from .interaction_features import InteractionFeatureComputer
from .state_encoder import StateEncoder
from .trajectory_sampler import TrajectoryDiffusionSampler
from .risk_evaluator import RiskEvaluator
from .risk_aggregation import RiskAggregationPipeline

logger = logging.getLogger(__name__)


class AccidentAnticipationPipeline:
    """
    Complete pipeline for accident risk prediction.
    Integrates all modules in sequence.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary with parameters
        """
        if config is None:
            config = self.get_default_config()
        
        self.config = config
        
        # Initialize modules
        logger.info("Initializing AccidentAnticipationPipeline...")
        
        # Preprocessing
        self.video_preprocessor = VideoPreprocessor(
            target_fps=config['target_fps'],
            target_width=config['frame_width'],
            target_height=config['frame_height']
        )
        
        # Detection & tracking
        device = 'cuda:0' if config['use_cuda'] else 'cpu'
        self.detection_tracker = DetectionTrackingPipeline(
            model_name=config['yolo_model'],
            confidence=config['detection_confidence'],
            device=device,
            use_byte_track=config['use_byte_track']
        )
        
        # Trajectory extraction
        self.trajectory_extractor = TrajectoryExtractor(
            pixels_per_meter=config['pixels_per_meter']
        )
        
        # Interaction features
        self.interaction_computer = InteractionFeatureComputer(
            proximity_threshold=config['proximity_threshold'],
            ttc_threshold=config['ttc_threshold']
        )
        
        # State encoder
        self.state_encoder = StateEncoder(
            latent_dim=config['latent_dim'],
            device=device,
            window_size=config['observation_window']
        )
        
        # Trajectory sampler
        self.trajectory_sampler = TrajectoryDiffusionSampler(
            latent_dim=config['latent_dim'],
            device=device,
            num_denoising_steps=config['num_diffusion_steps'],
            future_frames=config['prediction_horizon']
        )
        
        # Risk evaluator
        self.risk_evaluator = RiskEvaluator(
            collision_distance=config['collision_distance'],
            near_miss_distance=config['near_miss_distance']
        )
        
        # Risk aggregation & smoothing
        self.risk_aggregator = RiskAggregationPipeline(
            aggregation_method=config['aggregation_method'],
            temporal_alpha=config['temporal_alpha']
        )
        
        logger.info("Pipeline initialized successfully")
    
    @staticmethod
    def get_default_config() -> dict:
        """Get default configuration."""
        return {
            # Preprocessing
            'target_fps': 10,
            'frame_width': 640,
            'frame_height': 480,
            
            # Detection
            'yolo_model': 'yolov8m.pt',
            'detection_confidence': 0.5,
            'use_byte_track': True,
            'use_cuda': True,
            
            # Trajectory
            'pixels_per_meter': 1.0,
            
            # Interaction
            'proximity_threshold': 300.0,
            'ttc_threshold': 2.0,
            
            # State encoding
            'latent_dim': 32,
            'observation_window': 20,
            
            # Trajectory sampling
            'num_diffusion_steps': 10,
            'prediction_horizon': 30,  # frames
            'num_trajectory_samples': 5,
            
            # Risk evaluation
            'collision_distance': 30.0,
            'near_miss_distance': 100.0,
            
            # Risk aggregation
            'aggregation_method': 'mean',
            'temporal_alpha': 0.3,
        }
    
    def run(self, video_path: str, output_dir: str = 'output',
           save_video: bool = False, save_csv: bool = True) -> dict:
        """
        Run complete pipeline on a video.
        
        Args:
            video_path: Path to input video
            output_dir: Directory for outputs
            save_video: Whether to save annotated video
            save_csv: Whether to save CSV results
            
        Returns:
            dict: Summary results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing video: {video_path}")
        
        # Step 1: Load and preprocess video
        logger.info("Step 1: Loading and preprocessing video...")
        if not self.video_preprocessor.open_video(video_path):
            logger.error("Failed to open video")
            return {}
        
        frames = []
        frame_times = []
        
        for frame, idx, timestamp in self.video_preprocessor.get_frame_generator():
            frames.append(frame)
            frame_times.append(timestamp)
        
        self.video_preprocessor.close_video()
        logger.info(f"Loaded {len(frames)} preprocessed frames")
        
        # Step 2: Detection and tracking
        logger.info("Step 2: Running detection and tracking...")
        all_tracks = {}
        frame_det_info = []
        
        for frame_idx, frame in enumerate(frames):
            frame_det, det_to_track = self.detection_tracker.process_frame(
                frame, frame_times[frame_idx]
            )
            frame_det_info.append((frame_det, det_to_track))
        
        all_tracks = self.detection_tracker.get_tracks(min_duration_frames=3)
        logger.info(f"Detected {len(all_tracks)} tracks")
        
        if len(all_tracks) == 0:
            logger.warning("No tracks detected")
            return {}
        
        # Step 3: Trajectory extraction
        logger.info("Step 3: Extracting trajectories...")
        trajectories = self.trajectory_extractor.extract_trajectories(all_tracks)
        self.trajectory_extractor.update_timestamps(frame_times)
        self.trajectory_extractor.smooth_all_trajectories(window_size=3)
        logger.info(f"Extracted {len(trajectories)} trajectories")
        
        # Step 4: Interaction features
        logger.info("Step 4: Computing interaction features...")
        all_interactions = self.interaction_computer.compute_trajectory_interactions(trajectories)
        logger.info(f"Computed interactions for {len(all_interactions)} agents")
        
        # Step 5-10: Risk prediction
        logger.info("Step 5-10: Running risk prediction pipeline...")
        all_risk_scores_by_frame = {}
        
        # Collect all frame indices
        frame_indices = set()
        for trajectory in trajectories.values():
            for state in trajectory.states:
                frame_indices.add(state.frame_idx)
        
        frame_indices = sorted(frame_indices)
        
        for frame_idx in frame_indices:
            # Get agents present in this frame
            frame_agents = {}
            for track_id, trajectory in trajectories.items():
                state = trajectory.get_state_at_frame(frame_idx)
                if state is not None:
                    frame_agents[track_id] = (state, trajectory)
            
            if not frame_agents:
                continue
            
            # Step 5: Encode states
            latent_encodings = {}
            for track_id, (state, trajectory) in frame_agents.items():
                latent_state = self.state_encoder.encode_trajectory_at_frame(trajectory, frame_idx)
                if latent_state is not None:
                    latent_encodings[track_id] = latent_state.latent_vector
            
            # Step 6: Sample future trajectories
            future_trajectories = {}
            current_positions = {}
            for track_id, (state, trajectory) in frame_agents.items():
                if track_id in latent_encodings:
                    current_positions[track_id] = np.array([state.center_x, state.center_y])
            
            sampled_futures = self.trajectory_sampler.batch_sample_trajectories(
                latent_encodings, current_positions,
                num_samples=self.config['num_trajectory_samples']
            )
            
            # Prepare reference trajectories for risk evaluation
            other_trajectories = {}
            for other_id, other_trajectory in trajectories.items():
                other_state = other_trajectory.get_state_at_frame(frame_idx)
                if other_state is not None and other_id != track_id:
                    # Get future-like trajectory from current and next frames
                    future_states = other_trajectory.get_states_in_frame_range(
                        frame_idx, min(frame_idx + 30, other_trajectory.states[-1].frame_idx)
                    )
                    if len(future_states) > 1:
                        other_traj = other_trajectory.get_positions()
                        other_vels = other_trajectory.get_velocities()
                        other_trajectories[other_id] = (other_traj, other_vels)
            
            # Step 7-8: Evaluate risks for each sample
            risk_scores = self.risk_evaluator.batch_evaluate(
                sampled_futures,
                other_trajectories,
                fps=self.config['target_fps']
            )
            
            # Step 9-10: Aggregate and smooth risks
            # Convert to format expected by aggregator
            risk_by_agent = {}
            for track_id, scores in risk_scores.items():
                risk_by_agent[track_id] = scores
            
            all_risk_scores_by_frame[frame_idx] = risk_by_agent
        
        # Final aggregation and smoothing
        logger.info("Performing final risk aggregation and smoothing...")
        final_risks = self.risk_aggregator.process_trajectory_risks(all_risk_scores_by_frame)
        
        # Prepare results
        results = {
            'video_path': str(video_path),
            'num_tracks': len(trajectories),
            'num_frames': len(frames),
            'frame_times': frame_times,
            'trajectories': trajectories,
            'final_risks': final_risks,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save outputs
        logger.info("Saving results...")
        if save_csv:
            self._save_csv_results(results, output_dir)
        
        # Save JSON summary
        self._save_json_summary(results, output_dir)
        
        # Save annotated video
        if save_video:
            self._save_annotated_video(frames, results, output_dir)
        
        logger.info(f"Pipeline complete. Results saved to {output_dir}")
        
        return results
    
    def _save_csv_results(self, results: dict, output_dir: Path):
        """Save results to CSV files."""
        import csv
        
        # Trajectories CSV
        traj_path = output_dir / 'trajectories.csv'
        with open(traj_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['track_id', 'frame_idx', 'timestamp', 'center_x', 'center_y',
                           'velocity_x', 'velocity_y', 'speed', 'class_name'])
            
            for track_id, trajectory in results['trajectories'].items():
                for state in trajectory.states:
                    writer.writerow([
                        track_id, state.frame_idx, state.timestamp,
                        state.center_x, state.center_y,
                        state.velocity_x, state.velocity_y, state.speed,
                        trajectory.class_name
                    ])
        
        logger.info(f"Saved trajectories to {traj_path}")
        
        # Risk scores CSV
        risk_path = output_dir / 'risk_scores.csv'
        with open(risk_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['track_id', 'frame_idx', 'timestamp',
                           'collision_risk', 'near_miss_risk', 'abrupt_brake_risk',
                           'mean_composite_risk', 'smoothed_risk', 'num_samples'])
            
            for track_id, risk_sequence in results['final_risks'].items():
                for risk_score in risk_sequence:
                    writer.writerow([
                        track_id, risk_score.frame_idx, risk_score.timestamp,
                        risk_score.collision_risk, risk_score.near_miss_risk,
                        risk_score.abrupt_brake_risk, risk_score.mean_composite_risk,
                        risk_score.smoothed_risk, risk_score.num_samples
                    ])
        
        logger.info(f"Saved risk scores to {risk_path}")
    
    def _save_json_summary(self, results: dict, output_dir: Path):
        """Save summary to JSON."""
        summary = {
            'video_path': results['video_path'],
            'timestamp': results['timestamp'],
            'num_tracks': results['num_tracks'],
            'num_frames': results['num_frames'],
            'target_fps': self.config['target_fps'],
            'prediction_horizon': self.config['prediction_horizon'],
            'num_samples_per_agent': self.config['num_trajectory_samples'],
        }
        
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_path}")
    
    def _save_annotated_video(self, frames: list, results: dict, output_dir: Path):
        """Save annotated video with risk overlays."""
        # Get video info
        orig_height, orig_width = frames[0].shape[:2]
        
        video_path = output_dir / 'annotated_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.config['target_fps'],
                            (orig_width, orig_height))
        
        for frame_idx, frame in enumerate(frames):
            frame_annotated = frame.copy()
            
            # Draw tracks and risk scores
            for track_id, risk_sequence in results['final_risks'].items():
                for risk_score in risk_sequence:
                    if risk_score.frame_idx == frame_idx:
                        # Get trajectory position at this frame
                        trajectory = results['trajectories'][track_id]
                        state = trajectory.get_state_at_frame(frame_idx)
                        
                        if state is not None:
                            # Draw bounding circle
                            cx, cy = int(state.center_x), int(state.center_y)
                            
                            # Color based on risk
                            risk = risk_score.smoothed_risk
                            if risk < 0.2:
                                color = (0, 255, 0)  # Green
                            elif risk < 0.5:
                                color = (0, 255, 255)  # Yellow
                            else:
                                color = (0, 0, 255)  # Red
                            
                            cv2.circle(frame_annotated, (cx, cy), 20, color, 2)
                            cv2.putText(frame_annotated,
                                      f"ID:{track_id} Risk:{risk:.2f}",
                                      (cx - 20, cy - 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            out.write(frame_annotated)
        
        out.release()
        logger.info(f"Saved annotated video to {video_path}")
