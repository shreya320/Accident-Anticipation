#!/usr/bin/env python
"""
Full working pipeline demo - processes video up to state_encoder.
This demonstrates the complete detection, tracking, trajectory extraction, 
and state encoding pipeline.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=" * 80)
        logger.info("ACCIDENT ANTICIPATION PIPELINE - WORKING DEMO")
        logger.info("Demonstrating: Preprocessing → Detection → Tracking → State Encoding")
        logger.info("=" * 80)
        
        # Import modules
        logger.info("\n[1/5] Importing modules...")
        from src.preprocessing import VideoPreprocessor
        from src.detection_tracking import DetectionTrackingPipeline, Track, Detection
        from src.trajectory import AgentTrajectory, FrameState
        from src.state_encoder import StateEncoder
        
        logger.info("✓ Modules imported")
        
        # Load video
        logger.info("\n[2/5] Loading video...")
        video_path = Path('data/videos/08PPPXtzN4A.mp4')
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return 1
        
        preprocessor = VideoPreprocessor(target_fps=10, target_width=640, target_height=480)
        preprocessor.open_video(str(video_path))
        
        frames = []
        frame_times = []
        num_frames_to_process = 10  # Process 10 frames for demo
        
        for frame, idx, timestamp in preprocessor.get_frame_generator():
            frames.append(frame)
            frame_times.append(timestamp)
            if len(frames) >= num_frames_to_process:
                break
        
        preprocessor.close_video()
        logger.info(f"✓ Loaded {len(frames)} frames for processing")
        
        # Detection and tracking
        logger.info("\n[3/5] Running detection and tracking...")
        detector_tracker = DetectionTrackingPipeline(
            model_name='yolov8m.pt',
            confidence=0.5,
            device='cpu',
            use_byte_track=False
        )
        
        for frame_idx, frame in enumerate(frames):
            logger.info(f"  Frame {frame_idx + 1}/{len(frames)}: Detecting objects...")
            frame_det, det_to_track = detector_tracker.process_frame(frame, frame_times[frame_idx])
            logger.info(f"    ✓ Found {len(frame_det.detections)} objects")
        
        all_tracks = detector_tracker.get_tracks(min_duration_frames=1)
        logger.info(f"✓ Detected and tracked {len(all_tracks)} objects")
        
        # Trajectory extraction
        logger.info("\n[4/5] Extracting trajectories...")
        trajectories = {}
        for track_id, track in all_tracks.items():
            traj = AgentTrajectory(
                track_id=track_id,
                class_name=track.class_name,
                class_id=track.class_id
            )
            
            for frame_idx, det in track.detections:
                state = FrameState(
                    frame_idx=frame_idx,
                    timestamp=frame_times[frame_idx] if frame_idx < len(frame_times) else 0.0,
                    center_x=det.center[0],
                    center_y=det.center[1],
                    velocity_x=0.0,
                    velocity_y=0.0,
                    speed=0.0,
                    acceleration_x=0.0,
                    acceleration_y=0.0
                )
                traj.add_state(state)
            
            trajectories[track_id] = traj
        
        logger.info(f"✓ Extracted {len(trajectories)} trajectories")
        for track_id, traj in trajectories.items():
            logger.info(f"  - Track {track_id} ({traj.class_name}): {len(traj.states)} frames")
        
        # State encoding
        logger.info("\n[5/5] Encoding trajectories to latent space...")
        state_encoder = StateEncoder(
            input_dim=6,
            hidden_dim=64,
            latent_dim=32,
            device='cpu',
            window_size=20
        )
        
        all_encodings = {}
        total_states = sum(len(traj.states) for traj in trajectories.values())
        states_encoded = 0
        
        for track_id, trajectory in trajectories.items():
            encodings = {}
            for state in trajectory.states:
                latent_state = state_encoder.encode_trajectory_at_frame(trajectory, state.frame_idx)
                if latent_state is not None:
                    encodings[state.frame_idx] = latent_state
                    states_encoded += 1
            
            if encodings:
                all_encodings[track_id] = encodings
        
        logger.info(f"✓ Encoded {states_encoded} states")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ Input: {len(frames)} video frames")
        logger.info(f"✓ Detection: {len(all_tracks)} tracked objects")
        logger.info(f"✓ Trajectories: {len(trajectories)} time series")
        logger.info(f"✓ State Encoding: {states_encoded} latent representations")
        logger.info(f"✓ Latent Dimension: 32")
        logger.info("\n✓✓✓ PIPELINE SUCCESSFULLY PROCESSES VIDEO UP TO STATE_ENCODER! ✓✓✓")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\n✗ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n✗ Pipeline error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
