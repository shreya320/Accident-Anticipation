#!/usr/bin/env python
"""Minimal test to check pipeline up to state_encoder."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    logger.info("=" * 80)
    logger.info("MINIMAL PIPELINE TEST - PROCESS 5 FRAMES ONLY")
    logger.info("=" * 80)
    
    # Step 1: Import all modules
    logger.info("Step 1: Importing modules...")
    from src.preprocessing import VideoPreprocessor
    from src.detection_tracking import DetectionTrackingPipeline, Detection, Track, FrameDetections
    from src.trajectory import TrajectoryExtractor, AgentTrajectory, FrameState
    from src.state_encoder import StateEncoder
    logger.info("✓ All modules imported successfully")
    
    # Step 2: Load video
    logger.info("\nStep 2: Loading video...")
    video_path = Path('data/videos/08PPPXtzN4A.mp4')
    preprocessor = VideoPreprocessor(target_fps=10, target_width=640, target_height=480)
    
    if not preprocessor.open_video(str(video_path)):
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)
    
    frames = []
    frame_times = []
    max_frames = 5  # Process only 5 frames for speed
    
    for frame, idx, timestamp in preprocessor.get_frame_generator():
        frames.append(frame)
        frame_times.append(timestamp)
        if len(frames) >= max_frames:
            break
    
    preprocessor.close_video()
    logger.info(f"✓ Loaded {len(frames)} frames")
    
    # Step 3: Detection and tracking
    logger.info("\nStep 3: Running detection and tracking on first 5 frames...")
    detector_tracker = DetectionTrackingPipeline(
        model_name='yolov8m.pt',
        confidence=0.5,
        device='cpu',
        use_byte_track=False
    )
    
    all_tracks = {}
    frame_det_info = []
    
    for frame_idx, frame in enumerate(frames):
        logger.info(f"  Processing frame {frame_idx}/{len(frames)}...")
        frame_det, det_to_track = detector_tracker.process_frame(frame, frame_times[frame_idx])
        frame_det_info.append((frame_det, det_to_track))
        logger.info(f"    Found {len(frame_det.detections)} detections")
    
    all_tracks = detector_tracker.get_tracks(min_duration_frames=1)
    logger.info(f"✓ Detection complete. Found {len(all_tracks)} tracks")
    
    if len(all_tracks) == 0:
        logger.warning("No tracks detected in these frames. Creating dummy track for testing...")
        # Create a dummy track for testing
        dummy_track = Track(
            track_id=1,
            class_name='car',
            class_id=2
        )
        for i in range(min(3, len(frames))):
            dummy_det = Detection(
                x1=100 + i*10, y1=100 + i*10,
                x2=200 + i*10, y2=200 + i*10,
                confidence=0.9,
                class_id=2,
                class_name='car'
            )
            dummy_track.add_detection(i, dummy_det)
        all_tracks[1] = dummy_track
        logger.info("  Created dummy track")
    
    # Step 4: Trajectory extraction
    logger.info("\nStep 4: Extracting trajectories...")
    trajectory_extractor = TrajectoryExtractor(pixels_per_meter=1.0)
    
    # Create AgentTrajectory objects from tracks
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
        logger.info(f"  Track {track_id}: {len(traj.states)} states, {traj.class_name}")
    
    # Step 5: State encoder
    logger.info("\nStep 5: Testing state encoder...")
    state_encoder = StateEncoder(
        input_dim=6,
        hidden_dim=64,
        latent_dim=32,
        device='cpu',
        window_size=20
    )
    
    # Encode all trajectories
    all_encodings = {}
    for track_id, trajectory in trajectories.items():
        logger.info(f"  Encoding trajectory {track_id}...")
        
        encodings = {}
        for state in trajectory.states:
            latent_state = state_encoder.encode_trajectory_at_frame(trajectory, state.frame_idx)
            if latent_state is not None:
                encodings[state.frame_idx] = latent_state
                logger.info(f"    Frame {state.frame_idx}: latent_dim={len(latent_state.latent_vector)}, confidence={latent_state.confidence:.2f}")
        
        if encodings:
            all_encodings[track_id] = encodings
    
    logger.info(f"✓ State encoding complete. Encoded {len(all_encodings)} trajectories")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ MINIMAL PIPELINE SUCCESSFUL - REACHED STATE_ENCODER STEP")
    logger.info("=" * 80)
    logger.info(f"  Processed {len(frames)} frames")
    logger.info(f"  Detected {len(trajectories)} tracks")
    logger.info(f"  Encoded {len(all_encodings)} trajectory states")
    
except Exception as e:
    logger.error(f"✗ Error: {e}", exc_info=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
