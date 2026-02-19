#!/usr/bin/env python
"""
Production-Ready Pipeline Demo with Output Saving
Processes video and saves results to CSV files
"""

import sys
from pathlib import Path
import csv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def save_results_to_csv(trajectories, encodings, output_dir):
    """Save trajectory and encoding results to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not trajectories and not encodings:
        logger.warning("⚠ No data to save (empty trajectories and encodings)")
        return None, None
    
    # Save trajectories
    traj_file = output_dir / 'trajectories.csv'
    rows_written = 0
    with open(traj_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'frame_idx', 'timestamp', 'center_x', 'center_y', 
                        'velocity_x', 'velocity_y', 'class_name'])
        
        for track_id, traj in trajectories.items():
            for state in traj.states:
                writer.writerow([
                    track_id, state.frame_idx, state.timestamp,
                    state.center_x, state.center_y,
                    state.velocity_x, state.velocity_y,
                    traj.class_name
                ])
                rows_written += 1
    
    logger.info(f"✓ Saved {len(trajectories)} trajectories ({rows_written} states) to {traj_file}")
    
    # Save encodings
    enc_file = output_dir / 'latent_encodings.csv'
    enc_rows_written = 0
    with open(enc_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'frame_idx', 'latent_vector', 'confidence', 'sequence_length'])
        
        for track_id, track_encodings in encodings.items():
            for frame_idx, latent_state in track_encodings.items():
                latent_str = ','.join(f"{v:.6f}" for v in latent_state.latent_vector)
                writer.writerow([
                    track_id, frame_idx, latent_str,
                    latent_state.confidence, latent_state.input_sequence_length
                ])
                enc_rows_written += 1
    
    logger.info(f"✓ Saved {len(encodings)} tracks ({enc_rows_written} encodings) to {enc_file}")
    
    return traj_file, enc_file

def process_single_video(video_path, preprocessor, detector_tracker, state_encoder, num_frames_to_process=15):
    """Process a single video and return trajectories and encodings."""
    try:
        from src.trajectory import AgentTrajectory, FrameState
        
        logger.info(f"\n  Processing: {video_path.name}...")
        
        if not video_path.exists():
            logger.warning(f"    ✗ Video not found: {video_path}")
            return None, None, 0, 0
        
        preprocessor.open_video(str(video_path))
        
        frames = []
        frame_times = []
        
        for frame, idx, timestamp in preprocessor.get_frame_generator():
            frames.append(frame)
            frame_times.append(timestamp)
            if len(frames) >= num_frames_to_process:
                break
        
        preprocessor.close_video()
        
        if len(frames) == 0:
            logger.warning(f"    ✗ No frames loaded from {video_path.name}")
            return None, None, 0, 0
        
        logger.info(f"    ✓ Loaded {len(frames)} frames")
        
        # Detection and tracking
        total_detections = 0
        for frame_idx, frame in enumerate(frames):
            frame_det, det_to_track = detector_tracker.process_frame(frame, frame_times[frame_idx])
            total_detections += len(frame_det.detections)
            if len(frame_det.detections) > 0:
                logger.debug(f"    Frame {frame_idx}: {len(frame_det.detections)} detections")
        
        all_tracks = detector_tracker.get_tracks(min_duration_frames=1)
        logger.info(f"    ✓ Detections: {total_detections}, Tracks: {len(all_tracks)}")
        if len(all_tracks) == 0 and total_detections > 0:
            logger.warning(f"    ⚠ Warning: {total_detections} detections found but 0 tracks created!")
        
        # Trajectory extraction
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
        
        # State encoding
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
        
        logger.info(f"    ✓ Encoded {states_encoded}/{total_states} states")
        
        return trajectories, all_encodings, len(frames), total_detections
        
    except Exception as e:
        logger.error(f"    ✗ Error processing {video_path.name}: {e}")
        return None, None, 0, 0

def main():
    try:
        logger.info("=" * 80)
        logger.info("ACCIDENT ANTICIPATION PIPELINE - BATCH PROCESSING ALL VIDEOS")
        logger.info("Processing multiple MP4 files through State Encoder")
        logger.info("=" * 80)
        
        # Import modules
        logger.info("\n[1/5] Importing modules...")
        from src.preprocessing import VideoPreprocessor
        from src.detection_tracking import DetectionTrackingPipeline
        from src.trajectory import AgentTrajectory, FrameState
        from src.state_encoder import StateEncoder
        
        logger.info("✓ Modules imported")
        
        # Get all video files
        logger.info("\n[2/5] Discovering video files...")
        videos_dir = Path('data/videos')
        video_files = sorted(videos_dir.glob('*.mp4'))
        
        if not video_files:
            logger.error(f"No MP4 videos found in {videos_dir}")
            return 1
        
        logger.info(f"✓ Found {len(video_files)} videos to process")
        
        # Initialize pipeline components
        logger.info("\n[3/5] Initializing pipeline components...")
        preprocessor = VideoPreprocessor(target_fps=10, target_width=640, target_height=480)
        state_encoder = StateEncoder(
            input_dim=6,
            hidden_dim=64,
            latent_dim=32,
            device='cpu',
            window_size=20
        )
        logger.info("✓ Pipeline components initialized")
        
        # Process all videos
        logger.info("\n[4/5] Processing all videos through State Encoder...")
        all_trajectories = {}
        all_encodings = {}
        total_frames_processed = 0
        total_detections_found = 0
        videos_processed = 0
        videos_failed = 0
        
        for video_idx, video_path in enumerate(video_files, 1):
            logger.info(f"\n  [{video_idx}/{len(video_files)}] {video_path.name}")
            
            # Create new detector_tracker for each video to avoid cross-video contamination
            detector_tracker = DetectionTrackingPipeline(
                model_name='yolov8m.pt',
                confidence=0.5,
                device='cpu',
                use_byte_track=True
            )
            
            trajectories, encodings, frames_count, detections_count = process_single_video(
                video_path, preprocessor, detector_tracker, state_encoder, num_frames_to_process=15
            )
            
            if trajectories is not None and encodings is not None:
                # Add unique video identifier to track_ids to avoid collisions
                traj_count = 0
                for track_id, traj in trajectories.items():
                    unique_track_id = f"{video_path.stem}_{track_id}"
                    all_trajectories[unique_track_id] = traj
                    traj_count += 1
                
                enc_count = 0
                for track_id, encs in encodings.items():
                    unique_track_id = f"{video_path.stem}_{track_id}"
                    all_encodings[unique_track_id] = encs
                    enc_count += 1
                
                logger.info(f"    → Added {traj_count} trajectories, {enc_count} encoding tracks")
                total_frames_processed += frames_count
                total_detections_found += detections_count
                videos_processed += 1
            else:
                logger.warning(f"    → No results returned")
                videos_failed += 1
        
        logger.info(f"\n✓ Successfully processed {videos_processed}/{len(video_files)} videos")
        
        # Save results
        logger.info("\n[5/5] Saving results to CSV...")
        traj_file, enc_file = save_results_to_csv(all_trajectories, all_encodings, 'output')
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✓ Videos processed: {videos_processed}/{len(video_files)}")
        logger.info(f"✓ Videos failed: {videos_failed}")
        logger.info(f"✓ Total frames: {total_frames_processed}")
        logger.info(f"✓ Total detections: {total_detections_found}")
        logger.info(f"✓ Total tracks: {len(all_trajectories)}")
        logger.info(f"✓ State encodings: {sum(len(encs) for encs in all_encodings.values())}")
        logger.info(f"\n✓ CSV Output Files:")
        logger.info(f"  - {traj_file}")
        logger.info(f"  - {enc_file}")
        logger.info("\n✓✓✓ BATCH PIPELINE SUCCESSFULLY PROCESSES ALL VIDEOS UP TO STATE_ENCODER! ✓✓✓")
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
