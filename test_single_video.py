#!/usr/bin/env python
"""Debug single video processing"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    from src.preprocessing import VideoPreprocessor
    from src.detection_tracking import DetectionTrackingPipeline
    from src.trajectory import AgentTrajectory, FrameState
    
    logger.info("Loading video...")
    video_path = Path('data/videos/08PPPXtzN4A.mp4')
    preprocessor = VideoPreprocessor(target_fps=10, target_width=640, target_height=480)
    preprocessor.open_video(str(video_path))
    
    frames = []
    frame_times = []
    
    for frame, idx, timestamp in preprocessor.get_frame_generator():
        frames.append(frame)
        frame_times.append(timestamp)
        if len(frames) >= 15:
            break
    
    preprocessor.close_video()
    logger.info(f"Loaded {len(frames)} frames")
    
    # Detection and tracking
    logger.info("\nCreating detector...")
    detector_tracker = DetectionTrackingPipeline(
        model_name='yolov8m.pt',
        confidence=0.5,
        device='cpu',
        use_byte_track=False
    )
    
    logger.info("\nProcessing frames...")
    for frame_idx, frame in enumerate(frames):
        logger.info(f"\nFrame {frame_idx}:")
        frame_det, det_to_track = detector_tracker.process_frame(frame, frame_times[frame_idx])
        logger.info(f"  Frame detections: {len(frame_det.detections)}")
        for det in frame_det.detections:
            logger.info(f"    - {det.class_name} (conf={det.confidence:.2f})")
    
    logger.info("\n\nGetting all tracks...")
    all_tracks = detector_tracker.get_tracks(min_duration_frames=1)
    logger.info(f"Total tracks: {len(all_tracks)}")
    
    for track_id, track in all_tracks.items():
        logger.info(f"  Track {track_id}: {track.class_name}, {len(track.detections)} detections")
        logger.info(f"    Duration: {track.duration_frames()} frames")
    
    # Trajectory extraction
    logger.info("\nExtracting trajectories...")
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
    
    logger.info(f"Trajectories: {len(trajectories)}")
    for track_id, traj in trajectories.items():
        logger.info(f"  Track {track_id}: {len(traj.states)} states")

if __name__ == '__main__':
    main()
