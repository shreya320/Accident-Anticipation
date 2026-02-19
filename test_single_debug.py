#!/usr/bin/env python
"""Debug: Test single video in isolation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from src.preprocessing import VideoPreprocessor
from src.detection_tracking import DetectionTrackingPipeline
from src.trajectory import AgentTrajectory, FrameState
from src.state_encoder import StateEncoder

# Load first video
video_path = Path('data/videos/-GpvLzopst8.mp4')
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

# Detection & Tracking
detector_tracker = DetectionTrackingPipeline(
    model_name='yolov8m.pt',
    confidence=0.5,
    device='cpu',
    use_byte_track=False
)

for frame_idx, frame in enumerate(frames):
    frame_det, det_to_track = detector_tracker.process_frame(frame, frame_times[frame_idx])
    logger.debug(f"Frame {frame_idx}: {len(frame_det.detections)} detections")

all_tracks = detector_tracker.get_tracks(min_duration_frames=1)
logger.info(f"Got {len(all_tracks)} tracks")

# Extract trajectories
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

logger.info(f"Created {len(trajectories)} trajectories")

# Encode
state_encoder = StateEncoder(
    input_dim=6,
    hidden_dim=64,
    latent_dim=32,
    device='cpu',
    window_size=20
)

all_encodings = {}
for track_id, trajectory in trajectories.items():
    encodings = {}
    for state in trajectory.states:
        latent_state = state_encoder.encode_trajectory_at_frame(trajectory, state.frame_idx)
        if latent_state is not None:
            encodings[state.frame_idx] = latent_state
    
    if encodings:
        all_encodings[track_id] = encodings
        logger.debug(f"Track {track_id}: {len(encodings)} encodings")

logger.info(f"Encoded {len(all_encodings)} tracks with latent representations")
logger.info(f"Total encoding entries: {sum(len(encs) for encs in all_encodings.values())}")
