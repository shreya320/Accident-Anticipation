#!/usr/bin/env python
"""
Pipeline runner that processes video and saves state encoder outputs to CSV.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def process_video(video_path: Path, output_dir: Path, num_frames: int, device: str = 'cpu') -> tuple:
    """Process a single video and save state encodings + summary CSVs.

    Returns tuple(csv_path, summary_path).
    """
    from src.preprocessing import VideoPreprocessor
    from src.detection_tracking import DetectionTrackingPipeline
    from src.trajectory import AgentTrajectory, FrameState
    from src.state_encoder import StateEncoder

    logger.info(f"\nProcessing video: {video_path.name}")

    preprocessor = VideoPreprocessor(target_fps=10, target_width=640, target_height=480)
    preprocessor.open_video(str(video_path))

    frames = []
    frame_times = []
    for frame, idx, timestamp in preprocessor.get_frame_generator():
        frames.append(frame)
        frame_times.append(timestamp)
        if len(frames) >= num_frames:
            break

    preprocessor.close_video()
    logger.info(f"  ✓ Loaded {len(frames)} frames")

    detector_tracker = DetectionTrackingPipeline(
        model_name='yolov8m.pt',
        confidence=0.5,
        device=device,
        use_byte_track=False
    )

    for frame_idx, frame in enumerate(frames):
        frame_det, det_to_track = detector_tracker.process_frame(frame, frame_times[frame_idx])

    all_tracks = detector_tracker.get_tracks(min_duration_frames=1)
    logger.info(f"  ✓ Detected and tracked {len(all_tracks)} objects")

    # Extract trajectories
    trajectories = {}
    for track_id, track in all_tracks.items():
        traj = AgentTrajectory(track_id=track_id, class_name=track.class_name, class_id=track.class_id)
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

    logger.info(f"  ✓ Extracted {len(trajectories)} trajectories")

    # Encode
    state_encoder = StateEncoder(input_dim=6, hidden_dim=64, latent_dim=32, device=device, window_size=20)
    all_encodings = {}
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

    logger.info(f"  ✓ Encoded {states_encoded} states")

    # Save CSVs
    all_data = []
    for track_id, trajectory in trajectories.items():
        for state in trajectory.states:
            frame_idx = state.frame_idx
            if track_id in all_encodings and frame_idx in all_encodings[track_id]:
                latent_state = all_encodings[track_id][frame_idx]
                latent_vector = latent_state.latent_vector
                row = {
                    'video': video_path.name,
                    'track_id': track_id,
                    'class_name': trajectory.class_name,
                    'class_id': trajectory.class_id,
                    'frame_idx': frame_idx,
                    'timestamp': state.timestamp,
                    'center_x': state.center_x,
                    'center_y': state.center_y,
                    'velocity_x': state.velocity_x,
                    'velocity_y': state.velocity_y,
                    'speed': state.speed,
                    'acceleration_x': state.acceleration_x,
                    'acceleration_y': state.acceleration_y,
                    'sequence_length': latent_state.input_sequence_length,
                    'confidence': latent_state.confidence,
                }
                for i, val in enumerate(latent_vector):
                    row[f'latent_{i}'] = val
                all_data.append(row)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = output_dir / f"state_encodings_{video_path.stem}_{timestamp}.csv"
    pd.DataFrame(all_data).to_csv(csv_filename, index=False)

    summary_data = []
    for track_id, trajectory in trajectories.items():
        summary_data.append({
            'video': video_path.name,
            'track_id': track_id,
            'class_name': trajectory.class_name,
            'class_id': trajectory.class_id,
            'num_frames': len(trajectory.states),
            'num_encoded': len(all_encodings.get(track_id, {})),
            'encoding_rate': len(all_encodings.get(track_id, {})) / len(trajectory.states) if trajectory.states else 0
        })
    summary_filename = output_dir / f"encoding_summary_{video_path.stem}_{timestamp}.csv"
    pd.DataFrame(summary_data).to_csv(summary_filename, index=False)

    logger.info(f"  ✓ Saved {len(all_data)} encodings to {csv_filename.name}")
    logger.info(f"  ✓ Saved summary to {summary_filename.name}")

    return str(csv_filename), str(summary_filename)


def main():
    parser = argparse.ArgumentParser(description='Process all videos and save state encodings to CSV')
    parser.add_argument('--videos-dir', type=str, default='data/videos', help='Directory with videos')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save CSVs')
    parser.add_argument('--num-frames', type=int, default=100, help='Number of frames to process per video')
    parser.add_argument('--max-videos', type=int, default=0, help='Maximum number of videos to process (0 = all)')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model inference')
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    video_files = sorted(videos_dir.glob('*.mp4'))
    if not video_files:
        logger.error(f"No MP4 videos found in {videos_dir}")
        return 1

    if args.max_videos > 0:
        video_files = video_files[:args.max_videos]

    created = []
    for vid in video_files:
        try:
            csv_path, summary_path = process_video(vid, output_dir, args.num_frames, device=args.device)
            created.append((vid.name, csv_path, summary_path))
        except Exception as e:
            logger.error(f"Error processing {vid.name}: {e}", exc_info=True)

    # Master index
    if created:
        index_rows = [{'video': v, 'csv': c, 'summary': s} for v, c, s in created]
        index_df = pd.DataFrame(index_rows)
        index_file = output_dir / f'master_index_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        index_df.to_csv(index_file, index=False)
        logger.info(f"\nDone. Created {len(created)} output pairs. Master index: {index_file.name}")
    else:
        logger.info("\nDone. No outputs created.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
