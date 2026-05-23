import sys
import os
import math
import glob
from pathlib import Path
import logging
import csv
from datetime import datetime

# ---------------- CONFIG ----------------

# Generic local paths - adjust as needed for your environment
SHARED_BASE = Path(".")

VIDEO_DIR = Path("data/videos")  # Local video directory
TIMESTAMP_FILE = Path("data/duration_list.txt")  # Local timestamp file

OUT_DIR = Path("evaluation_results_b")  # Local output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add current directory to path for local imports
sys.path.insert(0, ".")

from .preprocessing import VideoPreprocessor
from .detection_tracking import DetectionTrackingPipeline
from .trajectory import FrameState, AgentTrajectory
from .interaction_features import InteractionFeatureComputer
from state_encoder import StateEncoder

CONFIG = {
    "target_fps": 10,
    "frame_width": 640,
    "frame_height": 480,
    "detection_confidence": 0.5,
    "latent_dim": 32,
    "observation_window": 20,
    "prediction_horizon": 20,
    "use_cuda": True
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("encode_states")

# ---------------- UTILITIES ----------------

def parse_timestamp_to_seconds(ts):
    h, m, s = ts.split(":")
    return int(h)*3600 + int(m)*60 + float(s)

def load_accident_timestamps(file_path):
    accident_times = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            name, time_str = line.strip().split()
            accident_times[name] = parse_timestamp_to_seconds(time_str)
    return accident_times

ACCIDENT_TIMES = load_accident_timestamps(TIMESTAMP_FILE)

def collect_videos(base_dir):
    return sorted(base_dir.rglob("*.mp4"))

def already_processed(video_name):
    existing = list(OUT_DIR.glob(f"{video_name}_processed.flag"))
    return len(existing) > 0

# ---------------- CORE PROCESS ----------------

def process_video(video_path):

    video_name = video_path.name

    if video_name not in ACCIDENT_TIMES:
        logger.warning(f"Skipping {video_name} (no timestamp)")
        return

    if already_processed(video_name):
        logger.info(f"Skipping {video_name} (already processed)")
        return

    logger.info(f"Processing {video_name}")

    accident_time = ACCIDENT_TIMES[video_name]
    accident_frame = int(accident_time * CONFIG["target_fps"])

    vp = VideoPreprocessor(CONFIG["target_fps"],
                           CONFIG["frame_width"],
                           CONFIG["frame_height"])

    if not vp.open_video(str(video_path)):
        return

    device = "cuda:0" if CONFIG["use_cuda"] else "cpu"

    pipeline = DetectionTrackingPipeline(
        confidence=CONFIG["detection_confidence"],
        device=device
    )

    frame_times = []

    for frame, idx, timestamp in vp.get_frame_generator():
        frame_times.append(timestamp)
        pipeline.process_frame(frame, timestamp)

    vp.close_video()

    tracks = pipeline.get_tracks(min_duration_frames=1)

    trajectories = {}

    for track_id, track in tracks.items():

        states = []
        prev = None

        for (frame_idx, det) in track.detections:

            t = frame_times[frame_idx]
            cx, cy = det.center

            if prev is None:
                vx = vy = ax = ay = 0
            else:
                dt = max(t - prev["t"], 1e-6)
                vx = (cx - prev["cx"]) / dt
                vy = (cy - prev["cy"]) / dt
                ax = (vx - prev["vx"]) / dt
                ay = (vy - prev["vy"]) / dt

            state = FrameState(
                frame_idx=frame_idx,
                timestamp=t,
                center_x=cx,
                center_y=cy,
                velocity_x=vx,
                velocity_y=vy,
                speed=math.hypot(vx, vy),
                acceleration_x=ax,
                acceleration_y=ay,
                acceleration_mag=math.hypot(ax, ay),
                heading_angle=math.atan2(vy, vx) if abs(vx)+abs(vy) > 1e-6 else 0
            )

            states.append(state)
            prev = {"cx":cx,"cy":cy,"vx":vx,"vy":vy,"t":t}

        if states:
            trajectories[track_id] = AgentTrajectory(
                track_id=track_id,
                class_name=track.class_name,
                class_id=track.class_id,
                states=states
            )

    interaction_comp = InteractionFeatureComputer()
    interactions = interaction_comp.compute_trajectory_interactions(trajectories)

    encoder = StateEncoder(
        latent_dim=CONFIG["latent_dim"],
        device=device,
        window_size=CONFIG["observation_window"]
    )

    latent_all = encoder.encode_all_trajectories(trajectories)

    out_csv = OUT_DIR / f"{video_path.stem}.csv"

    latent_dim = CONFIG["latent_dim"]

    header = [
        "video","track_id","class_id","frame_idx","timestamp",
        "center_x","center_y",
        "velocity_x","velocity_y","speed",
        "acceleration_x","acceleration_y",
        "distance_to_nearest",
        "num_nearby_agents",
        "relative_velocity_x",
        "relative_velocity_y",
        "relative_velocity_mag",
        "relative_heading",
        "ttc_nearest",
        "collision_indicator",
        "intersection_risk",
        "concurrent_agents",
        "accident_time_sec","accident_frame","label"
    ] + [f"latent_{i}" for i in range(latent_dim)]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for track_id, traj in trajectories.items():
            for state in traj.states:

                if state.frame_idx < accident_frame - CONFIG["prediction_horizon"]:
                    label = 0
                elif accident_frame - CONFIG["prediction_horizon"] <= state.frame_idx < accident_frame:
                    label = 1
                else:
                    continue

                latent_state = latent_all.get(track_id, {}).get(state.frame_idx)
                latent_vec = latent_state.latent_vector.tolist() if latent_state else [0]*latent_dim

                inter = interactions.get(track_id, {}).get(state.frame_idx)
                interaction_vals = [
                    inter.distance_to_nearest,
                    inter.num_nearby_agents,
                    inter.relative_velocity_x,
                    inter.relative_velocity_y,
                    inter.relative_velocity_mag,
                    inter.relative_heading,
                    inter.ttc_nearest,
                    inter.collision_indicator,
                    inter.intersection_risk,
                    inter.concurrent_agents
                ] if inter else [0]*10

                row = [
                    video_name, track_id, traj.class_id,
                    state.frame_idx, state.timestamp,
                    state.center_x, state.center_y,
                    state.velocity_x, state.velocity_y, state.speed,
                    state.acceleration_x, state.acceleration_y
                ] + interaction_vals + [
                    accident_time, accident_frame, label
                ] + latent_vec

                writer.writerow(row)

    # Mark as processed
    (OUT_DIR / f"{video_name}_processed.flag").touch()

    logger.info(f"Finished {video_name}")

# ---------------- MAIN ----------------

videos = collect_videos(VIDEO_DIR)

for v in videos:
    process_video(v)

print("ALL DONE")