# ==========================================
# KAGGLE SINGLE VIDEO DEMO NOTEBOOK
# CONSISTENT WITH FINAL EVALUATION PIPELINE
# ==========================================

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PATHS
# ==========================================

sys.path.append("")
sys.path.append("")

MODEL_PATH = ""

# ==========================================
# USER INPUT
# ==========================================

VIDEO_FILE = "/kaggle/input/datasets/shreyagupta12345/video-update/zVzXEht1aME.mp4"
START_SEC = 30
END_SEC   = 39
TARGET_FPS = 10

# ==========================================
# IMPORT YOUR MODULES
# ==========================================

from preprocessing import VideoPreprocessor
from detection_tracking import DetectionTrackingPipeline
from trajectory import FrameState, AgentTrajectory
from interaction_features import InteractionFeatureComputer
from state_encoder import StateEncoder
from risk_evaluator import RiskEvaluator

# ==========================================
# DEVICE
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)

# ==========================================
# CONFIG (MATCH FINAL NOTEBOOK)
# ==========================================

FUTURE_FRAMES   = 30
OUTPUT_DIM      = FUTURE_FRAMES * 2
DIFFUSION_STEPS = 50
NUM_SAMPLES     = 10
TEMPERATURE     = 0.6

MEAN = -0.009011603891849518
STD  = 0.831705629825592

# ==========================================
# MODEL
# ==========================================

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embed = nn.Embedding(DIFFUSION_STEPS, 32)

        self.net = nn.Sequential(
            nn.Linear(127, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_DIM)
        )

    def forward(self, z, x_t, t):
        t_embed = self.time_embed(t)
        x = torch.cat([z, x_t, t_embed], dim=1)
        return self.net(x)

betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS).to(DEVICE)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

model = Denoiser().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Diffusion model loaded")

# ==========================================
# FUNCTIONS (MATCH FINAL NOTEBOOK)
# ==========================================

def sample_trajectory(z, start_pos, velocity, others):

    with torch.no_grad():

        z = torch.tensor(z, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        x_t = torch.randn((1, OUTPUT_DIM), device=DEVICE) * TEMPERATURE

        for t in reversed(range(DIFFUSION_STEPS)):

            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)

            noise_pred = model(z, x_t, t_tensor)

            alpha = alphas[t]
            alpha_bar = alpha_cumprod[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x_t) * TEMPERATURE
            else:
                noise = torch.zeros_like(x_t)

            x_t = (
                (1 / torch.sqrt(alpha)) *
                (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_pred)
                + torch.sqrt(beta) * noise
            )

        x0 = x_t.squeeze(0)
        x0 = x0 * STD + MEAN

        increments = x0.view(FUTURE_FRAMES, 2).cpu().numpy()
        increments = increments * 50.0

        min_dist = float("inf")
        closest_dir = None

        for _, (traj_o, _) in others.items():

            direction = traj_o[0] - start_pos
            dist = np.linalg.norm(direction)

            if dist < min_dist:
                min_dist = dist
                closest_dir = direction / (dist + 1e-6)

        if closest_dir is not None:
            increments[:5] += 0.1 * closest_dir

        traj = []
        vel = []

        current = start_pos.copy()

        for inc in increments:
            current = current + inc
            traj.append(current.copy())
            vel.append(inc.copy())

        return np.array(traj), np.array(vel)

# ------------------------------------------

class TrajectoryFilter:
    def __init__(self, max_angle=25, max_speed_ratio=2.0):
        self.max_angle = np.deg2rad(max_angle)
        self.max_speed_ratio = max_speed_ratio

    def apply(self, traj, vel, init_vel):

        traj = traj.copy()
        vel = vel.copy()

        prev_dir = init_vel / (np.linalg.norm(init_vel) + 1e-6)
        init_speed = np.linalg.norm(init_vel)

        for i in range(len(vel)):

            v = vel[i]
            speed = np.linalg.norm(v)

            if speed > 1e-6:

                dir_vec = v / speed

                angle = np.arccos(
                    np.clip(np.dot(prev_dir, dir_vec), -1, 1)
                )

                if angle > self.max_angle:
                    dir_vec = 0.7 * prev_dir + 0.3 * dir_vec
                    dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
                    vel[i] = dir_vec * speed

                prev_dir = dir_vec

            if speed > self.max_speed_ratio * init_speed:
                vel[i] *= (
                    self.max_speed_ratio * init_speed /
                    (speed + 1e-6)
                )

            if i > 0:
                traj[i] = traj[i-1] + vel[i]

        return traj, vel

# ------------------------------------------

def constant_velocity_future(pos, vel, steps=30):

    traj = []
    vels = []

    current = pos.copy()

    for i in range(steps):

        decay = max(0.0, 1 - i / 10.0)
        v_step = vel * decay

        current = current + v_step

        traj.append(current.copy())
        vels.append(v_step.copy())

    return np.array(traj), np.array(vels)

# ==========================================
# STEP 1 : VIDEO PROCESSING
# ==========================================

vp = VideoPreprocessor(
    target_fps=TARGET_FPS,
    target_width=640,
    target_height=480
)

vp.open_video(VIDEO_FILE)

pipeline = DetectionTrackingPipeline(
    confidence=0.5,
    device=DEVICE
)

frames_used = []

for frame, idx, timestamp in vp.get_frame_generator():

    if timestamp < START_SEC:
        continue

    if timestamp > END_SEC:
        break

    pipeline.process_frame(frame, timestamp)
    frames_used.append((frame, idx, timestamp))

vp.close_video()

print("Frames processed:", len(frames_used))

# ==========================================
# STEP 2 : TRACKS
# ==========================================

tracks = pipeline.get_tracks(min_duration_frames=3)

print("Tracks detected:", len(tracks))

# ==========================================
# STEP 3 : BUILD TRAJECTORIES
# ==========================================

trajectories = {}

for track_id, track in tracks.items():

    states = []
    prev = None

    for frame_idx, det in track.detections:

        cx, cy = det.center
        t = frame_idx / TARGET_FPS

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
            speed=np.hypot(vx, vy),
            acceleration_x=ax,
            acceleration_y=ay,
            acceleration_mag=np.hypot(ax, ay),
            heading_angle=np.arctan2(vy, vx) if abs(vx)+abs(vy)>1e-6 else 0
        )

        states.append(state)

        prev = {
            "cx": cx,
            "cy": cy,
            "vx": vx,
            "vy": vy,
            "t": t
        }

    if len(states) > 0:
        trajectories[track_id] = AgentTrajectory(
            track_id=track_id,
            class_name=track.class_name,
            class_id=track.class_id,
            states=states
        )

print("Trajectories created:", len(trajectories))

# ==========================================
# STEP 4 : INTERACTION FEATURES
# ==========================================

interaction_comp = InteractionFeatureComputer()
interactions = interaction_comp.compute_trajectory_interactions(trajectories)

# ==========================================
# STEP 5 : LATENT STATES
# ==========================================

encoder = StateEncoder(
    latent_dim=32,
    device=DEVICE,
    window_size=20
)

latent_all = encoder.encode_all_trajectories(trajectories)

# ==========================================
# STEP 6 : REAL RISK EVALUATION
# ==========================================

risk_evaluator = RiskEvaluator(
    collision_distance=20.0,
    near_miss_distance=50.0,
    ttc_threshold=1.5
)

risk_scores = []

VALID_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
MIN_TRACK_LEN = 5

for track_id, traj in trajectories.items():

    if traj.class_name.lower() not in VALID_CLASSES:
        continue

    if len(traj.states) < MIN_TRACK_LEN:
        continue

    latest = traj.states[-1]

    start_pos = np.array([latest.center_x, latest.center_y])
    velocity  = np.array([latest.velocity_x, latest.velocity_y])

    latent_state = latent_all.get(track_id, {}).get(latest.frame_idx, None)

    if latent_state is None:
        continue

    latent = latent_state.latent_vector

    inter = interactions.get(track_id, {}).get(latest.frame_idx, None)

    dist = 100.0 if inter is None else inter.distance_to_nearest
    ttc  = 10.0 if inter is None else inter.ttc_nearest

    interaction_feat = np.array(
        [np.exp(-dist / 50.0)],
        dtype=np.float32
    )

    z = np.concatenate([
        latent,
        velocity,
        interaction_feat
    ]).astype(np.float32)

    # other agents
    others = {}

    for other_id, other_traj in trajectories.items():

        if other_id == track_id:
            continue

        other_latest = other_traj.states[-1]

        pos_o = np.array([
            other_latest.center_x,
            other_latest.center_y
        ])

        vel_o = np.array([
            other_latest.velocity_x,
            other_latest.velocity_y
        ])

        traj_o, vel_o_seq = constant_velocity_future(
            pos_o,
            vel_o,
            FUTURE_FRAMES
        )

        others[other_id] = (traj_o, vel_o_seq)

    # multi-sample risks
    risks = []

    for _ in range(NUM_SAMPLES):

        pred_traj, pred_vel = sample_trajectory(
            z,
            start_pos,
            velocity,
            others
        )

        pred_traj, pred_vel = TrajectoryFilter().apply(
            pred_traj,
            pred_vel,
            velocity
        )

        risk_obj = risk_evaluator.evaluate_trajectory(
            track_id,
            pred_traj,
            pred_vel,
            others,
            TARGET_FPS
        )

        risks.append(risk_obj.composite_risk)

    # MATCH FINAL EVALUATION LOGIC
    top_k = sorted(risks)[-3:]
    base_risk = np.mean(top_k)
    
    # softer distance gate
    distance_gate = 0.55 + 0.45 * np.exp(-dist / 175.0)
    
    # TTC if reasonably near
    if dist < 160:
        dataset_risk = np.clip((3.0 - ttc) / 3.0, 0, 1)
    else:
        dataset_risk = 0.0
    
    final_risk = (0.75 * base_risk + 0.25 * dataset_risk) * distance_gate
    risk_scores.append(
        (
            track_id,
            final_risk,
            latest.center_x,
            latest.center_y
        )
    )

risk_dict = {tid:[risk,x,y] for tid,risk,x,y in risk_scores}

ids = list(risk_dict.keys())

for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        a = ids[i]
        b = ids[j]

        ra, xa, ya = risk_dict[a]
        rb, xb, yb = risk_dict[b]

        d = np.linalg.norm([xa-xb, ya-yb])

        if d < 70:   # nearby interacting pair
            shared = max(ra, rb)
            risk_dict[a][0] = shared
            risk_dict[b][0] = shared

risk_scores = [(tid,v[0],v[1],v[2]) for tid,v in risk_dict.items()]

# ==========================================
# STEP 7 : SORT RESULTS
# ==========================================
merged = []
used = set()

for i,a in enumerate(risk_scores):
    if i in used:
        continue

    tid,r,x,y = a

    for j,b in enumerate(risk_scores[i+1:], start=i+1):
        tid2,r2,x2,y2 = b

        if np.linalg.norm([x-x2,y-y2]) < 35:
            r = max(r,r2)
            used.add(j)

    merged.append((tid,r,x,y))

risk_scores = merged

risk_scores = sorted(
    risk_scores,
    key=lambda x: x[1],
    reverse=True
)

print("\nTop Risky Agents:")
for r in risk_scores[:5]:
    print("Track:", r[0], "| Risk:", round(r[1], 3))

# ==========================================
# STEP 8 : VISUALIZE
# ==========================================

frame = frames_used[-10][0].copy()

plt.figure(figsize=(12,7))
plt.imshow(frame[:,:,::-1])

for tid, risk, x, y in risk_scores[:5]:
    plt.scatter(x, y, s=90)
    plt.text(
        x+5,
        y+5,
        f"ID {tid} | Risk {risk:.2f}",
        fontsize=10
    )

plt.title("Detected High Risk Agents")
plt.axis("off")
plt.show()

# ==========================================
# STEP 9 : BAR GRAPH
# ==========================================

ids = [str(x[0]) for x in risk_scores[:5]]
vals = [x[1] for x in risk_scores[:5]]

plt.figure(figsize=(10,5))
plt.bar(ids, vals)
plt.title("Top Agent Risk Scores")
plt.xlabel("Track ID")
plt.ylabel("Risk Score")
plt.show()

# ==========================================
# FINAL MESSAGE
# ==========================================

if len(risk_scores) > 0 and risk_scores[0][1] > 0.25:
    print("WARNING: Elevated collision risk detected.")
else:
    print("Low / Moderate Risk Scene.")