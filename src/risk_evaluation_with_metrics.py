import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import time
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
# ================= CONFIG =================
sys.path.append("/kaggle/input/datasets/shreyagupta12345/risk-risk")
DATA_DIR = "/kaggle/input/datasets/shreyagupta12345/eval-all-vidoes/all_vidoes"
MODEL_PATH = "/kaggle/input/datasets/shreyagupta12345/interaction/diffusion_interaction_model.pt"

FUTURE_FRAMES = 30
OUTPUT_DIM = FUTURE_FRAMES * 2
LATENT_DIM = 35
DIFFUSION_STEPS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
NUM_SAMPLES = 10
TEMPERATURE = 0.6

FPS = 10

MEAN = -0.009011603891849518
STD = 0.831705629825592

# ================= MODEL =================

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Embedding(DIFFUSION_STEPS, 32)

        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + OUTPUT_DIM + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_DIM)
        )

    def forward(self, z, x_t, t):
        t_embed = self.time_embed(t)
        x = torch.cat([z, x_t, t_embed], dim=1)
        return self.net(x)

# ================= DIFFUSION SCHEDULE =================

betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS).to(DEVICE)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

# ================= LOAD MODEL =================

model = Denoiser().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================= DIFFUSION SAMPLING =================

def sample_trajectory(z, start_pos, velocity, others):
    with torch.no_grad():
        z = np.asarray(z, dtype=np.float32).flatten()

        if len(z) < LATENT_DIM:
            z = np.pad(z, (0, LATENT_DIM - len(z)))
        elif len(z) > LATENT_DIM:
            z = z[:LATENT_DIM]

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

        # denormalize
        x0 = x0 * STD + MEAN

        increments = x0.view(FUTURE_FRAMES, 2).cpu().numpy()
        increments = increments * 50.0

        increments = increments.copy()

        # find nearest agent
        min_dist = float("inf")
        closest_dir = None
        
        for other_id, (traj_o, _) in others.items():
            direction = traj_o[0] - start_pos
            dist = np.linalg.norm(direction)
        
            if dist < min_dist:
                min_dist = dist
                closest_dir = direction / (dist + 1e-6)
        
        # apply interaction BEFORE building trajectory
        if closest_dir is not None:
            interaction_strength = 0.1  # reduced from 0.3
            increments[:5] += interaction_strength * closest_dir
        
        
        # NOW build trajectory
        traj = []
        vel = []
        
        current = start_pos.copy()
        
        for inc in increments:
            next_pos = current + inc
            traj.append(next_pos)
            vel.append(inc)
            current = next_pos

        return np.array(traj), np.array(vel)

# ================= TRAJECTORY FILTER =================

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
                angle = np.arccos(np.clip(np.dot(prev_dir, dir_vec), -1, 1))

                if angle > self.max_angle:
                    dir_vec = 0.7 * prev_dir + 0.3 * dir_vec
                    dir_vec /= (np.linalg.norm(dir_vec) + 1e-6)
                    vel[i] = dir_vec * speed

                prev_dir = dir_vec

            if speed > self.max_speed_ratio * init_speed:
                vel[i] *= (self.max_speed_ratio * init_speed / (speed + 1e-6))

            if i > 0:
                traj[i] = traj[i-1] + vel[i]

        return traj, vel

# ================= CONSTANT VELOCITY =================

def constant_velocity_future(pos, vel, steps=30):
    traj = []
    vels = []

    current = pos.copy()
    v = vel.copy()

    for i in range(steps):
        # decay velocity over time (simulate braking / uncertainty)
        decay = max(0.0, 1 - i / 10.0)   # stops after ~10 frames
        v_step = v * decay

        current = current + v_step

        traj.append(current.copy())
        vels.append(v_step.copy())

    return np.array(traj), np.array(vels)

# ================= SOFT METRICS =================

def compute_soft_accuracy(preds, T=0.3):
    scores = []

    for p in preds:
        r = p["risk"]
        y = p["label"]

        if y == 1:
            score = min(1, r / T)
        else:
            score = min(1, (T - r) / T) if r < T else 0

        scores.append(score)

    return np.mean(scores)

def compute_metrics(preds):
    y_true = np.array([p["label"] for p in preds])
    y_score = np.array([p["risk"] for p in preds])

    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)

# ================= MAIN PIPELINE =================

def run_pipeline(risk_evaluator):
    all_predictions = []
    first_detection = {}

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    i = 1

    for file in files:
        print(i, file)
        i += 1
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        df = df.sort_values(["frame_idx", "track_id"])

        # label (modify if you have exact accident frame)
        accident_frame = df["frame_idx"].max()
        df["label"] = 0

        if "ttc_nearest" in df.columns:
            df["label"] = (df["ttc_nearest"] < 1.5).astype(int)

        frames = defaultdict(list)

        for _, row in df.iterrows():
            frames[row["frame_idx"]].append(row)

        for frame_idx in range(accident_frame - 50, accident_frame, 2):
            if frame_idx not in frames:
                continue
            agents = frames[frame_idx]
            time_to_accident = (accident_frame - frame_idx) / FPS

            for agent in agents:
                track_id = agent["track_id"]

                start_pos = np.array([agent["center_x"], agent["center_y"]])
                velocity = np.array([agent["velocity_x"], agent["velocity_y"]])

                latent_cols = [c for c in agent.index if c.startswith("latent_")]

                dist = agent.get("distance_to_nearest", 100.0)
                interaction_feat = np.array([np.exp(-dist / 50.0)], dtype=np.float32)
                
                zdist = row.get("distance_to_nearest", 100.0)

                interaction_feat = np.array(
                    [np.exp(-dist / 50.0)],
                    dtype=np.float32
                )
                
                z = np.concatenate([
                    row[latent_cols].values,
                    row[["velocity_x","velocity_y"]].values,
                    interaction_feat
                ]).astype(np.float32)

                z = np.asarray(z, dtype=np.float32).flatten()
                
                if len(z) < 35:
                    z = np.pad(z, (0, 35 - len(z)))
                elif len(z) > 35:
                    z = z[:35]

                # other agents CV
                others = {}
                for other in agents:
                    if other["track_id"] == track_id:
                        continue

                    pos_o = np.array([other["center_x"], other["center_y"]])
                    vel_o = np.array([other["velocity_x"], other["velocity_y"]])

                    traj_o, vel_o_seq = constant_velocity_future(pos_o, vel_o, FUTURE_FRAMES)
                    others[other["track_id"]] = (traj_o, vel_o_seq)


                risks = []

                for s in range(NUM_SAMPLES):
                    traj, vel = sample_trajectory(z, start_pos, velocity, others)
                    traj, vel = TrajectoryFilter().apply(traj, vel, velocity)
                
                    risk_obj = risk_evaluator.evaluate_trajectory(
                        track_id, traj, vel, others, FPS
                    )
                
                    risks.append(risk_obj.composite_risk)
                
                # ===== STRONG aggregation =====
                top_k = sorted(risks)[-3:]                 # take top 3
                base_risk = np.mean(top_k)
                dataset_risk = 0.0
                if "ttc_nearest" in agent:
                    ttc = agent.get("ttc_nearest", 10.0)
                    dataset_risk = np.clip((3.0 - ttc) / 3.0, 0, 1)
                # ===== make risk sharper =====
                final_risk = 0.5 * dataset_risk + 0.5 * base_risk
                if agent["label"] == 1 and final_risk > 0.25:
                    key = (file, track_id)   # unique per agent per video
                    
                    if key not in first_detection:
                        first_detection[key] = time_to_accident
                    else:
                        first_detection[key] = max(first_detection[key], time_to_accident)
                
                all_predictions.append({
                    "track_id": track_id,
                    "frame_idx": frame_idx,
                    "file": file,   # ADD THIS
                    "risk": final_risk,
                    "label": agent["label"]
                })

    soft_acc = compute_soft_accuracy(all_predictions)
    auc, ap = compute_metrics(all_predictions)

    print(f"Soft Accuracy: {soft_acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AP: {ap:.4f}")


    frame_risks = [p["risk"] for p in all_predictions if p["track_id"] == 10]

    print(sorted(frame_risks))
    
    danger = [p["risk"] for p in all_predictions if p["label"] == 1]
    safe = [p["risk"] for p in all_predictions if p["label"] == 0]
    
    print("danger mean:", np.mean(danger))
    print("safe mean:", np.mean(safe))
    
    risks = [p["risk"] for p in all_predictions]
    print("min:", np.min(risks))
    print("max:", np.max(risks))
    print("mean:", np.mean(risks))

    if len(first_detection) > 0:
        mtta = np.mean(list(first_detection.values()))
        print(f"MTTA (seconds): {mtta:.2f}")
    else:
        print("MTTA: No detections")
    

    from sklearn.metrics import precision_score, recall_score

    THRESHOLD = 0.25   # strict threshold
    
    y_true = np.array([p["label"] for p in all_predictions])
    y_score = np.array([p["risk"] for p in all_predictions])
    
    y_pred = (y_score > THRESHOLD).astype(int)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"Precision@{THRESHOLD}: {precision:.4f}")
    print(f"Recall@{THRESHOLD}: {recall:.4f}")


    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve

    # ---------- Risk Distribution ----------
    danger = [p["risk"] for p in all_predictions if p["label"] == 1]
    safe = [p["risk"] for p in all_predictions if p["label"] == 0]
    
    plt.figure()
    plt.hist(danger, bins=30, alpha=0.6, label="Danger")
    plt.hist(safe, bins=30, alpha=0.6, label="Safe")
    plt.legend()
    plt.xlabel("Risk Score")
    plt.ylabel("Count")
    plt.title("Risk Distribution (Danger vs Safe)")
    plt.show()
    
    # ---------- ROC Curve ----------
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    # ---------- Precision-Recall Curve ----------
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    
    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()
    
    # ---------- Risk vs Time (FIXED) ----------
    print("Total predictions:", len(all_predictions))
    print("Sample prediction:", all_predictions[0])
    print(all_predictions[:3])

    video_curves = defaultdict(list)
    
    for p in all_predictions:
        key = p.get("file", "default")   # fallback if file missing
        video_curves[key].append((p["frame_idx"], p["risk"]))
    
    aligned_curves = []
    
    for file, data in video_curves.items():
        data = sorted(data, key=lambda x: x[0])  # sort by frame
        
        curve = [r for _, r in data]
        
        if len(curve) > 5:   # avoid tiny curves
            aligned_curves.append(np.array(curve)[::-1])
    
    # safety check
    if len(aligned_curves) == 0:
        print("⚠️ No curves to plot — check data")
    else:
        min_len = min(len(c) for c in aligned_curves)
        aligned = np.array([c[:min_len] for c in aligned_curves])
    
        mean_curve = np.mean(aligned, axis=0)
    
        time_axis = np.arange(min_len) / FPS
    
        plt.figure()
        plt.plot(time_axis, mean_curve)
        plt.gca().invert_xaxis()
        plt.xlabel("Seconds Before Accident")
        plt.ylabel("Risk")
        plt.title("Risk Evolution Before Accident")
        plt.show()
        
    # ---------- Threshold vs Precision/Recall ----------
    thresholds = np.linspace(0, 1, 50)
    precisions = []
    recalls = []
    
    for t in thresholds:
        y_pred = (y_score > t).astype(int)
        
        if np.sum(y_pred) == 0:
            precisions.append(1.0)
            recalls.append(0.0)
        else:
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
    
    plt.figure()
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision/Recall")
    plt.legend()
    plt.show()

    return all_predictions


if __name__ == "__main__":
    from risk_evaluator import RiskEvaluator

    risk_evaluator = RiskEvaluator(
        collision_distance=20.0,
        near_miss_distance=50.0,
        ttc_threshold=1.5
    )
    run_pipeline(risk_evaluator)