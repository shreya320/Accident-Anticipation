# ================= CONTINUAL LEARNING =================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.append("/kaggle/input/datasets/shreyagupta12345/all-files-alls")

from sklearn.metrics import roc_auc_score, average_precision_score

from risk_evaluator import RiskEvaluator
from risk_aggregation import RiskAggregationPipeline

from preprocessing import VideoPreprocessor
from detection_tracking import DetectionTrackingPipeline
from trajectory import FrameState, AgentTrajectory
from interaction_features import InteractionFeatureComputer
from state_encoder import StateEncoder

# ================= CONFIG =================

DATA_DIR = "/kaggle/input/datasets/shreyagupta12345/eval-all-vidoes/all_vidoes"
MODEL_PATH = "/kaggle/input/datasets/shreyagupta12345/interaction/diffusion_interaction_model.pt"
NEW_VIDEO_PATH = "/kaggle/input/datasets/shreyagupta12345/video-update/zVzXEht1aME.mp4"

FUTURE_FRAMES = 30
OUTPUT_DIM = FUTURE_FRAMES * 2
LATENT_DIM = 35
DIFFUSION_STEPS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = 10
TEMPERATURE = 0.5

FPS = 10
RISK_THRESHOLD = 0.05

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
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_DIM)
        )

    def forward(self,z,x_t,t):

        t_embed = self.time_embed(t)
        x = torch.cat([z,x_t,t_embed],dim=1)

        return self.net(x)

model = Denoiser().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
model.eval()

# ================= DIFFUSION =================

betas = torch.linspace(1e-4,0.02,DIFFUSION_STEPS).to(DEVICE)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas,dim=0)

def sample_trajectory(z,start_pos):

    x_t = torch.randn(1,OUTPUT_DIM,device=DEVICE)

    for t in reversed(range(DIFFUSION_STEPS)):

        t_tensor = torch.tensor([t],device=DEVICE)

        noise_pred = model(z,x_t,t_tensor)

        alpha = alphas[t]
        alpha_bar = alpha_cumprod[t]
        beta = betas[t]

        posterior_mean = (1/torch.sqrt(alpha)) * (
            x_t - (beta/torch.sqrt(1-alpha_bar))*noise_pred
        )

        if t>0:
            x_t = posterior_mean + TEMPERATURE * torch.sqrt(beta) * torch.randn_like(x_t)
        else:
            x_t = posterior_mean

    traj = x_t.detach().cpu().numpy().reshape(FUTURE_FRAMES,2)

    traj = traj * STD + MEAN
    traj = traj * 50.0

    pos = start_pos.astype(np.float32)

    positions=[]
    velocities=[]

    for step in traj:

        new_pos = pos + step
        vel = new_pos - pos

        positions.append(new_pos.copy())
        velocities.append(vel.copy())

        pos = new_pos

    return np.array(positions), np.array(velocities)

# ================= RISK PIPELINE =================

risk_evaluator = RiskEvaluator()
aggregation_pipeline = RiskAggregationPipeline()

# ================= DATASET EVALUATION =================

def evaluate_dataset():

    print("\nEvaluating existing dataset...\n")

    all_risks=[]
    all_labels=[]
    tta_list=[]

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    files = [files[1]]

    for idx,file in enumerate(files):

        print(f"Processing dataset video {idx+1}/{len(files)} : {file}")

        df = pd.read_csv(os.path.join(DATA_DIR,file))
        df = df.sort_values(["track_id","frame_idx"])

        latent_cols = [c for c in df.columns if c.startswith("latent_")]

        accident_frame = df["frame_idx"].max()
        positive_start = accident_frame - FUTURE_FRAMES

        track_ids = df.track_id.unique()

        prediction_frame = None
        risk_history = []

        for frame in range(accident_frame-50, accident_frame-5, 5):

            future_trajectories={}
            other_trajectories={}

            for track_id in track_ids:

                g = df[df.track_id==track_id]

                if frame not in g.frame_idx.values:
                    continue

                row = g[g.frame_idx==frame].iloc[0]

                dist = row.get("distance_to_nearest", 100.0)

                interaction_feat = np.array(
                    [np.exp(-dist / 50.0)],
                    dtype=np.float32
                )
                
                z = np.concatenate([
                    row[latent_cols].values,
                    row[["velocity_x","velocity_y"]].values,
                    interaction_feat
                ]).astype(np.float32)

                start_pos = row[["center_x","center_y"]].values.astype(np.float32)

                z = torch.tensor(z).unsqueeze(0).to(DEVICE)

                samples=[]

                for s in range(NUM_SAMPLES):

                    traj,vel = sample_trajectory(z,start_pos)

                    samples.append(type("Obj",(),{
                        "trajectory":traj,
                        "velocities":vel,
                        "sample_idx":s
                    }))

                future_trajectories[track_id]=samples

                pos = row[["center_x","center_y"]].values.astype(np.float32)
                vel = row[["velocity_x","velocity_y"]].values.astype(np.float32)

                traj=[]
                vels=[]

                p = pos.copy()

                for _ in range(FUTURE_FRAMES):
                    p = p + vel
                    traj.append(p.copy())
                    vels.append(vel.copy())

                other_trajectories[track_id]=(np.array(traj),np.array(vels))

            risk_scores = risk_evaluator.batch_evaluate(
                future_trajectories,
                other_trajectories
            )

            frame_risk=[]

            for track_id,scores in risk_scores.items():

                agg = aggregation_pipeline.aggregator.aggregate_all_risks(scores)
                frame_risk.append(agg.mean_composite_risk)

            frame_risk = [r for r in frame_risk if r > 0.1]

            if len(frame_risk)==0:
                risk = 0.0
            else:
                frame_risk = sorted(frame_risk,reverse=True)

                base_risk = np.percentile(frame_risk,90)

                risk_history.append(base_risk)

                risk = np.mean(risk_history[-5:])
                risk = 1/(1+np.exp(-10*(risk-0.2)))

            print("Frame",frame,"risk:",round(risk,3))

            label = 1 if positive_start<=frame<=accident_frame else 0

            all_risks.append(risk)
            all_labels.append(label)

            if prediction_frame is None and risk > RISK_THRESHOLD:
                prediction_frame = frame

        if prediction_frame is not None:
            tta_list.append(accident_frame-prediction_frame)

    auc = roc_auc_score(all_labels,all_risks)
    ap = average_precision_score(all_labels,all_risks)
    mtta = np.mean(tta_list) if len(tta_list)>0 else 0

    print("\nDataset metrics:")
    print("AUC:",auc)
    print("AP:",ap)
    print("mTTA:",mtta/FPS)

    return auc,ap,mtta

# ================= VIDEO ENCODING =================

def encode_video(video_path):

    print("\nEncoding new video:",video_path)

    vp = VideoPreprocessor(10,640,480)
    vp.open_video(video_path)

    pipeline = DetectionTrackingPipeline()

    frame_times=[]

    for frame,idx,timestamp in vp.get_frame_generator():

        frame_times.append(timestamp)
        pipeline.process_frame(frame,timestamp)

    vp.close_video()

    tracks = pipeline.get_tracks()
    trajectories={}

    for track_id,track in tracks.items():

        states=[]
        prev=None

        for frame_idx,det in track.detections:

            cx,cy = det.center

            if prev is None:
                vx=vy=0
            else:
                vx=cx-prev[0]
                vy=cy-prev[1]

            prev=(cx,cy)

            ts = frame_times[frame_idx] if frame_idx < len(frame_times) else frame_idx/FPS

            states.append(FrameState(
                frame_idx=frame_idx,
                timestamp=ts,
                center_x=cx,
                center_y=cy,
                velocity_x=vx,
                velocity_y=vy,
                speed=np.hypot(vx,vy),
                acceleration_x=0,
                acceleration_y=0,
                acceleration_mag=0,
                heading_angle=0
            ))

        trajectories[track_id]=AgentTrajectory(
            track_id=track_id,
            class_name="vehicle",
            class_id=0,
            states=states
        )

    encoder = StateEncoder(
        latent_dim=32,
        device=DEVICE,
        window_size=20
    )

    latent_all = encoder.encode_all_trajectories(trajectories)

    rows=[]

    for track_id,traj in trajectories.items():

        for state in traj.states:

            latent = latent_all.get(track_id,{}).get(state.frame_idx)

            if latent is None:
                continue

            row = {
                "track_id":track_id,
                "frame_idx":state.frame_idx,
                "center_x":state.center_x,
                "center_y":state.center_y,
                "velocity_x":state.velocity_x,
                "velocity_y":state.velocity_y
            }

            for i,v in enumerate(latent.latent_vector):
                row[f"latent_{i}"]=v

            rows.append(row)

    df = pd.DataFrame(rows)

    print("Encoded rows:",len(df))
    return df

# ================= PREDICT RISK =================

def predict_risk(df):

    latent_cols=[c for c in df.columns if c.startswith("latent_")]

    risk_values=[]
    track_ids=df.track_id.unique()

    for frame in sorted(df.frame_idx.unique())[-10:]:

        future={}
        other={}

        for track_id in track_ids:

            g=df[df.track_id==track_id]

            if frame not in g.frame_idx.values:
                continue

            row=g[g.frame_idx==frame].iloc[0]

            z=np.concatenate([
                row[latent_cols].values,
                row[["velocity_x","velocity_y"]].values
            ]).astype(np.float32)

            start_pos=row[["center_x","center_y"]].values.astype(np.float32)

            z=torch.tensor(z).unsqueeze(0).to(DEVICE)

            samples=[]

            for s in range(NUM_SAMPLES):

                traj,vel = sample_trajectory(z,start_pos)

                samples.append(type("Obj",(),{
                    "trajectory":traj,
                    "velocities":vel,
                    "sample_idx":s
                }))

            future[track_id]=samples
            other[track_id]=(traj,vel)

        scores=risk_evaluator.batch_evaluate(future,other)

        frame_risk=[]

        for track_id,sc in scores.items():

            agg=aggregation_pipeline.aggregator.aggregate_all_risks(sc)
            frame_risk.append(agg.mean_composite_risk)

        if len(frame_risk)>0:

            risk=np.mean(frame_risk)
            risk_values.append(risk)

            print("Frame",frame,"risk:",risk)

    avg_risk=np.mean(risk_values) if len(risk_values)>0 else 0

    print("\nPredicted accident probability:",avg_risk)

    if avg_risk > RISK_THRESHOLD:
        print("\n⚠ ACCIDENT WARNING")

    return avg_risk

# ================= CONTINUAL UPDATE =================

def continual_update(df):

    print("\nRunning continual learning update...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    latent_cols=[c for c in df.columns if c.startswith("latent_")]

    model.train()

    for _,row in df.tail(20).iterrows():

        z=np.concatenate([
            row[latent_cols].values,
            row[["velocity_x","velocity_y"]].values
        ]).astype(np.float32)

        z=torch.tensor(z).unsqueeze(0).to(DEVICE)

        x=torch.randn(1,OUTPUT_DIM,device=DEVICE)
        t=torch.randint(0,DIFFUSION_STEPS,(1,),device=DEVICE)

        pred=model(z,x,t)

        loss=(pred**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    print("Model updated successfully.")

# ================= RUN PIPELINE =================

print("\n===== BASELINE DATASET METRICS =====")
auc,ap,mtta = evaluate_dataset()

print("\n===== PROCESSING NEW VIDEO =====")
df_new = encode_video(NEW_VIDEO_PATH)

risk = predict_risk(df_new)

if risk > RISK_THRESHOLD:
    continual_update(df_new)

print("\n===== RE-EVALUATING DATASET =====")
auc2,ap2,mtta2 = evaluate_dataset()

print("\nUPDATED METRICS")
print("AUC:",auc2)
print("AP:",ap2)
print("mTTA:",mtta2/FPS)


risk_after = predict_risk(df_new)

print("Risk before update:", risk)
print("Risk after update:", risk_after)