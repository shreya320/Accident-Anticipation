import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

MODEL_PATH = "path/to/diffusion_interaction_model.pt"
DATA_DIR = "path/to/eval-all-videos"

FUTURE_FRAMES = 30
OUTPUT_DIM = FUTURE_FRAMES * 2

# checkpoint expects 126 total input
# 126 = LATENT_DIM + OUTPUT_DIM + 32
LATENT_DIM = 35

DIFFUSION_STEPS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

model = Denoiser().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================= DIFFUSION =================

betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS).to(DEVICE)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

# ================= FILTER =================

def filter_trajectory(
    steps,
    velocity,
    max_angle_deg=25,
    max_speed_factor=2.0,
    max_total_dev=400
):

    prev_dir = velocity / (np.linalg.norm(velocity) + 1e-8)
    base_speed = np.linalg.norm(velocity) + 1e-8
    cumulative_pos = np.zeros(2)

    for step in steps:

        step_norm = np.linalg.norm(step)

        if step_norm > max_speed_factor * base_speed:
            return False

        step_dir = step / (step_norm + 1e-8)

        dot = np.dot(prev_dir, step_dir)

        angle = np.degrees(
            np.arccos(np.clip(dot, -1.0, 1.0))
        )

        if angle > max_angle_deg:
            return False

        cumulative_pos += step

        if np.linalg.norm(cumulative_pos) > max_total_dev:
            return False

        prev_dir = step_dir

    return True

# ================= SAMPLING =================

def sample(z, start_pos):

    x_t = torch.randn(1, OUTPUT_DIM, device=DEVICE)

    for t in reversed(range(DIFFUSION_STEPS)):

        t_tensor = torch.tensor([t], device=DEVICE)

        noise_pred = model(z, x_t, t_tensor)

        alpha = alphas[t]
        alpha_bar = alpha_cumprod[t]
        beta = betas[t]

        posterior_mean = (1 / torch.sqrt(alpha)) * (
            x_t - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
        )

        if t > 0:
            temperature = 0.25
            x_t = posterior_mean + temperature * torch.sqrt(beta) * torch.randn_like(x_t)
        else:
            x_t = posterior_mean

    traj = x_t.detach().cpu().numpy().reshape(FUTURE_FRAMES, 2)

    traj = traj * STD + MEAN
    traj = traj * 50.0

    steps = traj.copy()

    pos = start_pos.copy()
    positions = []

    for step in steps:
        pos = pos + step
        positions.append(pos.copy())

    return steps, np.array(positions)

# ================= TEST =================

file = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")][0]

df = pd.read_csv(os.path.join(DATA_DIR, file))
df = df.sort_values(["track_id","frame_idx"])

latent_cols = [c for c in df.columns if c.startswith("latent_")]

track_id = df["track_id"].unique()[0]
g = df[df["track_id"] == track_id]

i = 10

dist = g.iloc[i].get("distance_to_nearest", 100.0)

interaction_feat = np.array(
    [np.exp(-dist / 50.0)],
    dtype=np.float32
)

z = np.concatenate([
    g.iloc[i][latent_cols].values,
    g.iloc[i][["velocity_x","velocity_y"]].values,
    interaction_feat
]).astype(np.float32)

start_pos = g.iloc[i][["center_x","center_y"]].values
velocity = g.iloc[i][["velocity_x","velocity_y"]].values

z = torch.tensor(z).unsqueeze(0).to(DEVICE)

plt.figure(figsize=(7,7))
plt.scatter(start_pos[0], start_pos[1], c='red', s=100)

valid_count = 0
tries = 0

while valid_count < 10 and tries < 2000:

    steps, traj = sample(z, start_pos)

    if filter_trajectory(steps, velocity):
        plt.plot(traj[:,0], traj[:,1], linewidth=2)
        valid_count += 1

    tries += 1

plt.gca().invert_yaxis()
plt.title("Diffusion Future Trajectories")
plt.show()

print("Valid Paths Generated:", valid_count)