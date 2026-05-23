import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# ================= CONFIG =================

DATA_DIR = "/content/drive/.shortcut-targets-by-id/1Cdu5ESiLBps0D5jkXFtnoqU20zMDj_Up/CarAccidentProject/evaluation_results_b"
SAVE_PATH = "/content/drive/MyDrive/diffusion_interaction_model.pt"

FUTURE_FRAMES = 30
OUTPUT_DIM = FUTURE_FRAMES * 2
LATENT_DIM = 35  # ✅ 34 + 1 interaction feature
DIFFUSION_STEPS = 50
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= DIFFUSION SCHEDULE =================

betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS).to(DEVICE)
alphas = 1 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

# ================= LOAD DATA =================

def load_dataset():
    all_latents = []
    all_targets = []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    for file in tqdm(files):
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        df = df.sort_values(["track_id", "frame_idx"])

        latent_cols = [c for c in df.columns if c.startswith("latent_")]

        for track_id in df["track_id"].unique():
            g = df[df["track_id"] == track_id].sort_values("frame_idx")

            for i in range(len(g) - FUTURE_FRAMES - 1):

                row = g.iloc[i]

                # ===== interaction feature =====
                dist = row.get("distance_to_nearest", 100.0)
                interaction_feat = np.array([np.exp(-dist / 50.0)], dtype=np.float32)

                z = np.concatenate([
                    row[latent_cols].values,
                    row[["velocity_x", "velocity_y"]].values,
                    interaction_feat
                ]).astype(np.float32)

                future = g.iloc[i+1:i+1+FUTURE_FRAMES][["center_x","center_y"]].values
                current = row[["center_x","center_y"]].values

                increments = future - current
                increments = increments.astype(np.float32)

                # normalize
                increments = increments / 50.0
                increments = increments.reshape(-1)

                all_latents.append(z)
                all_targets.append(increments)

    return torch.tensor(all_latents), torch.tensor(all_targets)

print("Loading dataset...")
CACHE_PATH = "diffusion_dataset_cache_interaction.pt"

if os.path.exists(CACHE_PATH):
    print("Loading cached dataset...")
    data = torch.load(CACHE_PATH)
    Z = data["Z"]
    X0 = data["X0"]
else:
    print("Building dataset from CSVs...")
    Z, X0 = load_dataset()

    print("Saving cache...")
    torch.save({"Z": Z, "X0": X0}, CACHE_PATH)

print("Total samples:", len(Z))

# ================= NORMALIZATION =================

mean = X0.mean()
std = X0.std()

print("Mean:", mean.item())
print("Std:", std.item())

X0 = (X0 - mean) / (std + 1e-8)

# ================= TRAIN =================

for epoch in range(EPOCHS):
    perm = torch.randperm(len(Z))
    epoch_loss = 0

    for i in range(0, len(Z), BATCH_SIZE):

        idx = perm[i:i+BATCH_SIZE]
        z = Z[idx].to(DEVICE)
        x0 = X0[idx].to(DEVICE)

        t = torch.randint(0, DIFFUSION_STEPS, (z.size(0),), device=DEVICE)

        alpha_bar = alpha_cumprod[t].unsqueeze(1)
        noise = torch.randn_like(x0)

        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

        noise_pred = model(z, x_t, t)

        # ===== weighted loss (focus on interaction-heavy samples) =====
        weight = 1 + 2 * z[:, -1:].detach()
        loss = ((noise_pred - noise) ** 2 * weight).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= (len(Z) // BATCH_SIZE)
    print(f"Epoch {epoch+1} Avg Loss: {epoch_loss:.4f}")

# ================= SAVE =================

torch.save(model.state_dict(), SAVE_PATH)
print("Model saved at:", SAVE_PATH)