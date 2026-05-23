import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)

sys.path.append(" ")
sys.path.append(" ")

VIDEO_FILE = "path/to/video.mp4"
MODEL_PATH = "path/to/diffusion_interaction_model.pt"

START_SEC = 30
END_SEC = 39
TARGET_FPS = 10

from preprocessing import VideoPreprocessor
from detection_tracking import DetectionTrackingPipeline

# ==========================================
# VIDEO
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
    frames_used.append((frame.copy(), idx, timestamp))

vp.close_video()

print("Frames processed:", len(frames_used))

raw = frames_used[0][0]

plt.figure(figsize=(12,7))
plt.imshow(raw[:,:,::-1])
plt.title("Original Input Frame")
plt.axis("off")
plt.show()

tracks = pipeline.get_tracks(min_duration_frames=1)

print("Tracks detected:", len(tracks))

# ==========================================
# DRAW BOXES + LABELS
# ==========================================

boxed = raw.copy()

for track_id, track in tracks.items():

    if len(track.detections) == 0:
        continue

    frame_idx, det = track.detections[0]

    x1 = int(det.x1)
    y1 = int(det.y1)
    x2 = int(det.x2)
    y2 = int(det.y2)

    conf = float(det.confidence)

    cv2.rectangle(
        boxed,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"ID {track_id} | {det.class_name} | {conf:.2f}"

    cv2.putText(
        boxed,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )

plt.figure(figsize=(12,7))
plt.imshow(boxed[:,:,::-1])
plt.title("Detected Agents + Tracking IDs + Confidence")
plt.axis("off")
plt.show()