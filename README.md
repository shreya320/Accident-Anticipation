# Traffic Accident Risk Prediction System

A research-grade system for predicting accident risk in traffic scenes using CCTV video. Uses YOLOv8 detection, ByteTrack for multi-agent tracking, and diffusion-inspired trajectory sampling with risk evaluation.

## System Overview

```
INPUT VIDEO (MP4)
    ↓
PREPROCESSING (FPS normalization, frame resizing)
    ↓
DETECTION & TRACKING (YOLOv8 + ByteTrack)
    ↓
TRAJECTORY EXTRACTION (Motion features per agent)
    ↓
INTERACTION FEATURES (Distance, velocity, TTC)
    ↓
STATE ENCODING (LSTM-based temporal encoder)
    ↓
FUTURE TRAJECTORY SAMPLING (Diffusion-inspired)
    ↓
RISK EVALUATION (Collision, near-miss detection)
    ↓
RISK AGGREGATION & SMOOTHING (EMA temporal smoothing)
    ↓
OUTPUT (CSV, JSON, annotated video)
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (recommended for GPU acceleration)

### Setup

1. Clone or navigate to the project directory:
```bash
cd "Accident Anticipation"
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Download YOLOv8 model weights:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

## Quick Start

### Basic Usage

```bash
python main.py --video data/videos/sample.mp4 --output output --save-video
```

### Advanced Options

```bash
python main.py \
    --video data/videos/sample.mp4 \
    --output output \
    --yolo-model yolov8l.pt \
    --confidence 0.5 \
    --latent-dim 32 \
    --num-samples 5 \
    --prediction-horizon 30 \
    --observation-window 20 \
    --target-fps 10 \
    --save-video \
    --log-level INFO
```

### Command-Line Arguments

```
--video                Path to input video file (REQUIRED)
--output              Output directory (default: output)
--save-video          Save annotated video with risk overlays
--save-csv            Save results to CSV (default: True)
--yolo-model          YOLOv8 model: nano/small/medium/large/xlarge (default: yolov8m.pt)
--confidence          Detection confidence threshold 0-1 (default: 0.5)
--latent-dim          Latent vector dimension (default: 32)
--num-samples         Number of future trajectory samples (default: 5)
--prediction-horizon  Prediction horizon in frames (default: 30)
--observation-window  Observation window in frames (default: 20)
--target-fps          Target FPS for processing (default: 10)
--use-cuda            Use CUDA for GPU acceleration (default: True)
--no-cuda             Disable CUDA
--log-level           Logging level: DEBUG/INFO/WARNING/ERROR (default: INFO)
```

## Project Structure

```
Accident Anticipation/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Video loading & FPS normalization
│   ├── detection_tracking.py      # YOLOv8 + ByteTrack
│   ├── trajectory.py              # Motion feature extraction
│   ├── interaction_features.py    # Agent interactions (distance, TTC, etc.)
│   ├── state_encoder.py           # LSTM temporal encoder
│   ├── trajectory_sampler.py      # Diffusion-inspired trajectory sampling
│   ├── risk_evaluator.py          # Risk scoring for futures
│   ├── risk_aggregation.py        # Aggregation & EMA smoothing
│   └── pipeline.py                # Main integrated pipeline
├── main.py                         # Entry point
├── requirements.txt                # Dependencies
├── data/
│   └── videos/                    # Input videos
├── output/                         # Results (created automatically)
│   ├── trajectories.csv           # Per-agent trajectories
│   ├── risk_scores.csv            # Per-agent risk scores over time
│   ├── summary.json               # Processing summary
│   └── annotated_output.mp4       # Video with risk overlays
└── models/                         # Pretrained checkpoints (optional)
```

## Output Files

### trajectories.csv
Per-frame trajectory data for each tracked agent:
- `track_id`: Unique agent identifier
- `frame_idx`: Frame index
- `timestamp`: Frame timestamp (seconds)
- `center_x`, `center_y`: Agent center position (pixels)
- `velocity_x`, `velocity_y`: Velocity components (pixels/frame)
- `speed`: Velocity magnitude
- `class_name`: Object class (car, pedestrian, etc.)

### risk_scores.csv
Per-frame risk scores for each agent:
- `track_id`: Agent identifier
- `frame_idx`: Frame index
- `timestamp`: Frame timestamp
- `collision_risk`: Risk of collision [0, 1]
- `near_miss_risk`: Risk of near-miss [0, 1]
- `abrupt_brake_risk`: Risk of abrupt braking [0, 1]
- `mean_composite_risk`: Mean risk across samples [0, 1]
- `smoothed_risk`: Temporally smoothed risk [0, 1]
- `num_samples`: Number of trajectory samples used

### summary.json
Metadata and processing parameters

### annotated_output.mp4
Video with agent trajectories and risk scores overlaid
- Green: Low risk (< 0.2)
- Yellow: Medium risk (0.2 - 0.5)
- Red: High risk (> 0.5)

## Module Details

### 1. Preprocessing (`preprocessing.py`)
- Loads MP4 video files
- Normalizes FPS (default: 10 FPS)
- Resizes frames for YOLO (default: 640×480)
- Handles frame iteration with metadata

**Key Classes:**
- `VideoPreprocessor`: Main preprocessing handler
- `load_video_frames()`: Convenience function

### 2. Detection & Tracking (`detection_tracking.py`)
- YOLOv8 multi-class object detection
- ByteTrack for consistent track IDs (no license plates, no personal data)
- Detects: car, truck, bus, motorcycle, pedestrian

**Key Classes:**
- `MultiAgentDetector`: YOLO-based detector
- `MultiAgentTracker`: ByteTrack integration
- `DetectionTrackingPipeline`: Integrated pipeline
- `Track`: Represents one tracked agent over time

### 3. Trajectory Extraction (`trajectory.py`)
- Per-frame: center position, velocity, acceleration, heading
- Smooth velocity estimates using moving average
- Temporal sequences indexed by track_id

**Key Classes:**
- `AgentTrajectory`: Complete trajectory for one agent
- `FrameState`: Single frame state with motion features
- `TrajectoryExtractor`: Main extraction handler

### 4. Interaction Features (`interaction_features.py`)
- Distance to nearest agents
- Relative velocity vectors
- Time-to-collision (TTC) approximation
- Collision/intersection heuristics

**Key Classes:**
- `InteractionFeatures`: Per-agent frame features
- `InteractionFeatureComputer`: Feature computation

### 5. State Encoder (`state_encoder.py`)
- LSTM-based temporal encoder
- Sliding window of T_obs frames (default: 20)
- Outputs latent motion representation
- Freezes weights (research prototype)

**Key Classes:**
- `TemporalMotionEncoder`: PyTorch LSTM module
- `StateEncoder`: Encoding wrapper
- `LatentMotionState`: Encoded representation

### 6. Trajectory Sampler (`trajectory_sampler.py`)
- Diffusion-inspired stochastic sampler (NOT pixel-based)
- Generates K plausible futures from latent state
- Small denoising network
- Also includes simple linear predictor as fallback

**Key Classes:**
- `TrajectoryDiffusionSampler`: Main sampler
- `DenoisingNetwork`: Small denoising MLP
- `FutureTrajectory`: Single sample
- `SimpleTrajectoryPredictor`: Baseline linear model

### 7. Risk Evaluator (`risk_evaluator.py`)
- Per-future trajectory evaluation
- Detects: collision, near-miss, abrupt braking
- Outputs soft risk score [0, 1]

**Key Classes:**
- `RiskEvaluator`: Main evaluator
- `FutureRiskScore`: Risk for one sample
- `CollisionEvent`: Detected event

### 8. Risk Aggregation (`risk_aggregation.py`)
- Aggregates risk across K samples (mean, max, weighted mean)
- Exponential moving average temporal smoothing
- Normalizes scores

**Key Classes:**
- `RiskAggregator`: Sample aggregation
- `TemporalSmoother`: EMA smoothing
- `AggregatedRiskScore`: Final aggregated score
- `RiskAggregationPipeline`: Complete pipeline

### 9. Main Pipeline (`pipeline.py`)
- Orchestrates all modules
- Handles data flow
- Manages I/O

**Key Classes:**
- `AccidentAnticipationPipeline`: Main entry point

## Configuration

Default configuration (can be customized):

```python
{
    # Preprocessing
    'target_fps': 10,
    'frame_width': 640,
    'frame_height': 480,
    
    # Detection
    'yolo_model': 'yolov8m.pt',
    'detection_confidence': 0.5,
    'use_byte_track': True,
    'use_cuda': True,
    
    # Trajectory
    'pixels_per_meter': 1.0,
    
    # Interaction
    'proximity_threshold': 300.0,
    'ttc_threshold': 2.0,
    
    # State encoding
    'latent_dim': 32,
    'observation_window': 20,
    
    # Trajectory sampling
    'num_diffusion_steps': 10,
    'prediction_horizon': 30,
    'num_trajectory_samples': 5,
    
    # Risk evaluation
    'collision_distance': 30.0,
    'near_miss_distance': 100.0,
    
    # Risk aggregation
    'aggregation_method': 'mean',
    'temporal_alpha': 0.3,
}
```

## Performance Notes

- **GPU recommended**: CUDA acceleration significantly speeds up YOLOv8 and encoder
- **Processing time**: Typically 2-5× real-time with GPU
- **Memory**: ~2-4 GB GPU memory for medium model, ~8-16 GB CPU RAM for full video
- **Scalability**: Processes multiple agents in parallel; independent per-agent encoding/sampling

## Research Notes

### Diffusion-Inspired Sampling
The trajectory sampler uses a simplified diffusion process:
1. Start with random noise
2. Iteratively denoise using latent motion information
3. Produce stochastic future trajectories

This is NOT a pixel-level diffusion model but trajectory-level, keeping it lightweight and interpretable.

### Risk Scoring
Risk combines three components:
- **Collision Risk**: Proximity to other agents (hard threshold)
- **Near-Miss Risk**: Soft proximity indicator (TTC-based)
- **Abrupt Braking Risk**: Velocity discontinuities

Final risk = weighted average of components (default: 60% collision, 30% near-miss, 10% braking)

### Temporal Smoothing
Exponential moving average (EMA) eliminates flickering:
```
smoothed[t] = α × current[t] + (1-α) × smoothed[t-1]
```
Default α = 0.3 (30% current, 70% history). Adjustable via `--temporal-alpha`.

## Limitations & Future Work

### Current Limitations
1. Static camera assumption (no ego-motion compensation)
2. No scene context or lane information
3. Simplified collision model (circular approximation)
4. No multi-agent interaction in sampling (independent futures)

### Future Enhancements
1. **Ego-motion compensation**: Handle camera movement
2. **Scene context**: Incorporate road structure, traffic rules
3. **Graph neural networks**: Model agent interactions explicitly
4. **Attention mechanisms**: Learn importance weighting
5. **Calibration**: Per-class risk thresholds
6. **Real-time optimization**: Reduce latency for live systems

## Troubleshooting

### CUDA Out of Memory
- Use smaller YOLOv8 model: `--yolo-model yolov8n.pt`
- Reduce frame size: `--frame-width 480 --frame-height 360`
- Reduce samples: `--num-samples 3`

### Slow Processing
- Enable CUDA: Ensure `--use-cuda` (not `--no-cuda`)
- Use GPU monitoring: `nvidia-smi` to verify GPU usage
- Reduce observation window: `--observation-window 10`

### No Detections
- Lower confidence threshold: `--confidence 0.3`
- Use larger YOLOv8 model: `--yolo-model yolov8l.pt`
- Check video quality and lighting

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify installation
python -c "from ultralytics import YOLO; print('OK')"
```

## Citation

If you use this system in research, please cite:

```bibtex
@software{accident_anticipation_2024,
  title = {Traffic Accident Risk Prediction from CCTV Video},
  author = {Research Team},
  year = {2024},
  url = {https://github.com/your-repo/accident-anticipation}
}
```

## License

Research use only. See LICENSE file for details.

## Contact

For questions or issues, please open an issue in the repository.
