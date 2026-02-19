# Module Reference Guide

## Quick Module Index

| Module | Purpose | Key Classes | Status |
|--------|---------|-------------|--------|
| `preprocessing.py` | Video I/O & FPS normalization | `VideoPreprocessor` | ✅ |
| `detection_tracking.py` | YOLOv8 + ByteTrack | `DetectionTrackingPipeline`, `Track` | ✅ |
| `trajectory.py` | Motion feature extraction | `AgentTrajectory`, `FrameState` | ✅ |
| `interaction_features.py` | Agent interactions | `InteractionFeatureComputer` | ✅ |
| `state_encoder.py` | LSTM temporal encoding | `StateEncoder`, `TemporalMotionEncoder` | ✅ |
| `trajectory_sampler.py` | Diffusion-inspired sampling | `TrajectoryDiffusionSampler` | ✅ |
| `risk_evaluator.py` | Risk scoring | `RiskEvaluator`, `FutureRiskScore` | ✅ |
| `risk_aggregation.py` | Aggregation & smoothing | `RiskAggregationPipeline`, `TemporalSmoother` | ✅ |
| `pipeline.py` | Main orchestration | `AccidentAnticipationPipeline` | ✅ |

---

## Module Details

### 1. preprocessing.py

**Purpose**: Load and normalize video

**Main Classes**:
- `VideoPreprocessor` - Main video handler
  - `__init__(target_fps, target_width, target_height)`
  - `open_video(video_path) -> bool`
  - `close_video()`
  - `get_frame_generator() -> Generator`
  - `get_info() -> dict`

**Functions**:
- `load_video_frames(video_path, ...) -> (frames, info)`

**Usage**:
```python
preprocessor = VideoPreprocessor(target_fps=10, target_width=640, target_height=480)
preprocessor.open_video('video.mp4')
for frame, idx, timestamp in preprocessor.get_frame_generator():
    # Process frame
    pass
preprocessor.close_video()
```

---

### 2. detection_tracking.py

**Purpose**: Detect and track agents

**Data Classes**:
- `Detection` - Single detection bbox
  - center, width, height, area, to_xyxy()
  
- `Track` - Tracked object over time
  - add_detection(), get_detection_at_frame()
  - get_center_trajectory(), duration_frames()
  
- `FrameDetections` - Detections in one frame
  - frame_idx, timestamp, detections[]

**Main Classes**:
- `MultiAgentDetector` - YOLOv8 wrapper
  - `__init__(model_name, confidence, device)`
  - `detect_frame(frame) -> Detection[]`

- `MultiAgentTracker` - ByteTrack wrapper
  - `__init__(use_byte_track, iou_threshold)`
  - `update(detections) -> Dict[Detection, int]`
  - `get_active_tracks(min_duration_frames)`
  - `get_all_tracks() -> Dict[int, Track]`

- `DetectionTrackingPipeline` - Integrated pipeline
  - `process_frame(frame, timestamp) -> (FrameDetections, det_to_track_mapping)`
  - `get_tracks(min_duration_frames) -> Dict[int, Track]`

**Usage**:
```python
pipeline = DetectionTrackingPipeline(use_byte_track=True)
for frame in frames:
    frame_det, det_to_track = pipeline.process_frame(frame, timestamp)
tracks = pipeline.get_tracks(min_duration_frames=3)
```

---

### 3. trajectory.py

**Purpose**: Extract motion features

**Data Classes**:
- `FrameState` - Single frame state
  - frame_idx, timestamp
  - center_x, center_y, velocity_x, velocity_y, acceleration_x, acceleration_y
  - speed, heading_angle
  - position, velocity, acceleration properties

- `AgentTrajectory` - Complete trajectory
  - track_id, class_name, states[]
  - get_positions(), get_velocities(), get_accelerations()
  - get_speeds(), get_headings()
  - get_state_at_frame(), get_states_in_frame_range()
  - duration_frames(), duration_sec()
  - smooth_velocities()

**Main Classes**:
- `TrajectoryExtractor` - Motion feature computer
  - `extract_trajectories(tracks) -> Dict[int, AgentTrajectory]`
  - `update_timestamps(frame_times)`
  - `smooth_all_trajectories(window_size)`
  - `get_trajectory(track_id)`
  - `filter_trajectories(min_duration_frames)`
  - `get_summary_stats()`

**Usage**:
```python
extractor = TrajectoryExtractor(pixels_per_meter=1.0)
trajectories = extractor.extract_trajectories(tracks)
extractor.update_timestamps(frame_times)
extractor.smooth_all_trajectories(window_size=3)
traj = extractor.get_trajectory(track_id=1)
```

---

### 4. interaction_features.py

**Purpose**: Compute agent interactions

**Data Classes**:
- `InteractionFeatures` - Per-agent frame features
  - distance_to_nearest, nearest_agent_id
  - relative_velocity_x, relative_velocity_y
  - ttc_nearest, collision_indicator
  - intersection_risk, concurrent_agents

**Main Classes**:
- `InteractionFeatureComputer` - Feature computation
  - `compute_frame_interactions(frame_idx, frame_states) -> Dict[int, InteractionFeatures]`
  - `compute_trajectory_interactions(trajectories) -> Dict[int, Dict[int, InteractionFeatures]]`
  - `get_interaction_stats(all_interactions) -> dict`

**Usage**:
```python
computer = InteractionFeatureComputer(proximity_threshold=300.0, ttc_threshold=2.0)
interactions = computer.compute_trajectory_interactions(trajectories)
stats = computer.get_interaction_stats(interactions)
```

---

### 5. state_encoder.py

**Purpose**: Encode motion state to latent vectors

**PyTorch Classes**:
- `TemporalMotionEncoder(nn.Module)` - LSTM encoder
  - Input: (batch, seq_len, input_dim)
  - Output: (batch, latent_dim)

**Data Classes**:
- `LatentMotionState` - Encoded state
  - track_id, frame_idx, latent_vector, confidence

**Main Classes**:
- `StateEncoder` - Wrapper for encoding
  - `encode_trajectory_at_frame(trajectory, frame_idx) -> LatentMotionState`
  - `encode_all_trajectories(trajectories) -> Dict[int, Dict[int, LatentMotionState]]`
  - `get_latest_encoding(trajectory)`
  - `load_pretrained(checkpoint_path)`, `save_checkpoint(checkpoint_path)`

- `PretrainedMotionEncoder` - Ready-to-use encoder

**Usage**:
```python
encoder = StateEncoder(latent_dim=32, device='cuda:0', window_size=20)
encodings = encoder.encode_all_trajectories(trajectories)
latent_state = encoder.get_latest_encoding(trajectory)
```

---

### 6. trajectory_sampler.py

**Purpose**: Generate future trajectory samples

**PyTorch Classes**:
- `DenoisingNetwork(nn.Module)` - Small denoising MLP
  - Input: (batch, latent_dim + 2)
  - Output: (batch, 2)

**Data Classes**:
- `FutureTrajectory` - Single future sample
  - track_id, sample_idx
  - trajectory (N×2), velocities (N×2)

**Main Classes**:
- `TrajectoryDiffusionSampler` - Diffusion-inspired sampler
  - `sample_trajectories(latent, start_pos, num_samples) -> FutureTrajectory[]`
  - `batch_sample_trajectories(latent_vectors, current_positions, num_samples) -> Dict[int, FutureTrajectory[]]`
  - `load_pretrained(checkpoint_path)`, `save_checkpoint(checkpoint_path)`

- `SimpleTrajectoryPredictor` - Baseline linear predictor
  - `predict_trajectory(current_pos, current_vel, num_samples) -> FutureTrajectory[]`

**Usage**:
```python
sampler = TrajectoryDiffusionSampler(latent_dim=32, future_frames=30)
samples = sampler.sample_trajectories(latent_vector, start_pos, num_samples=5)

predictor = SimpleTrajectoryPredictor(future_frames=30)
samples = predictor.predict_trajectory(pos, vel, num_samples=5)
```

---

### 7. risk_evaluator.py

**Purpose**: Evaluate risk for future trajectories

**Data Classes**:
- `CollisionEvent` - Detected collision/near-miss
  - frame_idx, agent_id, other_agent_id
  - distance, event_type (collision/near_miss/brake), severity

- `FutureRiskScore` - Risk for one future
  - track_id, sample_idx
  - collision_risk, near_miss_risk, abrupt_brake_risk
  - composite_risk, events[]
  - compute_composite()

**Main Classes**:
- `RiskEvaluator` - Risk computation
  - `evaluate_trajectory(track_id, trajectory, velocities, other_trajectories) -> FutureRiskScore`
  - `batch_evaluate(futures, other_trajectories) -> Dict[int, FutureRiskScore[]]`

- `FastRiskEvaluator` - Lightweight evaluator

**Usage**:
```python
evaluator = RiskEvaluator(collision_distance=30.0, near_miss_distance=100.0)
risks = evaluator.batch_evaluate(future_trajectories, other_trajectories)
```

---

### 8. risk_aggregation.py

**Purpose**: Aggregate and smooth risks

**Data Classes**:
- `AggregatedRiskScore` - Aggregated score per frame
  - mean_composite_risk, max_composite_risk, percentile_95_risk
  - smoothed_risk, num_samples

**Main Classes**:
- `RiskAggregator` - Combine K samples
  - `aggregate_frame_risks(frame_risks, weights) -> (mean, max, p95)`
  - `aggregate_all_risks(risks_per_sample) -> AggregatedRiskScore`

- `TemporalSmoother` - EMA smoothing
  - `smooth(track_id, current_risk) -> smoothed_risk`
  - `smooth_trajectory(track_id, risk_sequence) -> smoothed_sequence`
  - `reset()`

- `RiskAggregationPipeline` - Complete pipeline
  - `process_frame_risks(frame_idx, timestamp, risks_by_agent)`
  - `apply_temporal_smoothing(track_id, aggregated_risk)`
  - `process_trajectory_risks(all_frame_risks)`

- `RiskScoreNormalizer` - Normalization utilities
  - `normalize(risk_score, min_val, max_val)`
  - `get_risk_category(risk_score) -> str`

**Functions**:
- `compute_final_risk_per_agent(aggregated_trajectory) -> float`

**Usage**:
```python
pipeline = RiskAggregationPipeline(aggregation_method='mean', temporal_alpha=0.3)
frame_agg = pipeline.process_frame_risks(frame_idx, timestamp, risks_by_agent)
smoothed = pipeline.apply_temporal_smoothing(track_id, agg_risk)
```

---

### 9. pipeline.py

**Purpose**: Main pipeline orchestration

**Main Classes**:
- `AccidentAnticipationPipeline` - Complete integration
  - `__init__(config)`
  - `run(video_path, output_dir, save_video, save_csv) -> dict`
  - `get_default_config() -> dict`

**Static Methods**:
- `_save_csv_results(results, output_dir)`
- `_save_json_summary(results, output_dir)`
- `_save_annotated_video(frames, results, output_dir)`

**Usage**:
```python
config = AccidentAnticipationPipeline.get_default_config()
pipeline = AccidentAnticipationPipeline(config)
results = pipeline.run('video.mp4', output_dir='output', save_video=True)
```

---

## Data Flow Summary

```
VideoPreprocessor
    ↓
DetectionTrackingPipeline (YOLOv8 + ByteTrack)
    ↓
TrajectoryExtractor
    ↓
InteractionFeatureComputer
    ↓
StateEncoder (LSTM)
    ↓
TrajectoryDiffusionSampler
    ↓
RiskEvaluator
    ↓
RiskAggregationPipeline
    ↓
Output (CSV, JSON, MP4)
```

---

## Key Data Structures

```python
# Detection
Detection: x1, y1, x2, y2, conf, class_id, class_name

# Tracking
Track: track_id, class_name, detections[]
    └─ Detection[]: per-frame detections for this track

# Trajectory
AgentTrajectory: track_id, class_name, states[]
    └─ FrameState[]: 
        ├─ position: (cx, cy)
        ├─ velocity: (vx, vy)
        ├─ acceleration: (ax, ay)
        ├─ speed, heading_angle

# Interaction
InteractionFeatures: distance, relative_velocity, ttc, collision_indicator

# Encoding
LatentMotionState: latent_vector (32-dim), confidence

# Future
FutureTrajectory: trajectory (N×2), velocities (N×2)

# Risk
FutureRiskScore: collision_risk, near_miss_risk, brake_risk, composite_risk
AggregatedRiskScore: mean/max/p95_risk, smoothed_risk
```

---

## Configuration Parameters

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
    
    # Encoding
    'latent_dim': 32,
    'observation_window': 20,
    
    # Sampling
    'num_diffusion_steps': 10,
    'prediction_horizon': 30,
    'num_trajectory_samples': 5,
    
    # Risk
    'collision_distance': 30.0,
    'near_miss_distance': 100.0,
    
    # Aggregation
    'aggregation_method': 'mean',
    'temporal_alpha': 0.3,
}
```

---

## Command-Line Usage

```bash
# Basic
python main.py --video video.mp4

# Full options
python main.py \
    --video video.mp4 \
    --output output \
    --yolo-model yolov8m.pt \
    --confidence 0.5 \
    --latent-dim 32 \
    --num-samples 5 \
    --prediction-horizon 30 \
    --observation-window 20 \
    --target-fps 10 \
    --save-video \
    --log-level INFO

# Help
python main.py --help
```

---

## Testing

```bash
# Run all component tests
python test_components.py

# Run specific module test (in test file)
# - test_preprocessing()
# - test_trajectory()
# - test_interaction_features()
# - test_state_encoder()
# - test_trajectory_sampler()
# - test_risk_evaluator()
# - test_risk_aggregation()
# - test_pipeline()
```

---

## Output Files

**CSV Format**:
- `trajectories.csv` - Position and velocity per frame
- `risk_scores.csv` - Risk scores per frame

**JSON Format**:
- `summary.json` - Metadata and statistics

**Video Format** (optional):
- `annotated_output.mp4` - Annotated with overlays

---

## Common Patterns

### Iterate Trajectories
```python
for track_id, trajectory in trajectories.items():
    for state in trajectory.states:
        print(f"Agent {track_id}: ({state.center_x}, {state.center_y})")
```

### Access State at Frame
```python
state = trajectory.get_state_at_frame(frame_idx)
if state:
    print(f"Position: ({state.center_x}, {state.center_y})")
    print(f"Velocity: ({state.velocity_x}, {state.velocity_y})")
```

### Custom Configuration
```python
config = AccidentAnticipationPipeline.get_default_config()
config['num_trajectory_samples'] = 10
config['temporal_alpha'] = 0.2
pipeline = AccidentAnticipationPipeline(config)
```

---

**All modules fully documented and ready for use!** ✅
