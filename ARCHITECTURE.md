# System Architecture

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: MP4 VIDEO FILE                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING (preprocessing.py)                                   │
│  ├─ Load MP4 video                                                 │
│  ├─ Normalize FPS (e.g., 10 FPS)                                   │
│  ├─ Resize frames (640×480)                                        │
│  └─ Yield preprocessed frame stream                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │ For each frame:                        │
         │  {frame, timestamp}                   │
         ▼                                        ▼
┌──────────────────────────────────────┐
│ DETECTION (detection_tracking.py)    │
│                                      │
│ YOLOv8 Object Detection              │
│ ├─ Classes: car, truck, bus,        │
│ │           motorcycle, pedestrian   │
│ └─ Output: Detections               │
│    {x1,y1,x2,y2, conf, class_id}   │
│                                      │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ TRACKING (detection_tracking.py)     │
│                                      │
│ ByteTrack Multi-Object Tracker       │
│ ├─ Input: Detections                │
│ ├─ Output: Track IDs                │
│ └─ Maintains consistent IDs          │
│    (no license plates, identity)     │
│                                      │
└────────┬─────────────────────────────┘
         │
         │ {track_id, Detection, timestamp}
         │
         ▼
┌───────────────────────────────────────────────────────────────────┐
│ TRAJECTORY EXTRACTION (trajectory.py)                             │
│                                                                   │
│ For each track_id over time:                                     │
│  ├─ center_x, center_y (position)                               │
│  ├─ velocity_x, velocity_y (motion)                             │
│  ├─ acceleration_x, acceleration_y (2nd order)                  │
│  ├─ speed = ||velocity||                                        │
│  └─ heading_angle = atan2(vy, vx)                              │
│                                                                   │
│ Output: AgentTrajectory[]                                        │
│  └─ Each has FrameState[] indexed by frame_idx                 │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────────────┐
│ INTERACTION FEATURES (interaction_features.py)                    │
│                                                                   │
│ For each frame, compute pairwise agent interactions:             │
│  ├─ Distance to nearest neighbors                               │
│  ├─ Relative velocity vectors                                   │
│  ├─ Time-to-collision (TTC)                                     │
│  ├─ Collision course detection                                  │
│  └─ Intersection heuristics                                     │
│                                                                   │
│ Output: InteractionFeatures[track_id][frame_idx]               │
└────────┬────────────────────────────────────────────────────────┘
         │
         │ Parallel processing for each agent:
         │
    ┌────┴──────┬─────────────┬────────────────┬────────────┐
    │            │             │                │            │
    ▼            ▼             ▼                ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ...
│Agent 1   │ │Agent 2   │ │Agent 3   │ │Agent N   │
└─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘
      │            │            │            │
      │ {observed trajectory, frame_index}
      │            │            │            │
      ▼            ▼            ▼            ▼
┌──────────────────────────────────────────────────────┐
│ STATE ENCODING (state_encoder.py)                   │
│                                                      │
│ LSTM-Based Temporal Encoder:                        │
│  Input:  Sliding window of T_obs frames             │
│          [cx, cy, vx, vy, ax, ay] per frame        │
│  Process: LSTM → extract final hidden state         │
│  Output: Latent vector z ∈ ℝ^32                    │
│          (compressed motion representation)         │
│                                                      │
│ LatentMotionState:                                  │
│  ├─ track_id                                        │
│  ├─ latent_vector (32-dim)                         │
│  ├─ confidence (based on available history)        │
│  └─ frame_idx                                      │
│                                                      │
└────────────┬─────────────────────────────────────────┘
             │
             │ {latent_vector, current_position}
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ TRAJECTORY SAMPLING (trajectory_sampler.py)         │
│                                                      │
│ Diffusion-Inspired Stochastic Sampler:              │
│  1. Start with random noise (K samples)             │
│  2. Iterative denoising using:                      │
│     ├─ Latent motion information                   │
│     └─ Small denoising network (MLP)               │
│  3. Generate K plausible futures                    │
│                                                      │
│ For each sample:                                    │
│  FutureTrajectory:                                  │
│  ├─ trajectory ∈ ℝ^(T_future × 2)                 │
│  ├─ velocities ∈ ℝ^(T_future × 2)                 │
│  └─ sample_idx ∈ [0, K-1]                         │
│                                                      │
│ Output: K future trajectories per agent             │
│                                                      │
└────────────┬────────────────────────────────────────┘
             │
             │ {trajectory, velocities} × K samples
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ RISK EVALUATION (risk_evaluator.py)                 │
│                                                      │
│ For each future trajectory, evaluate:               │
│  ├─ Collision risk                                  │
│  │  └─ Min distance to other agents                │
│  ├─ Near-miss risk                                  │
│  │  └─ TTC-based soft indicator                    │
│  └─ Abrupt braking risk                            │
│     └─ Velocity discontinuities                    │
│                                                      │
│ FutureRiskScore:                                   │
│  ├─ collision_risk ∈ [0,1]                        │
│  ├─ near_miss_risk ∈ [0,1]                        │
│  ├─ abrupt_brake_risk ∈ [0,1]                     │
│  ├─ composite_risk = weighted average              │
│  └─ events[] (collision events)                    │
│                                                      │
│ Output: FutureRiskScore[] for K samples             │
│                                                      │
└────────────┬────────────────────────────────────────┘
             │
             │ K risk scores per agent per frame
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ RISK AGGREGATION (risk_aggregation.py)              │
│                                                      │
│ Aggregate K samples → Single score:                 │
│  ├─ mean_composite_risk = mean(scores)             │
│  ├─ max_composite_risk = max(scores)               │
│  └─ percentile_95_risk = p95(scores)               │
│                                                      │
│ AggregatedRiskScore:                                │
│  ├─ track_id                                        │
│  ├─ frame_idx                                       │
│  ├─ mean_composite_risk                            │
│  ├─ max_composite_risk                             │
│  └─ percentile_95_risk                             │
│                                                      │
└────────────┬────────────────────────────────────────┘
             │
             │ Aggregated risk per frame
             │
             ▼
┌──────────────────────────────────────────────────────┐
│ TEMPORAL SMOOTHING (risk_aggregation.py)            │
│                                                      │
│ Exponential Moving Average:                         │
│  smoothed[t] = α·current[t] +                       │
│                (1-α)·smoothed[t-1]                 │
│                                                      │
│ where α = temporal_alpha (default 0.3)             │
│                                                      │
│ Purpose: Reduce flickering, smooth predictions      │
│                                                      │
│ Final Output: smoothed_risk ∈ [0, 1]              │
│                                                      │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────┐
│                    OUTPUT GENERATION                 │
│                                                      │
│  1. CSV: Trajectories                              │
│     └─ position, velocity, acceleration            │
│                                                      │
│  2. CSV: Risk Scores                               │
│     └─ risk per agent per frame                    │
│                                                      │
│  3. JSON: Summary                                  │
│     └─ metadata, config, statistics                │
│                                                      │
│  4. MP4: Annotated Video (optional)               │
│     └─ overlaid trajectories & risk scores         │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Module Interaction Graph

```
                      ┌─────────────────────┐
                      │   VideoPreprocessor │
                      │  (preprocessing.py) │
                      └──────────┬──────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              frame stream          frame_times list
                    │                         │
                    ▼                         ▼
            ┌────────────────────────────────────────┐
            │  DetectionTrackingPipeline             │
            │  (detection_tracking.py)               │
            │  ├─ MultiAgentDetector (YOLOv8)      │
            │  └─ MultiAgentTracker (ByteTrack)    │
            └──────────┬───────────────────────────┘
                       │
              Track objects (track_id → detections)
                       │
                       ▼
            ┌────────────────────────────────────────┐
            │  TrajectoryExtractor                   │
            │  (trajectory.py)                       │
            └──────────┬───────────────────────────┘
                       │
          Trajectories (track_id → AgentTrajectory)
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
  ┌─────────┐  ┌──────────────┐  ┌──────────────┐
  │Interact.│  │State         │  │(frame_times) │
  │Features │  │Encoder (LSTM)│  └──────────────┘
  │(distance,   │(latent)      │
  │ TTC, ...)   └──────────────┘
  └────┬────────────┬──────────────┐
       │            │              │
  Interaction     Latent         (stored)
  Features        Vectors
       │            │
       │            └─────────┬─────────┐
       │                      │         │
       ▼                      ▼         ▼
┌────────────────────────────────────────────────┐
│  TrajectoryDiffusionSampler                   │
│  (trajectory_sampler.py)                      │
│  ├─ DenoisingNetwork (small MLP)             │
│  └─ Generates K future trajectories          │
└────────────┬─────────────────────────────────┘
             │
        K futures per agent
             │
             ▼
┌────────────────────────────────────────────────┐
│  RiskEvaluator                                │
│  (risk_evaluator.py)                          │
│  └─ Evaluates each future                    │
│     (collision, near-miss, braking)          │
└────────────┬─────────────────────────────────┘
             │
        K risk scores per agent
             │
             ▼
┌────────────────────────────────────────────────┐
│  RiskAggregationPipeline                      │
│  (risk_aggregation.py)                        │
│  ├─ RiskAggregator (combine K samples)       │
│  ├─ TemporalSmoother (EMA)                   │
│  └─ Output: smoothed_risk ∈ [0,1]           │
└────────────┬─────────────────────────────────┘
             │
        Final Risk Scores
             │
             ▼
    ┌────────────────────────┐
    │   OUTPUT HANDLERS      │
    │  (pipeline.py)         │
    ├─ Save CSV             │
    ├─ Save JSON            │
    ├─ Annotate video       │
    └─ Generate report      │
```

## Data Structure Hierarchy

```
Pipeline (AccidentAnticipationPipeline)
├── VideoPreprocessor
│   └─ Yields (frame, timestamp)
│
├── DetectionTrackingPipeline
│   ├─ MultiAgentDetector
│   │   └─ Detection[]
│   │       ├─ x1, y1, x2, y2
│   │       ├─ confidence
│   │       ├─ class_id
│   │       └─ class_name
│   │
│   └─ MultiAgentTracker
│       └─ Track[] (indexed by track_id)
│           ├─ track_id
│           ├─ class_name
│           └─ detections[]
│
├── TrajectoryExtractor
│   └─ AgentTrajectory[] (indexed by track_id)
│       ├─ track_id
│       ├─ class_name
│       └─ states[] (FrameState[])
│           ├─ frame_idx
│           ├─ timestamp
│           ├─ center_x, center_y
│           ├─ velocity_x, velocity_y
│           ├─ acceleration_x, acceleration_y
│           ├─ speed
│           └─ heading_angle
│
├── InteractionFeatureComputer
│   └─ InteractionFeatures[track_id][frame_idx]
│       ├─ distance_to_nearest
│       ├─ relative_velocity
│       ├─ ttc_nearest
│       ├─ collision_indicator
│       └─ intersection_risk
│
├── StateEncoder
│   └─ LatentMotionState[track_id][frame_idx]
│       ├─ track_id
│       ├─ latent_vector (32-dim)
│       ├─ confidence
│       └─ input_sequence_length
│
├── TrajectoryDiffusionSampler
│   └─ FutureTrajectory[track_id][sample_idx]
│       ├─ track_id
│       ├─ sample_idx
│       ├─ trajectory (N×2 array)
│       ├─ velocities (N×2 array)
│       └─ log_probability
│
├── RiskEvaluator
│   └─ FutureRiskScore[track_id][sample_idx]
│       ├─ collision_risk
│       ├─ near_miss_risk
│       ├─ abrupt_brake_risk
│       ├─ composite_risk
│       └─ events[]
│
└── RiskAggregationPipeline
    └─ AggregatedRiskScore[track_id][frame_idx]
        ├─ mean_composite_risk
        ├─ max_composite_risk
        ├─ percentile_95_risk
        ├─ smoothed_risk ← FINAL OUTPUT
        └─ num_samples
```

## Computational Complexity

| Module | Complexity | GPU Time | CPU Time |
|--------|-----------|----------|----------|
| Preprocessing | O(F) | - | ~0.1s/frame |
| YOLOv8 Detection | O(W×H) | ~30ms | ~100ms |
| ByteTrack | O(N²) | ~1ms | ~5ms |
| Trajectory Extract | O(N) | ~1ms | ~5ms |
| Interaction Features | O(N²) | ~5ms | ~20ms |
| State Encoding | O(N×T_obs) | ~10ms | ~50ms |
| Trajectory Sampling | O(N×K×T_f) | ~20ms | ~100ms |
| Risk Evaluation | O(N²×K×T_f) | ~30ms | ~150ms |
| Risk Aggregation | O(N×K) | ~5ms | ~20ms |
| **Total (10 FPS)** | - | **~100ms** | **~500ms** |

**F** = frames, **N** = agents, **K** = samples, **T_obs** = observation window, **T_f** = prediction horizon

Result: ~2-5× real-time with GPU, ~1-2× real-time with CPU (modern machine)
