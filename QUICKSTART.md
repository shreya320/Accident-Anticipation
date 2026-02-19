# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Navigate to project
cd "Accident Anticipation"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test installation
python test_components.py
```

## Running the System

### Minimum (produce CSV output)
```bash
python main.py --video data/videos/sample.mp4
```

### With video output
```bash
python main.py --video data/videos/sample.mp4 --save-video
```

### Full example (all options)
```bash
python main.py \
    --video data/videos/sample.mp4 \
    --output output \
    --yolo-model yolov8m.pt \
    --confidence 0.5 \
    --num-samples 5 \
    --prediction-horizon 30 \
    --save-video \
    --log-level INFO
```

## Output Interpretation

### Risk Score Levels
- **0.0 - 0.1**: Very safe (green)
- **0.1 - 0.25**: Safe (light green)
- **0.25 - 0.5**: Medium risk (yellow)
- **0.5 - 0.75**: High risk (orange)
- **0.75 - 1.0**: Critical risk (red)

### CSV Files

**risk_scores.csv**
- `smoothed_risk`: Final score you should use
- Higher values = more dangerous
- Use for alerting, statistics

**trajectories.csv**
- Position and velocity data for each agent
- For visualization, analysis, debugging

## Customization

### Change Detection Model
```bash
# Faster (GPU)
python main.py --video video.mp4 --yolo-model yolov8n.pt

# More accurate (slower)
python main.py --video video.mp4 --yolo-model yolov8x.pt
```

### Adjust Prediction Horizon
```bash
# Look further into future (slower)
python main.py --video video.mp4 --prediction-horizon 50

# Near-term predictions (faster)
python main.py --video video.mp4 --prediction-horizon 20
```

### More Trajectory Samples
```bash
# More robust but slower
python main.py --video video.mp4 --num-samples 10

# Fast and rough
python main.py --video video.mp4 --num-samples 3
```

### Adjust Smoothing
Smooth = less flickering, but lags behind sudden changes
```bash
# More smooth (lag = 70% history)
python main.py --video video.mp4 --temporal-alpha 0.1

# More responsive (lag = 30% history) [default]
python main.py --video video.mp4 --temporal-alpha 0.3

# Very responsive (lag = 10% history)
python main.py --video video.mp4 --temporal-alpha 0.5
```

## Performance Tips

### For Speed
```bash
# Smaller model + lower FPS + fewer samples
python main.py --video video.mp4 \
    --yolo-model yolov8n.pt \
    --target-fps 5 \
    --num-samples 2 \
    --prediction-horizon 20
```

### For Accuracy
```bash
# Larger model + higher FPS + more samples
python main.py --video video.mp4 \
    --yolo-model yolov8x.pt \
    --target-fps 15 \
    --num-samples 8 \
    --prediction-horizon 40
```

### For GPU Memory
```bash
# Reduce resolution
python main.py --video video.mp4 \
    --frame-width 480 \
    --frame-height 360
```

## Troubleshooting

### "CUDA out of memory"
→ Use smaller model: `--yolo-model yolov8n.pt`
→ Reduce samples: `--num-samples 2`

### "No agents detected"
→ Lower confidence: `--confidence 0.3`
→ Use larger model: `--yolo-model yolov8l.pt`

### "Processing is slow"
→ Enable GPU: Check `nvidia-smi` shows usage
→ Reduce FPS: `--target-fps 5`
→ Use smaller model: `--yolo-model yolov8n.pt`

### Import errors
```bash
# Reinstall
pip install -r requirements.txt --upgrade
```

## Understanding the Algorithm

### Pipeline Flow
1. **Preprocess** → Normalize FPS, resize frames
2. **Detect** → Find vehicles, pedestrians (YOLOv8)
3. **Track** → Assign consistent IDs (ByteTrack)
4. **Extract** → Compute velocities, accelerations
5. **Encode** → Compress motion to latent vectors (LSTM)
6. **Sample** → Generate possible futures (diffusion)
7. **Evaluate** → Score each future for risk
8. **Aggregate** → Combine scores, smooth over time
9. **Output** → Save results, annotate video

### Key Parameters

| Parameter | Impact | Default | Range |
|-----------|--------|---------|-------|
| `target_fps` | Temporal granularity | 10 | 5-30 |
| `num_samples` | Robustness | 5 | 2-10 |
| `prediction_horizon` | Look-ahead time | 30 | 10-60 |
| `temporal_alpha` | Smoothing | 0.3 | 0.1-0.5 |
| `confidence` | Detection quality | 0.5 | 0.3-0.7 |

### Computational Complexity
- YOLOv8 Detection: O(W×H) per frame
- ByteTrack: O(N²) per frame (N = agents)
- LSTM Encoding: O(N×T_obs) 
- Trajectory Sampling: O(N×K×T_future)
- Risk Evaluation: O(N²×K×T_future)

**Total: ~2-5× real-time with GPU**

## Results Analysis

### Postprocessing with Pandas
```python
import pandas as pd

# Load results
df = pd.read_csv('output/risk_scores.csv')

# Find high-risk moments
critical = df[df['smoothed_risk'] > 0.7]
print(f"Critical moments: {len(critical)}")

# Group by agent
by_agent = df.groupby('track_id')['smoothed_risk'].agg(['mean', 'max'])
print(by_agent)

# Time-series plot
df.set_index('timestamp').groupby('track_id')['smoothed_risk'].plot()
```

## Next Steps

1. **Try test video**: Copy sample video to `data/videos/`
2. **Run pipeline**: `python main.py --video data/videos/sample.mp4`
3. **Check outputs**: Look in `output/` folder
4. **Adjust parameters**: Rerun with different settings
5. **Integrate into app**: Use CSV outputs in your system

## Support

- **Documentation**: See `README.md`
- **Module details**: Each `.py` file has docstrings
- **Tests**: Run `python test_components.py`
- **Debugging**: Enable `--log-level DEBUG`

Good luck! 🚗🎯
