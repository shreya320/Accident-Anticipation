#!/usr/bin/env python
"""Debug detection to see what YOLO is actually detecting."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    from ultralytics import YOLO
    from src.preprocessing import VideoPreprocessor
    
    logger.info("Loading YOLO model...")
    model = YOLO('yolov8m.pt')
    
    logger.info("Loading video...")
    video_path = Path('data/videos/08PPPXtzN4A.mp4')
    preprocessor = VideoPreprocessor(target_fps=10, target_width=640, target_height=480)
    preprocessor.open_video(str(video_path))
    
    frames = []
    for frame, idx, timestamp in preprocessor.get_frame_generator():
        frames.append(frame)
        if len(frames) >= 5:  # Just test 5 frames
            break
    
    preprocessor.close_video()
    logger.info(f"Loaded {len(frames)} frames")
    
    # Test detection on each frame
    for frame_idx, frame in enumerate(frames):
        logger.info(f"\nFrame {frame_idx}:")
        logger.info(f"  Frame shape: {frame.shape}")
        
        # Run YOLO detection
        results = model(frame, conf=0.5, device='cpu', verbose=False)
        
        for result in results:
            if result.boxes is None:
                logger.info("  No detections found")
                continue
            
            logger.info(f"  Raw YOLO detections: {len(result.boxes)} boxes")
            
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]  # Get actual YOLO class name
                
                logger.info(f"    Box {i}: class_id={class_id}, class_name={class_name}, conf={confidence:.2f}")

if __name__ == '__main__':
    main()
