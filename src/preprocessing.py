"""
Video Preprocessing Module
- Loads MP4 video files
- Normalizes FPS to a target value
- Resizes frames for YOLO inference
- Handles frame iteration with metadata
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Generator
import logging

logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """Handles video loading, FPS normalization, and frame resizing."""
    
    def __init__(self, target_fps: int = 10, target_width: int = 640, target_height: int = 480):
        """
        Initialize preprocessor.
        
        Args:
            target_fps: Target FPS for normalization (default: 10 FPS)
            target_width: Target width for resizing (default: 640)
            target_height: Target height for resizing (default: 480)
        """
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.cap = None
        self.video_path = None
        self.original_fps = None
        self.total_frames = None
        self.frame_skip = 1
        self.width = None
        self.height = None
    
    def open_video(self, video_path: str) -> bool:
        """
        Open and validate a video file.
        
        Args:
            video_path: Path to the MP4 video file
            
        Returns:
            bool: True if video opened successfully
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        if not video_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            logger.warning(f"Unsupported video format: {video_path.suffix}")
        
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return False
        
        self.video_path = video_path
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame skip for FPS normalization
        self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        normalized_fps = self.original_fps / self.frame_skip
        
        logger.info(f"Video loaded: {video_path}")
        logger.info(f"Original FPS: {self.original_fps:.2f}, Target FPS: {self.target_fps}")
        logger.info(f"Normalized FPS: {normalized_fps:.2f}, Frame skip: {self.frame_skip}")
        logger.info(f"Original resolution: {self.width}x{self.height}")
        logger.info(f"Target resolution: {self.target_width}x{self.target_height}")
        logger.info(f"Total frames: {self.total_frames}")
        
        return True
    
    def close_video(self):
        """Release video capture resource."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video closed")
    
    def get_frame_generator(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generate preprocessed frames from video.
        Yields original frame, frame index, and timestamp.
        
        Yields:
            tuple: (resized_frame, frame_idx, timestamp_sec)
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Video not opened")
            return
        
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        actual_frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Apply frame skip for FPS normalization
            if frame_idx % self.frame_skip == 0:
                # Resize frame
                resized = cv2.resize(frame, (self.target_width, self.target_height))
                
                # Calculate timestamp (in seconds)
                timestamp = actual_frame_count / (self.original_fps / self.frame_skip)
                
                yield resized, actual_frame_count, timestamp
                actual_frame_count += 1
            
            frame_idx += 1
    
    def get_info(self) -> dict:
        """
        Get metadata about the loaded video.
        
        Returns:
            dict: Video metadata
        """
        return {
            'path': str(self.video_path),
            'original_fps': self.original_fps,
            'target_fps': self.target_fps,
            'frame_skip': self.frame_skip,
            'original_width': self.width,
            'original_height': self.height,
            'target_width': self.target_width,
            'target_height': self.target_height,
            'total_frames': self.total_frames,
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close_video()


def load_video_frames(video_path: str, target_fps: int = 10, 
                     target_width: int = 640, target_height: int = 480) -> Tuple[list, dict]:
    """
    Convenience function to load all frames from a video.
    
    Args:
        video_path: Path to video file
        target_fps: Target FPS
        target_width: Target width
        target_height: Target height
        
    Returns:
        tuple: (list of frames, metadata dict)
    """
    preprocessor = VideoPreprocessor(target_fps, target_width, target_height)
    if not preprocessor.open_video(video_path):
        return [], {}
    
    frames = []
    frame_times = []
    
    for frame, idx, timestamp in preprocessor.get_frame_generator():
        frames.append(frame)
        frame_times.append(timestamp)
    
    info = preprocessor.get_info()
    info['frame_count'] = len(frames)
    info['frame_times'] = frame_times
    
    preprocessor.close_video()
    
    return frames, info
