"""
Detection & Tracking Module
- YOLOv8 for multi-class object detection
- ByteTrack for multi-object tracking
- Track IDs only (no license plates, no personal data)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed. Install via: pip install ultralytics")

try:
    from boxmot import BYTETracker
except ImportError:
    BYTETracker = None
    logging.warning("boxmot not installed. ByteTrack will not be available. Install via: pip install boxmot")

logger = logging.getLogger(__name__)


# Target object classes from COCO dataset
DETECTION_CLASSES = {
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7,
    'pedestrian': 0,
}

REVERSE_CLASSES = {v: k for k, v in DETECTION_CLASSES.items()}


@dataclass(frozen=True)
class Detection:
    """Single detection bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    
    def __hash__(self) -> int:
        """Make Detection hashable by using object id."""
        return id(self)
    
    def __eq__(self, other) -> bool:
        """Compare by object identity."""
        return self is other
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_xyxy(self) -> np.ndarray:
        """Return as [x1, y1, x2, y2]."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


@dataclass
class Track:
    """Single tracked object over time."""
    track_id: int
    class_name: str
    class_id: int
    detections: List[Tuple[int, Detection]] = field(default_factory=list)
    first_frame: int = -1
    last_frame: int = -1
    
    def add_detection(self, frame_idx: int, detection: Detection):
        """Add a detection to this track."""
        self.detections.append((frame_idx, detection))
        if self.first_frame == -1:
            self.first_frame = frame_idx
        self.last_frame = frame_idx
    
    def get_detection_at_frame(self, frame_idx: int) -> Optional[Detection]:
        """Get detection at a specific frame."""
        for idx, det in self.detections:
            if idx == frame_idx:
                return det
        return None
    
    def get_detections_in_range(self, start_frame: int, end_frame: int) -> List[Tuple[int, Detection]]:
        """Get all detections within frame range."""
        return [(idx, det) for idx, det in self.detections 
                if start_frame <= idx <= end_frame]
    
    def get_center_trajectory(self) -> List[Tuple[int, Tuple[float, float]]]:
        """Get trajectory of centers: [(frame_idx, (x, y)), ...]."""
        return [(idx, det.center) for idx, det in self.detections]
    
    def duration_frames(self) -> int:
        """Duration of track in frames."""
        return self.last_frame - self.first_frame + 1


@dataclass
class FrameDetections:
    """Detections for a single frame."""
    frame_idx: int
    timestamp: float
    detections: List[Detection]


class MultiAgentDetector:
    """YOLOv8-based object detector."""
    
    def __init__(self, model_name: str = 'yolov8m.pt', confidence: float = 0.5, device: str = '0'):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: YOLOv8 model name (nano, small, medium, large, xlarge)
            confidence: Confidence threshold
            device: Device to run on (0 for GPU, 'cpu' for CPU)
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device
        logger.info(f"YOLOv8 model loaded: {model_name} on device {device}")
    
    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            list of Detection objects
        """
        results = self.model(frame, conf=self.confidence, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter to target classes
                if class_id not in REVERSE_CLASSES:
                    continue
                
                class_name = REVERSE_CLASSES[class_id]
                if class_name not in DETECTION_CLASSES:
                    continue
                
                det = Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(det)
        
        return detections


class MultiAgentTracker:
    """
    ByteTrack-based multi-object tracker.
    Falls back to simple ID matching if ByteTrack unavailable.
    """
    
    def __init__(self, use_byte_track: bool = True, iou_threshold: float = 0.5):
        """
        Initialize tracker.
        
        Args:
            use_byte_track: Use ByteTrack if available
            iou_threshold: IoU threshold for association (simple fallback)
        """
        self.use_byte_track = use_byte_track and BYTETracker is not None
        self.iou_threshold = iou_threshold
        
        if self.use_byte_track:
            # ByteTrack parameters
            track_thresh = 0.5
            track_buffer = 30
            match_thresh = 0.8
            self.tracker = BYTETracker(track_thresh, track_buffer, match_thresh)
            logger.info("ByteTrack initialized")
        else:
            logger.warning("ByteTrack not available, using simple IoU-based matching")
            self.tracker = None
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        intersect_xmin = max(x1_min, x2_min)
        intersect_ymin = max(y1_min, y2_min)
        intersect_xmax = min(x1_max, x2_max)
        intersect_ymax = min(y1_max, y2_max)
        
        if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
            return 0.0
        
        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersect_area
        
        return intersect_area / union_area if union_area > 0 else 0.0
    
    def _match_detections_simple(self, detections: List[Detection]) -> Dict[Detection, int]:
        """
        Improved IoU-based matching for tracks with centroid distance heuristic.
        
        Returns:
            dict: Detection -> track_id
        """
        matched_det_to_track = {}
        used_tracks = set()
        
        # Try to match detections to existing tracks (greedy, by IoU + distance)
        for det in detections:
            best_track_id = -1
            best_score = 0  # Combined IoU + distance score
            
            det_center_x = det.center[0]
            det_center_y = det.center[1]
            
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                
                # Only match same class
                if track.class_name != det.class_name:
                    continue
                
                # Get last detection in track
                if track.detections:
                    _, last_det = track.detections[-1]
                    
                    # Calculate IoU
                    iou = self._iou(last_det.to_xyxy(), det.to_xyxy())
                    
                    # Calculate normalized centroid distance (lower is better)
                    dx = det_center_x - last_det.center[0]
                    dy = det_center_y - last_det.center[1]
                    
                    max_dim = max(det.width, det.height, last_det.width, last_det.height)
                    dist_norm = (dx*dx + dy*dy) ** 0.5 / (max_dim + 1e-6)
                    
                    # Combined score: IoU is 0-1, normalize distance penalty to 0-1 range
                    distance_penalty = min(1.0, dist_norm / 2.0)
                    combined_score = iou * 0.6 + (1.0 - distance_penalty) * 0.4
                    
                    # Match if above combined threshold
                    if combined_score > best_score and combined_score > 0.3:
                        best_score = combined_score
                        best_track_id = track_id
            
            if best_track_id != -1:
                matched_det_to_track[det] = best_track_id
                used_tracks.add(best_track_id)
        
        return matched_det_to_track
    
    def _match_detections_byte_track(self, detections: List[Detection]) -> Dict[Detection, int]:
        """
        ByteTrack-based matching.
        
        Returns:
            dict: Detection -> track_id
        """
        if not detections:
            return {}
        
        # Prepare detections for ByteTrack
        dets_array = np.array([det.to_xyxy() for det in detections], dtype=np.float32)
        confs = np.array([det.confidence for det in detections], dtype=np.float32)
        
        # Run ByteTrack
        tracked_objects = self.tracker.update(dets_array, confs, [])
        
        matched_det_to_track = {}
        
        # Match tracked objects back to detections (by bounding box overlap)
        for track_obj in tracked_objects:
            track_id = int(track_obj.track_id)
            x1, y1, x2, y2 = track_obj.tlbr
            
            # Find best matching detection
            best_det = None
            best_iou = 0.5
            
            for det in detections:
                if det in matched_det_to_track:
                    continue
                
                iou = self._iou(np.array([x1, y1, x2, y2]), det.to_xyxy())
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
            
            if best_det is not None:
                matched_det_to_track[best_det] = track_id
        
        return matched_det_to_track
    
    def update(self, detections: List[Detection]) -> Dict[Detection, int]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections in current frame
            
        Returns:
            dict: Detection -> track_id mapping
        """
        self.frame_count += 1
        
        # Match detections to tracks
        if self.use_byte_track:
            matched_det_to_track = self._match_detections_byte_track(detections)
        else:
            matched_det_to_track = self._match_detections_simple(detections)
        
        # Update matched tracks
        for det, track_id in matched_det_to_track.items():
            if track_id not in self.tracks:
                self.tracks[track_id] = Track(
                    track_id=track_id,
                    class_name=det.class_name,
                    class_id=det.class_id
                )
            self.tracks[track_id].add_detection(self.frame_count - 1, det)
        
        # Create new tracks for unmatched detections
        unmatched_dets = [det for det in detections if det not in matched_det_to_track]
        for det in unmatched_dets:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                class_name=det.class_name,
                class_id=det.class_id
            )
            self.tracks[track_id].add_detection(self.frame_count - 1, det)
            matched_det_to_track[det] = track_id
        
        return matched_det_to_track
    
    def get_active_tracks(self, min_duration_frames: int = 1) -> Dict[int, Track]:
        """
        Get active tracks (minimum duration).
        
        Args:
            min_duration_frames: Minimum frames for a track to be considered active
            
        Returns:
            dict: track_id -> Track
        """
        return {tid: track for tid, track in self.tracks.items()
                if track.duration_frames() >= min_duration_frames}
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get a specific track by ID."""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> Dict[int, Track]:
        """Get all tracks."""
        return self.tracks.copy()


class DetectionTrackingPipeline:
    """Integrated detection and tracking pipeline."""
    
    def __init__(self, model_name: str = 'yolov8m.pt', confidence: float = 0.5,
                 device: str = '0', use_byte_track: bool = True):
        """
        Initialize detection and tracking pipeline.
        
        Args:
            model_name: YOLOv8 model
            confidence: Detection confidence threshold
            device: Device for YOLOv8
            use_byte_track: Use ByteTrack for tracking
        """
        self.detector = MultiAgentDetector(model_name, confidence, device)
        self.tracker = MultiAgentTracker(use_byte_track)
    
    def process_frame(self, frame: np.ndarray, timestamp: float = 0.0) -> Tuple[FrameDetections, Dict[Detection, int]]:
        """
        Process single frame with detection and tracking.
        
        Args:
            frame: BGR image
            timestamp: Frame timestamp in seconds
            
        Returns:
            tuple: (FrameDetections, detection_to_track_id mapping)
        """
        # Detect objects
        detections = self.detector.detect_frame(frame)
        
        # Track objects
        det_to_track = self.tracker.update(detections)
        
        frame_det = FrameDetections(
            frame_idx=self.tracker.frame_count - 1,
            timestamp=timestamp,
            detections=detections
        )
        
        return frame_det, det_to_track
    
    def get_tracks(self, min_duration_frames: int = 1) -> Dict[int, Track]:
        """Get all tracks above minimum duration."""
        return self.tracker.get_active_tracks(min_duration_frames)
