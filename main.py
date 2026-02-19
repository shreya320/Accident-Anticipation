"""
Main Entry Point
Usage: python main.py --video <path_to_video> --output <output_dir>
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pipeline import AccidentAnticipationPipeline


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('accident_anticipation.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Traffic Accident Risk Prediction System'
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save annotated video')
    parser.add_argument('--save-csv', action='store_true', default=True,
                       help='Save CSV results (default: True)')
    parser.add_argument('--yolo-model', type=str, default='yolov8m.pt',
                       help='YOLOv8 model size (nano/small/medium/large/xlarge)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Latent dimension for state encoder (default: 32)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of trajectory samples per agent (default: 5)')
    parser.add_argument('--prediction-horizon', type=int, default=30,
                       help='Prediction horizon in frames (default: 30)')
    parser.add_argument('--observation-window', type=int, default=20,
                       help='Observation window in frames (default: 20)')
    parser.add_argument('--target-fps', type=int, default=10,
                       help='Target FPS for video (default: 10)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                       help='Use CUDA if available (default: True)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 1
    
    logger.info("=" * 80)
    logger.info("TRAFFIC ACCIDENT RISK PREDICTION SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {args.output}")
    
    # Create custom config
    config = AccidentAnticipationPipeline.get_default_config()
    config.update({
        'yolo_model': args.yolo_model,
        'detection_confidence': args.confidence,
        'latent_dim': args.latent_dim,
        'num_trajectory_samples': args.num_samples,
        'prediction_horizon': args.prediction_horizon,
        'observation_window': args.observation_window,
        'target_fps': args.target_fps,
        'use_cuda': args.use_cuda and not args.no_cuda,
    })
    
    # Initialize pipeline
    try:
        pipeline = AccidentAnticipationPipeline(config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1
    
    # Run pipeline
    try:
        results = pipeline.run(
            str(video_path),
            output_dir=args.output,
            save_video=args.save_video,
            save_csv=args.save_csv
        )
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Processed {results['num_tracks']} tracks across {results['num_frames']} frames")
        logger.info(f"Results saved to: {args.output}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
