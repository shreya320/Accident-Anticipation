#!/usr/bin/env python3
"""
Evaluate Accident Anticipation Pipeline
- Load videos with accident/non-accident labels
- Run pipeline on all videos
- Compute performance metrics (AUC, precision, recall)
- Generate visualization plots
"""

import argparse
import logging
import json
from pathlib import Path
import numpy as np
from src.pipeline import AccidentAnticipationPipeline
from src.evaluation import VideoEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_videos_from_folder(video_dir: str, accident_subfolder: str = 'accidents',
                            non_accident_subfolder: str = 'non_accidents'):
    """
    Load video paths and labels from folder structure.
    
    Expected structure:
        video_dir/
            accidents/
                video1.mp4
                video2.mp4
            non_accidents/
                video3.mp4
                video4.mp4
    
    Args:
        video_dir: Root directory containing videos
        accident_subfolder: Subfolder name for accident videos
        non_accident_subfolder: Subfolder name for non-accident videos
        
    Returns:
        list of (video_path, label) tuples, where label=1 for accidents, 0 for non-accidents
    """
    video_dir = Path(video_dir)
    videos = []
    
    # Load accident videos (label=1)
    accident_dir = video_dir / accident_subfolder
    if accident_dir.exists():
        for video_file in accident_dir.glob('*.mp4'):
            videos.append((str(video_file), 1))
        logger.info(f"Found {len([v for v in videos if v[1] == 1])} accident videos")
    else:
        logger.warning(f"Accident folder not found: {accident_dir}")
    
    # Load non-accident videos (label=0)
    non_accident_dir = video_dir / non_accident_subfolder
    if non_accident_dir.exists():
        for video_file in non_accident_dir.glob('*.mp4'):
            videos.append((str(video_file), 0))
        logger.info(f"Found {len([v for v in videos if v[1] == 0])} non-accident videos")
    else:
        logger.warning(f"Non-accident folder not found: {non_accident_dir}")
    
    return videos


def load_videos_from_csv(csv_path: str, video_dir: str = '.'):
    """
    Load video paths and labels from CSV.
    
    CSV should have columns: 'video_path', 'label' (0 or 1)
    
    Args:
        csv_path: Path to CSV file
        video_dir: Directory prefix for video paths
        
    Returns:
        list of (video_path, label) tuples
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    videos = []
    
    for _, row in df.iterrows():
        video_path = Path(video_dir) / row['video_path']
        label = int(row['label'])
        videos.append((str(video_path), label))
    
    logger.info(f"Loaded {len(videos)} videos from {csv_path}")
    return videos


def run_evaluation(videos: list, pipeline_config: dict, 
                   output_dir: str = 'evaluation_results',
                   skip_processed: bool = True):
    """
    Run pipeline on all videos and evaluate.
    
    Args:
        videos: List of (video_path, label) tuples
        pipeline_config: Configuration dict for AccidentAnticipationPipeline
        output_dir: Directory to save results
        skip_processed: Skip videos that already have risk_scores.csv
        
    Returns:
        VideoEvaluator with all video statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = AccidentAnticipationPipeline(pipeline_config)
    evaluator = VideoEvaluator(risk_threshold=0.5)
    
    logger.info(f"Processing {len(videos)} videos...")
    
    for idx, (video_path, label) in enumerate(videos, 1):
        logger.info(f"\n[{idx}/{len(videos)}] Processing: {Path(video_path).name} (label={label})")
        
        try:
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                logger.error(f"Video not found: {video_path}")
                continue
            
            # Create output directory for this video
            video_output_dir = output_dir / video_path_obj.stem
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already processed
            risk_scores_path = video_output_dir / 'risk_scores.csv'
            if skip_processed and risk_scores_path.exists():
                logger.info(f"Skipping (already processed): {risk_scores_path}")
            else:
                # Run pipeline
                logger.info(f"Running pipeline on {video_path}...")
                results = pipeline.run(
                    video_path=str(video_path),
                    output_dir=str(video_output_dir),
                    save_video=False,  # Skip MP4 to save time
                    save_csv=True
                )
                logger.info(f"Pipeline completed. Saved to {video_output_dir}")
            
            # Load risk scores
            import pandas as pd
            if risk_scores_path.exists():
                df = pd.read_csv(risk_scores_path)
                risk_scores = df['smoothed_risk'].values
                
                # Compute statistics
                stats = evaluator.compute_video_stats(
                    video_path=video_path,
                    label=label,
                    risk_scores=risk_scores,
                    frame_info={'num_agents': int(df['agent_id'].nunique())}
                )
                logger.info(f"Stats - Mean: {stats.mean_risk:.3f}, Max: {stats.max_risk:.3f}, "
                          f"P95: {stats.p95_risk:.3f}")
            else:
                logger.error(f"Risk scores file not found: {risk_scores_path}")
        
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}", exc_info=True)
            continue
    
    return evaluator


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate accident anticipation pipeline on labeled video dataset'
    )
    parser.add_argument('--video-dir', type=str, default='videos',
                       help='Directory containing videos (use folder structure or CSV)')
    parser.add_argument('--csv', type=str, default=None,
                       help='CSV file with columns: video_path, label (0 or 1)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Pipeline configuration file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Risk threshold for binary classification')
    parser.add_argument('--accident-folder', type=str, default='accidents',
                       help='Name of accident subfolder (if using folder structure)')
    parser.add_argument('--non-accident-folder', type=str, default='non_accidents',
                       help='Name of non-accident subfolder (if using folder structure)')
    parser.add_argument('--plots', action='store_true',
                       help='Generate visualization plots (ROC, PR, distribution, confusion matrix)')
    parser.add_argument('--skip-processed', action='store_true', default=True,
                       help='Skip videos that already have risk_scores.csv')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("ACCIDENT ANTICIPATION - EVALUATION")
    logger.info("=" * 80)
    
    # Load pipeline config
    if Path(args.config).exists():
        import json
        with open(args.config) as f:
            pipeline_config = json.load(f)
        logger.info(f"Loaded config from {args.config}")
    else:
        from src.pipeline import AccidentAnticipationPipeline
        pipeline_config = AccidentAnticipationPipeline.get_default_config()
        logger.info("Using default pipeline config")
    
    # Load video list
    if args.csv:
        videos = load_videos_from_csv(args.csv, video_dir=args.video_dir)
    else:
        videos = load_videos_from_folder(
            args.video_dir,
            accident_subfolder=args.accident_folder,
            non_accident_subfolder=args.non_accident_folder
        )
    
    if not videos:
        logger.error("No videos found!")
        return
    
    logger.info(f"Total videos to process: {len(videos)}")
    
    # Run evaluation
    evaluator = run_evaluation(
        videos=videos,
        pipeline_config=pipeline_config,
        output_dir=args.output_dir,
        skip_processed=args.skip_processed
    )
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results(args.output_dir)
    
    # Generate plots
    if args.plots:
        logger.info("\nGenerating plots...")
        evaluator.plot_roc_curve(Path(args.output_dir) / 'roc_curve.png')
        evaluator.plot_precision_recall(Path(args.output_dir) / 'precision_recall.png')
        evaluator.plot_risk_distribution(Path(args.output_dir) / 'risk_distribution.png')
        evaluator.plot_confusion_matrix(Path(args.output_dir) / 'confusion_matrix.png')
        logger.info("Plots saved to evaluation results directory")
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
