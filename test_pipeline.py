#!/usr/bin/env python
"""Test script to debug pipeline issues."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    logger.info("=" * 80)
    logger.info("TESTING PIPELINE")
    logger.info("=" * 80)
    
    logger.info("Step 1: Importing modules...")
    from src.pipeline import AccidentAnticipationPipeline
    logger.info("Pipeline imported successfully")
    
    logger.info("Step 2: Creating pipeline...")
    config = AccidentAnticipationPipeline.get_default_config()
    config['use_cuda'] = False
    config['target_fps'] = 10
    
    pipeline = AccidentAnticipationPipeline(config)
    logger.info("Pipeline created successfully")
    
    logger.info("Step 3: Finding test video...")
    video_path = Path('data/videos/08PPPXtzN4A.mp4')
    if video_path.exists():
        logger.info(f"Found video: {video_path}")
        logger.info(f"Video size: {video_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    logger.info("Step 4: Running pipeline...")
    results = pipeline.run(
        str(video_path),
        output_dir='output',
        save_csv=True,
        save_video=False
    )
    
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Results: {results}")
    
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
