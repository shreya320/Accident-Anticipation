#!/usr/bin/env python
"""Test if ByteTrack/boxmot is working"""

import sys
from pathlib import Path

# Test 1: Try importing boxmot
print("=" * 80)
print("TEST 1: Checking if boxmot is installed...")
print("=" * 80)

try:
    from boxmot import BYTETracker
    print("✓ SUCCESS! boxmot.BYTETracker imported successfully!")
    print(f"  BYTETracker: {BYTETracker}")
except ImportError as e:
    print(f"✗ FAILED to import boxmot: {e}")
    sys.exit(1)

# Test 2: Try running pipeline with ByteTrack
print("\n" + "=" * 80)
print("TEST 2: Running pipeline with ByteTrack enabled...")
print("=" * 80)

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.detection_tracking import DetectionTrackingPipeline, MultiAgentTracker
from src.preprocessing import VideoPreprocessor

# Check tracker initialization
tracker = MultiAgentTracker(use_byte_track=True)
print(f"Tracker initialized with ByteTrack: {tracker.use_byte_track}")

if tracker.use_byte_track:
    print("✓ ByteTrack is ENABLED!")
else:
    print("✗ ByteTrack is NOT enabled")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
