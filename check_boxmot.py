import sys
sys.path.insert(0, '/path')
try:
    import boxmot
    from boxmot import BYTETracker
    print("✓ boxmot installed successfully")
    print(f"✓ BYTETracker can be imported")
except ImportError as e:
    print(f"✗ boxmot import failed: {e}")
