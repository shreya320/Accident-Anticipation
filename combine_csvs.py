#!/usr/bin/env python
"""
Combine all per-video state encodings CSVs into one master CSV.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

output_dir = Path('output')

# Find all state_encodings_*.csv files
state_files = sorted(output_dir.glob('state_encodings_*.csv'))
print(f"Found {len(state_files)} state encoding files")

# Combine them
dfs = []
for fpath in state_files:
    try:
        df = pd.read_csv(fpath)
        if len(df) > 0:
            dfs.append(df)
            print(f"  - {fpath.name}: {len(df)} rows")
        else:
            print(f"  - {fpath.name}: (empty, skipped)")
    except Exception as e:
        print(f"  - {fpath.name}: (error: {e}, skipped)")

combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined: {len(combined_df)} total rows")

# Save master
master_file = output_dir / f"ALL_state_encodings_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
combined_df.to_csv(master_file, index=False)
print(f"✓ Saved to {master_file.name}")
print(f"  Columns: {list(combined_df.columns)}")
print(f"  Videos: {combined_df['video'].nunique()}")
print(f"  Total tracks: {combined_df['track_id'].nunique()}")
print(f"  Total encoded states: {len(combined_df)}")
