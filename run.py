"""
run.py
──────
Quick-start script. Edit the paths below and run:

    python run.py

For the API server:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pipeline import FootballPipeline, PipelineConfig

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Set your video path
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH = "newmatch.mp4"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Camera calibration
#
# Find 4 known field landmarks in your video (e.g. penalty spot, corner flags).
# Measure their pixel coordinates (x, y) in the frame.
# Map them to their real-world positions in metres on a 105×68m pitch.
#
# If you don't have this yet, set pixel_pts=None and world_pts=None.
# The system will use an approximate full-frame mapping (less accurate speeds).
# ─────────────────────────────────────────────────────────────────────────────
pixel_pts = None   # Example: np.array([[120, 680], [1800, 680], [120, 40], [1800, 40]], dtype=np.float32)
world_pts = None   # Example: np.array([[0, 68], [105, 68], [0, 0], [105, 0]],         dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Configure and run
# ─────────────────────────────────────────────────────────────────────────────
cfg = PipelineConfig(
    video_path        = VIDEO_PATH,
    output_path       = "output/annotated.mp4",
    model_path        = r"runs\detect\runs\train\football_ball-3\weights\best.pt",     # swap with custom football .pt for best results
    device            = "cuda",           # or "cpu"
    pixel_pts         = pixel_pts,
    world_pts         = world_pts,
    enable_tactical   = True,
    tactical_interval = 25,               # run tactical AI every 25 frames (~1s at 25fps)
    save_heatmaps     = True,
    heatmap_dir       = "output/heatmaps",
    save_report       = True,
    report_path       = "output/report.json",
    max_frames        = None,             # None = full video; set e.g. 500 for a quick test
    frame_skip        = 1,               # 1 = every frame; 2 = every other frame (2× faster)
    anthropic_api_key = None,            # set for LLM-enhanced tactical commentary
)

pipeline = FootballPipeline(cfg)
report   = pipeline.run()

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  ANALYSIS COMPLETE")
print("═" * 60)
print(f"  Duration       : {report['duration_s']}s")
print(f"  Players tracked: {report['players_tracked']}")
print(f"  Formation      : {report['formation']}")
print(f"\n  Event counts:")
for k, v in report["events"].items():
    print(f"    {k:<25} {v}")
print(f"\n  Speed leaders:")
speed_rows = sorted(report["speed_stats"], key=lambda r: -r["max_speed_kmh"])[:5]
for r in speed_rows:
    print(f"    Player #{r['player_id']:>2} — "
          f"max {r['max_speed_kmh']} km/h | "
          f"dist {r['distance_m']} m")
print("\n  Outputs saved to output/")
print("═" * 60)