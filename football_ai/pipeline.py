"""
pipeline.py
───────────
Main Football AI Analysis Pipeline.

Wires together:
  Detection → Tracking → Speed → Events → Tactical AI → Annotation

Usage
-----
from pipeline import FootballPipeline, PipelineConfig

cfg = PipelineConfig(
    model_path  = "models/yolov8x-football.pt",
    video_path  = "match.mp4",
    output_path = "output/annotated.mp4",
    pixel_pts   = [...],   # field corner pixels
    world_pts   = [...],   # field corner metres
)

pipeline = FootballPipeline(cfg)
report   = pipeline.run()

print(report)
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2

from detection.detector   import FootballDetector, DetectionConfig
from tracking.tracker     import FootballTracker
from calibration.homography import HomographyCalibrator
from analytics.speed      import SpeedAnalyzer, SpeedConfig
from analytics.heatmap    import HeatmapGenerator
from analytics.events     import EventDetector, EventConfig
from tactical_ai.coach    import TacticalCoach, TacticalSituation
from utils.annotator      import FootballAnnotator


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    video_path:        str
    output_path:       str         = "output/annotated.mp4"
    model_path:        str         = "yolov8x.pt"
    device:            str         = "cuda"

    # Camera calibration (set to None for auto-estimate from frame size)
    pixel_pts:         Optional[np.ndarray] = None
    world_pts:         Optional[np.ndarray] = None

    # Outputs
    save_heatmaps:     bool = True
    save_report:       bool = True
    heatmap_dir:       str  = "output/heatmaps"
    report_path:       str  = "output/report.json"

    # Tactical AI
    enable_tactical:   bool = True
    tactical_interval: int  = 25       # frames between tactical analyses
    anthropic_api_key: Optional[str] = None   # set for LLM-enhanced advice

    # Processing
    max_frames:        Optional[int] = None   # None = full video
    frame_skip:        int = 1                # process every N frames (1 = all)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class FootballPipeline:

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

        print("[Pipeline] Initialising modules...")
        self.detector  = FootballDetector(
            DetectionConfig(model_path=config.model_path, device=config.device)
        )
        self.tracker   = FootballTracker()
        self.calibrator = self._setup_calibrator()
        self.speed_analyzer = SpeedAnalyzer(self.calibrator)
        self.heatmap_gen    = HeatmapGenerator()
        self.event_detector = EventDetector(EventConfig())
        self.tactical_coach = TacticalCoach() if config.enable_tactical else None
        self.annotator      = FootballAnnotator()

        self._tactical_log: list = []
        print("[Pipeline] Ready.")

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Run full analysis pipeline. Returns summary report dict."""
        Path(self.cfg.output_path).parent.mkdir(parents=True, exist_ok=True)

        cap   = cv2.VideoCapture(self.cfg.video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            self.cfg.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H)
        )

        start_time    = time.time()
        frame_index   = 0
        recent_events = []

        print(f"[Pipeline] Processing: {self.cfg.video_path}")
        print(f"[Pipeline] Resolution: {W}×{H}  FPS: {fps:.1f}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                max_f = self.cfg.max_frames
                if max_f and frame_index >= max_f:
                    break

                # ── Frame skip ────────────────────────────────────────────
                if frame_index % self.cfg.frame_skip != 0:
                    frame_index += 1
                    continue

                timestamp_s = frame_index / fps

                # ── Detection ─────────────────────────────────────────────
                fd = self.detector.detect_frame(frame, frame_index, fps)

                # ── Tracking ──────────────────────────────────────────────
                tf = self.tracker.update(fd)

                # ── Speed + heatmap ───────────────────────────────────────
                player_world: dict = {}
                for tid, (cx, cy) in tf.player_centers.items():
                    if self.calibrator.is_calibrated:
                        wx, wy = self.calibrator.to_world(cx, cy)
                    else:
                        wx, wy = cx / W * 105, cy / H * 68

                    player_world[tid] = (wx, wy)
                    self.speed_analyzer.update(tid, cx, cy, timestamp_s)
                    self.heatmap_gen.add_position(tid, wx, wy)

                # ── Ball world position ───────────────────────────────────
                ball_world = None
                if tf.ball_center:
                    bcx, bcy = tf.ball_center
                    if self.calibrator.is_calibrated:
                        ball_world = self.calibrator.to_world(bcx, bcy)
                    else:
                        ball_world = (bcx / W * 105, bcy / H * 68)

                # ── Events ────────────────────────────────────────────────
                events = self.event_detector.update(
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    ball_world=ball_world,
                    player_world_positions=player_world,
                )
                recent_events = (recent_events + events)[-5:]

                # ── Tactical AI ───────────────────────────────────────────
                if (self.tactical_coach
                        and frame_index % self.cfg.tactical_interval == 0
                        and ball_world and player_world):

                    situation = TacticalSituation(
                        ball_position=ball_world,
                        possessor_id=list(player_world.keys())[0],
                        teammate_positions=list(player_world.values()),
                        opponent_positions=[],    # team separation in Phase 3
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                    )
                    advice = self.tactical_coach.analyse_sync(situation)
                    self._tactical_log.append({
                        "frame": frame_index,
                        "t":     round(timestamp_s, 2),
                        "action": advice.recommended_action,
                        "xg":    advice.xg_if_shot,
                        "xt":    advice.xt_current,
                        "reason": advice.reason,
                    })

                # ── Annotation ────────────────────────────────────────────
                annotated = self.annotator.annotate(
                    frame           = frame,
                    players         = tf.players,
                    ball            = tf.ball,
                    speed_stats     = self.speed_analyzer.all_stats(),
                    ball_center     = tf.ball_center,
                    events          = recent_events if recent_events else None,
                    player_positions = player_world if player_world else None,
                )

                writer.write(annotated)

                if frame_index % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Frame {frame_index:5d} | "
                          f"{timestamp_s:6.1f}s | "
                          f"Players: {len(tf.player_centers):2d} | "
                          f"Ball: {'✓' if ball_world else '✗'} | "
                          f"Elapsed: {elapsed:.1f}s")

                frame_index += 1

        finally:
            cap.release()
            writer.release()

        print(f"[Pipeline] Video saved → {self.cfg.output_path}")

        # ── Post-processing ───────────────────────────────────────────────
        report = self._build_report(frame_index, fps)

        if self.cfg.save_heatmaps:
            self.heatmap_gen.save_all(
                self.cfg.heatmap_dir,
                list(self.tracker.player_states.keys()),
            )

        if self.cfg.save_report:
            Path(self.cfg.report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cfg.report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"[Pipeline] Report saved → {self.cfg.report_path}")

        return report

    # ── Internals ───────────────────────────────────────────────────────────

    def _setup_calibrator(self) -> HomographyCalibrator:
        calib = HomographyCalibrator()
        if self.cfg.pixel_pts is not None and self.cfg.world_pts is not None:
            calib.calibrate(
                np.array(self.cfg.pixel_pts, dtype=np.float32),
                np.array(self.cfg.world_pts,  dtype=np.float32),
            )
        else:
            print("[Pipeline] No calibration points supplied. "
                  "Using approx full-frame mapping. "
                  "Pass pixel_pts + world_pts for accurate speed/distance.")
        return calib

    def _build_report(self, total_frames: int, fps: float) -> dict:
        duration_s = total_frames / fps
        return {
            "video":     self.cfg.video_path,
            "duration_s": round(duration_s, 1),
            "total_frames": total_frames,
            "players_tracked": len(self.tracker.player_states),
            "speed_stats": self.speed_analyzer.summary_table(),
            "events":    {
                k: v for k, v in
                self.event_detector.get_summary().items()
            },
            "tactical_log": self._tactical_log[-20:],  # last 20 analyses
            "formation": self.heatmap_gen.detect_formation(
                list(self.tracker.player_states.keys())
            ),
        }