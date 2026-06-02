from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2

from detection.detector     import FootballDetector, DetectionConfig
from tracking.tracker       import FootballTracker
from calibration.homography import HomographyCalibrator
from analytics.speed        import SpeedAnalyzer, SpeedConfig
from analytics.heatmap      import HeatmapGenerator
from analytics.events       import EventDetector, EventConfig
from tactical_ai.coach      import TacticalCoach, TacticalSituation
from utils.annotator        import FootballAnnotator
from utils.camera           import CameraAutoDetector
from utils.reid             import ReIDSystem


@dataclass
class PipelineConfig:
    video_path:        str
    output_path:       str          = "output/annotated.mp4"
    model_path:        str          = "yolov8x.pt"
    device:            str          = "cuda"
    pixel_pts:         Optional[np.ndarray] = None
    world_pts:         Optional[np.ndarray] = None
    save_heatmaps:     bool = True
    save_report:       bool = True
    heatmap_dir:       str  = "output/heatmaps"
    report_path:       str  = "output/report.json"
    enable_tactical:   bool = True
    tactical_interval: int  = 25
    anthropic_api_key: Optional[str] = None
    max_frames:        Optional[int] = None
    frame_skip:        int  = 1


class FootballPipeline:

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        print("[Pipeline] Initialising modules...")

        cam_detector = CameraAutoDetector()
        cam_profile  = cam_detector.detect(config.video_path)

        det_config = DetectionConfig(
            model_path  = config.model_path,
            device      = config.device,
            conf_player = cam_profile.conf_player,
            conf_ball   = cam_profile.conf_ball,
            imgsz       = cam_profile.imgsz,
        )

        self.detector       = FootballDetector(det_config)
        self.tracker        = FootballTracker(
            track_buffer=cam_profile.track_buffer,
        )
        self.reid           = ReIDSystem(
            color_threshold    = 0.55,
            position_threshold = 350.0,
            max_lost_frames    = 500,
        )
        self.calibrator     = self._setup_calibrator()
        self.speed_analyzer = SpeedAnalyzer(
            self.calibrator,
            SpeedConfig(max_speed_kmh=cam_profile.max_speed_kmh),
        )
        self.heatmap_gen    = HeatmapGenerator()
        self.event_detector = EventDetector(EventConfig())
        self.tactical_coach = TacticalCoach() if config.enable_tactical else None
        self.annotator      = FootballAnnotator()

        self._tactical_log: list = []
        print("[Pipeline] Ready.")

    def run(self) -> dict:
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

                if self.cfg.max_frames and frame_index >= self.cfg.max_frames:
                    break

                if frame_index % self.cfg.frame_skip != 0:
                    frame_index += 1
                    continue

                timestamp_s = frame_index / fps

                # ── Detection + Tracking ──────────────────────────────────
                fd = self.detector.detect_frame(frame, frame_index, fps)
                tf = self.tracker.update(fd)

                # ── Re-ID ─────────────────────────────────────────────────
                id_mapping = {}
                if tf.players.tracker_id is not None and len(tf.players) > 0:
                    id_mapping = self.reid.update(
                        frame       = frame,
                        boxes_xyxy  = tf.players.xyxy,
                        tracker_ids = tf.players.tracker_id,
                        frame_index = frame_index,
                    )

                # ── Speed + Heatmap using stable IDs ──────────────────────
                player_world: dict = {}
                for tid, (cx, cy) in tf.player_centers.items():
                    stable_id = id_mapping.get(int(tid), int(tid))

                    if self.calibrator.is_calibrated:
                        wx, wy = self.calibrator.to_world(cx, cy)
                    else:
                        wx, wy = cx / W * 105, cy / H * 68

                    player_world[stable_id] = (wx, wy)
                    self.speed_analyzer.update(stable_id, cx, cy, timestamp_s)
                    self.heatmap_gen.add_position(stable_id, wx, wy)

                # ── Ball ──────────────────────────────────────────────────
                ball_world = None
                if tf.ball_center:
                    bcx, bcy = tf.ball_center
                    if self.calibrator.is_calibrated:
                        ball_world = self.calibrator.to_world(bcx, bcy)
                    else:
                        ball_world = (bcx / W * 105, bcy / H * 68)

                # ── Events ────────────────────────────────────────────────
                events = self.event_detector.update(
                    frame_index             = frame_index,
                    timestamp_s             = timestamp_s,
                    ball_world              = ball_world,
                    player_world_positions  = player_world,
                )
                recent_events = (recent_events + events)[-5:]

                # ── Tactical AI ───────────────────────────────────────────
                if (self.tactical_coach
                        and frame_index % self.cfg.tactical_interval == 0
                        and ball_world and player_world):
                    situation = TacticalSituation(
                        ball_position       = ball_world,
                        possessor_id        = list(player_world.keys())[0],
                        teammate_positions  = list(player_world.values()),
                        opponent_positions  = [],
                        frame_index         = frame_index,
                        timestamp_s         = timestamp_s,
                    )
                    advice = self.tactical_coach.analyse_sync(situation)
                    self._tactical_log.append({
                        "frame":  frame_index,
                        "t":      round(timestamp_s, 2),
                        "action": advice.recommended_action,
                        "xg":     advice.xg_if_shot,
                        "xt":     advice.xt_current,
                        "reason": advice.reason,
                    })

                # ── Annotation ────────────────────────────────────────────
                annotated = self.annotator.annotate(
                    frame            = frame,
                    players          = tf.players,
                    ball             = tf.ball,
                    speed_stats      = self.speed_analyzer.all_stats(),
                    ball_center      = tf.ball_center,
                    events           = recent_events if recent_events else None,
                    player_positions = player_world if player_world else None,
                )
                writer.write(annotated)

                if frame_index % 50 == 0:
                    unique = self.reid.total_unique_players()
                    elapsed = time.time() - start_time
                    print(f"  Frame {frame_index:5d} | "
                          f"{timestamp_s:6.1f}s | "
                          f"Players: {len(tf.player_centers):2d} | "
                          f"Unique IDs: {unique:3d} | "
                          f"Ball: {'✓' if ball_world else '✗'} | "
                          f"Elapsed: {elapsed:.1f}s")

                frame_index += 1

        finally:
            cap.release()
            writer.release()

        print(f"[Pipeline] Video saved → {self.cfg.output_path}")
        report = self._build_report(frame_index, fps)

        if self.cfg.save_heatmaps:
            self.heatmap_gen.save_all(
                self.cfg.heatmap_dir,
                list(set(self.speed_analyzer.all_stats().keys())),
            )

        if self.cfg.save_report:
            Path(self.cfg.report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cfg.report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"[Pipeline] Report saved → {self.cfg.report_path}")

        return report

    def _setup_calibrator(self) -> HomographyCalibrator:
        calib = HomographyCalibrator()
        if self.cfg.pixel_pts is not None and self.cfg.world_pts is not None:
            calib.calibrate(
                np.array(self.cfg.pixel_pts, dtype=np.float32),
                np.array(self.cfg.world_pts,  dtype=np.float32),
            )
        else:
            print("[Pipeline] No calibration points supplied. Using approx "
                  "full-frame mapping. Pass pixel_pts + world_pts for accurate "
                  "speed/distance.")
        return calib

    def _build_report(self, total_frames: int, fps: float) -> dict:
        duration_s = total_frames / fps
        return {
            "video":            self.cfg.video_path,
            "duration_s":       round(duration_s, 1),
            "total_frames":     total_frames,
            "unique_players":   self.reid.total_unique_players(),
            "speed_stats":      self.speed_analyzer.summary_table(),
            "events":           dict(self.event_detector.get_summary()),
            "tactical_log":     self._tactical_log[-20:],
            "formation":        self.heatmap_gen.detect_formation(
                                    list(self.speed_analyzer.all_stats().keys())
                                ),
        }