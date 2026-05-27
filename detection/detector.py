"""
detection/detector.py
─────────────────────
Player & ball detection using YOLOv8.
Supports both COCO-pretrained weights and custom football-tuned weights.
"""

from __future__ import annotations
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass, field
from pathlib import Path


# ─── COCO class IDs ───────────────────────────────────────────────────────────
PERSON_CLASS_ID      = 0
SPORTS_BALL_CLASS_ID = 32

# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    model_path: str  = "yolov8x.pt"   # swap with custom .pt for better ball detection
    conf_player: float = 0.40
    conf_ball:   float = 0.25          # lower → catches fast-moving ball more often
    imgsz:       int   = 1280          # higher → better small-object (ball) detection
    device:      str   = "cuda"        # "cpu" if no GPU


@dataclass
class FrameDetections:
    players:      sv.Detections
    ball:         sv.Detections
    frame_index:  int
    timestamp_s:  float


# ─── Detector ─────────────────────────────────────────────────────────────────

class FootballDetector:
    """
    Wraps YOLOv8 for football-specific detection.

    Usage
    -----
    detector = FootballDetector(DetectionConfig())
    for frame_det in detector.iter_video("match.mp4"):
        # frame_det.players  → sv.Detections
        # frame_det.ball     → sv.Detections
    """

    def __init__(self, config: DetectionConfig = DetectionConfig()):
        self.cfg   = config
        self.model = YOLO(config.model_path)
        print(f"[Detector] Loaded model: {config.model_path}")

    # ── Public API ──────────────────────────────────────────────────────────

    def detect_frame(self, frame: np.ndarray, frame_index: int = 0,
                     fps: float = 25.0) -> FrameDetections:
        """Run detection on a single BGR frame."""
        results = self.model(
            frame,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False
        )[0]

        all_det = sv.Detections.from_ultralytics(results)

        players = self._filter(all_det, PERSON_CLASS_ID,      self.cfg.conf_player)
        ball    = self._filter(all_det, SPORTS_BALL_CLASS_ID, self.cfg.conf_ball)

        return FrameDetections(
            players=players,
            ball=ball,
            frame_index=frame_index,
            timestamp_s=frame_index / max(fps, 1e-6),
        )

    def iter_video(self, video_path: str):
        """
        Generator — yields FrameDetections for every frame in the video.

        Example
        -------
        for fd in detector.iter_video("match.mp4"):
            print(fd.players)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield self.detect_frame(frame, frame_index, fps)
                frame_index += 1
        finally:
            cap.release()

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _filter(detections: sv.Detections,
                class_id: int,
                min_conf: float) -> sv.Detections:
        mask = (detections.class_id == class_id)
        if detections.confidence is not None:
            mask &= (detections.confidence >= min_conf)
        return detections[mask]