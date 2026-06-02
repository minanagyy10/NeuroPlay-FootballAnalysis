from __future__ import annotations
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass
from pathlib import Path


# New model classes: 0=ball, 1=goalkeeper, 2=player, 3=referee
BALL_CLASS_ID       = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID     = 2
REFEREE_CLASS_ID    = 3


@dataclass
class DetectionConfig:
    model_path:  str   = "yolov8x.pt"
    conf_player: float = 0.45
    conf_ball:   float = 0.10
    imgsz:       int   = 1280
    device:      str   = "cuda"


@dataclass
class FrameDetections:
    players:     sv.Detections
    ball:        sv.Detections
    frame_index: int
    timestamp_s: float


class FootballDetector:

    def __init__(self, config: DetectionConfig = DetectionConfig()):
        self.cfg   = config
        self.model = YOLO(config.model_path)
        print(f"[Detector] Loaded model: {config.model_path}")

    def detect_frame(self, frame: np.ndarray, frame_index: int = 0,
                     fps: float = 25.0) -> FrameDetections:
        results = self.model(
            frame,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False
        )[0]

        all_det = sv.Detections.from_ultralytics(results)

        players = self._filter_multi(
            all_det,
            [PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID, REFEREE_CLASS_ID],
            self.cfg.conf_player
        )
        ball = self._filter(all_det, BALL_CLASS_ID, self.cfg.conf_ball)

        return FrameDetections(
            players=players,
            ball=ball,
            frame_index=frame_index,
            timestamp_s=frame_index / max(fps, 1e-6),
        )

    def iter_video(self, video_path: str):
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

    @staticmethod
    def _filter(detections: sv.Detections,
                class_id: int,
                min_conf: float) -> sv.Detections:
        mask = (detections.class_id == class_id)
        if detections.confidence is not None:
            mask &= (detections.confidence >= min_conf)
        return detections[mask]

    @staticmethod
    def _filter_multi(detections: sv.Detections,
                      class_ids: list,
                      min_conf: float) -> sv.Detections:
        mask = np.isin(detections.class_id, class_ids)
        if detections.confidence is not None:
            mask &= (detections.confidence >= min_conf)
        return detections[mask]