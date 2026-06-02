from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto


class CameraType(Enum):
    BROADCAST    = auto()
    DRONE        = auto()
    FIXED_WIDE   = auto()
    FIXED_CLOSE  = auto()


@dataclass
class CameraProfile:
    camera_type:    CameraType
    conf_player:    float
    conf_ball:      float
    imgsz:          int
    max_speed_kmh:  float
    track_buffer:   int
    description:    str


PROFILES = {
    CameraType.BROADCAST: CameraProfile(
        camera_type   = CameraType.BROADCAST,
        conf_player   = 0.45,
        conf_ball     = 0.10,
        imgsz         = 1280,
        max_speed_kmh = 36.0,
        track_buffer  = 40,
        description   = "TV broadcast side view",
    ),
    CameraType.DRONE: CameraProfile(
        camera_type   = CameraType.DRONE,
        conf_player   = 0.30,
        conf_ball     = 0.10,
        imgsz         = 1280,
        max_speed_kmh = 36.0,
        track_buffer  = 90,
        description   = "Drone top-down view",
    ),
    CameraType.FIXED_WIDE: CameraProfile(
        camera_type   = CameraType.FIXED_WIDE,
        conf_player   = 0.40,
        conf_ball     = 0.10,
        imgsz         = 1280,
        max_speed_kmh = 36.0,
        track_buffer  = 50,
        description   = "Fixed wide full-pitch camera",
    ),
    CameraType.FIXED_CLOSE: CameraProfile(
        camera_type   = CameraType.FIXED_CLOSE,
        conf_player   = 0.50,
        conf_ball     = 0.15,
        imgsz         = 960,
        max_speed_kmh = 36.0,
        track_buffer  = 30,
        description   = "Fixed close-up half-pitch camera",
    ),
}


class CameraAutoDetector:

    def detect(self, video_path: str, sample_frames: int = 30) -> CameraProfile:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[Camera] Cannot open video — using BROADCAST profile")
            return PROFILES[CameraType.BROADCAST]

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect = w / max(h, 1)

        green_ratios = []
        frame_diffs  = []
        prev_gray    = None

        for i in range(sample_frames):
            ret, frame = cap.read()
            if not ret:
                break

            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,
                np.array([30, 40, 40]),
                np.array([90, 255, 255]))
            green_ratios.append(mask.sum() / (w * h * 255))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                frame_diffs.append(float(diff.mean()))
            prev_gray = gray

        cap.release()

        avg_green  = float(np.mean(green_ratios)) if green_ratios else 0.5
        avg_motion = float(np.mean(frame_diffs))  if frame_diffs  else 10.0

        camera_type = self._classify(aspect, avg_green, avg_motion)
        profile     = PROFILES[camera_type]

        print(f"[Camera] Detected: {profile.description}")
        print(f"[Camera] Green={avg_green:.2f} | Motion={avg_motion:.1f} | "
              f"Aspect={aspect:.2f}")

        return profile

    def _classify(self, aspect: float, green: float, motion: float) -> CameraType:
        if green > 0.55 and motion < 8.0:
            return CameraType.DRONE
        if green > 0.40 and motion < 5.0:
            return CameraType.FIXED_WIDE
        if green < 0.30 and motion < 6.0:
            return CameraType.FIXED_CLOSE
        return CameraType.BROADCAST