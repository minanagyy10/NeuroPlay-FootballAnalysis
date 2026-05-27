"""
utils/annotator.py
──────────────────
Video frame annotation utilities.

Draws:
  - Bounding boxes with player IDs
  - Speed overlay (km/h)
  - Ball trail
  - Event banners
  - Mini radar (top-down position map)
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import supervision as sv


# ─── Color palette ───────────────────────────────────────────────────────────
COLOR_PLAYER  = (30,  144, 255)   # dodger blue
COLOR_BALL    = (0,   255, 128)   # mint green
COLOR_TEXT    = (255, 255, 255)   # white
COLOR_BANNER  = (220,  20,  60)   # crimson (events)
COLOR_SPEED   = (255, 200,   0)   # amber
COLOR_RADAR_BG = (20,  60,  20)   # dark green


class FootballAnnotator:
    """
    Draws rich annotations onto video frames.

    Usage
    -----
    ann = FootballAnnotator()

    frame = ann.annotate(
        frame          = bgr_frame,
        tracked_frame  = tf,
        speed_stats    = analyzer.all_stats(),
        events         = recent_events,
    )
    """

    def __init__(
        self,
        trail_length: int = 20,     # ball trail history
        radar_size:   int = 200,    # pixels of mini radar overlay
    ) -> None:
        self._box_ann   = sv.BoxAnnotator(color=sv.Color.from_bgr_tuple(COLOR_PLAYER))
        self._label_ann = sv.LabelAnnotator(
            color=sv.Color.from_bgr_tuple(COLOR_PLAYER),
            text_color=sv.Color.from_bgr_tuple(COLOR_TEXT),
        )
        self._ball_trail: deque = deque(maxlen=trail_length)
        self._radar_size = radar_size

    # ── Main entry point ────────────────────────────────────────────────────

    def annotate(
        self,
        frame:          np.ndarray,
        players:        sv.Detections,
        ball:           sv.Detections,
        speed_stats:    Optional[Dict] = None,
        ball_center:    Optional[Tuple[float, float]] = None,
        events:         Optional[List] = None,
        player_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        out = frame.copy()

        # Players
        out = self._draw_players(out, players, speed_stats)

        # Ball + trail
        if ball_center:
            self._ball_trail.append(ball_center)
        out = self._draw_ball_trail(out)
        out = self._draw_ball(out, ball)

        # Event banner
        if events:
            out = self._draw_event_banner(out, events[-1])

        # Radar
        if player_positions:
            radar = self._build_radar(player_positions, ball_center)
            out   = self._overlay_radar(out, radar)

        return out

    # ── Player annotations ───────────────────────────────────────────────────

    def _draw_players(
        self,
        frame:       np.ndarray,
        players:     sv.Detections,
        speed_stats: Optional[Dict],
    ) -> np.ndarray:
        if players.tracker_id is None or len(players) == 0:
            return frame

        labels = []
        for tid in players.tracker_id:
            tid = int(tid)
            if speed_stats and tid in speed_stats:
                s = speed_stats[tid]
                labels.append(f"#{tid}  {s.current_speed_kmh:.1f} km/h")
            else:
                labels.append(f"#{tid}")

        frame = self._box_ann.annotate(frame, players)
        frame = self._label_ann.annotate(frame, players, labels)
        return frame

    # ── Ball ────────────────────────────────────────────────────────────────

    def _draw_ball(self, frame: np.ndarray, ball: sv.Detections) -> np.ndarray:
        for box in ball.xyxy:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            cv2.circle(frame, (cx, cy), 8,  COLOR_BALL, -1)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
        return frame

    def _draw_ball_trail(self, frame: np.ndarray) -> np.ndarray:
        trail = list(self._ball_trail)
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            color = tuple(int(c * alpha) for c in COLOR_BALL)
            thickness = max(1, int(alpha * 3))
            p1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            p2 = (int(trail[i][0]),     int(trail[i][1]))
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
        return frame

    # ── Event banner ─────────────────────────────────────────────────────────

    def _draw_event_banner(
        self, frame: np.ndarray, event
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        text = f"  {event.event_type.name}  "
        if event.player_id is not None:
            text += f"| Player #{event.player_id}  "

        font      = cv2.FONT_HERSHEY_DUPLEX
        scale     = 0.9
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

        # Semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 20), (tw + 20, 20 + th + 16), COLOR_BANNER, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame, text, (10, 20 + th + 6),
            font, scale, COLOR_TEXT, thickness, cv2.LINE_AA
        )
        return frame

    # ── Mini radar ───────────────────────────────────────────────────────────

    def _build_radar(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        ball_center:      Optional[Tuple[float, float]],
        pitch_w: float = 105.0,
        pitch_h: float = 68.0,
    ) -> np.ndarray:
        s = self._radar_size
        radar = np.full((int(s * 68 / 105), s, 3), COLOR_RADAR_BG, dtype=np.uint8)
        rh, rw = radar.shape[:2]

        # Pitch lines
        cv2.rectangle(radar, (0, 0), (rw - 1, rh - 1), (255, 255, 255), 1)
        cv2.line(radar, (rw // 2, 0), (rw // 2, rh), (255, 255, 255), 1)

        def to_radar(wx: float, wy: float) -> Tuple[int, int]:
            px = int(np.clip(wx / pitch_w * rw, 0, rw - 1))
            py = int(np.clip(wy / pitch_h * rh, 0, rh - 1))
            return px, py

        # Players
        for tid, (wx, wy) in player_positions.items():
            px, py = to_radar(wx, wy)
            cv2.circle(radar, (px, py), 4, COLOR_PLAYER, -1)

        # Ball
        if ball_center:
            bx, by = ball_center
            px, py = to_radar(bx, by)
            cv2.circle(radar, (px, py), 5, COLOR_BALL, -1)

        return radar

    def _overlay_radar(
        self, frame: np.ndarray, radar: np.ndarray
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        rh, rw = radar.shape[:2]
        margin = 10

        # Bottom-right corner
        x1 = w - rw - margin
        y1 = h - rh - margin
        x2, y2 = x1 + rw, y1 + rh

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        frame[y1:y2, x1:x2] = radar
        return frame