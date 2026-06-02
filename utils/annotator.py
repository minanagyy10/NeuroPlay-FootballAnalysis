from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import supervision as sv


COLOR_TEAM_A  = (255,  80,  80)   # blue
COLOR_TEAM_B  = ( 80, 255,  80)   # green
COLOR_REF     = (200, 200, 200)   # gray
COLOR_UNKNOWN = ( 30, 144, 255)   # dodger blue
COLOR_BALL    = (  0, 255, 128)   # mint
COLOR_TEXT    = (255, 255, 255)   # white
COLOR_BANNER  = (220,  20,  60)   # crimson
COLOR_RADAR_BG = (20,  60,  20)   # dark green


class FootballAnnotator:

    def __init__(
        self,
        trail_length: int = 20,
        radar_size:   int = 200,
    ) -> None:
        self._ball_trail: deque = deque(maxlen=trail_length)
        self._radar_size = radar_size

    def annotate(
        self,
        frame:            np.ndarray,
        players:          sv.Detections,
        ball:             sv.Detections,
        speed_stats:      Optional[Dict]  = None,
        ball_center:      Optional[Tuple] = None,
        events:           Optional[List]  = None,
        player_positions: Optional[Dict]  = None,
        id_to_team:       Optional[Dict]  = None,   # {stable_id: team_id}
        possession:       Optional[Dict]  = None,   # {"Team A": 55.0, ...}
        id_mapping:       Optional[Dict]  = None,   # {tracker_id: stable_id}
    ) -> np.ndarray:
        out = frame.copy()

        out = self._draw_players(out, players, speed_stats,
                                 id_to_team, id_mapping)

        if ball_center:
            self._ball_trail.append(ball_center)
        out = self._draw_ball_trail(out)
        out = self._draw_ball(out, ball)

        if events:
            out = self._draw_event_banner(out, events[-1])

        if possession and id_to_team:
            out = self._draw_possession_bar(out, possession)

        if player_positions:
            radar = self._build_radar(player_positions, ball_center,
                                      id_to_team)
            out   = self._overlay_radar(out, radar)

        return out

    # ── Players ─────────────────────────────────────────────────────────────

    def _draw_players(
        self,
        frame:      np.ndarray,
        players:    sv.Detections,
        speed_stats: Optional[Dict],
        id_to_team:  Optional[Dict],
        id_mapping:  Optional[Dict],
    ) -> np.ndarray:
        if players.tracker_id is None or len(players) == 0:
            return frame

        for box, tid in zip(players.xyxy, players.tracker_id):
            tid       = int(tid)
            stable_id = id_mapping.get(tid, tid) if id_mapping else tid
            team_id   = id_to_team.get(stable_id, -1) if id_to_team else -1

            # Pick color based on team
            if team_id == 0:
                color = COLOR_TEAM_A
            elif team_id == 1:
                color = COLOR_TEAM_B
            elif team_id == 2:
                color = COLOR_REF
            else:
                color = COLOR_UNKNOWN

            x1, y1, x2, y2 = map(int, box)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Build label
            if speed_stats and stable_id in speed_stats:
                s     = speed_stats[stable_id]
                label = f"#{stable_id} {s.current_speed_kmh:.1f}km/h"
            else:
                label = f"#{stable_id}"

            # Draw label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame,
                (x1, y1 - th - 6), (x1 + tw + 4, y1),
                color, -1)
            cv2.putText(frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_TEXT, 1, cv2.LINE_AA)

        return frame

    # ── Ball ────────────────────────────────────────────────────────────────

    def _draw_ball(self, frame: np.ndarray, ball: sv.Detections) -> np.ndarray:
        for box in ball.xyxy:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            cv2.circle(frame, (cx, cy),  8, COLOR_BALL, -1)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
        return frame

    def _draw_ball_trail(self, frame: np.ndarray) -> np.ndarray:
        trail = list(self._ball_trail)
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            color = tuple(int(c * alpha) for c in COLOR_BALL)
            p1 = (int(trail[i-1][0]), int(trail[i-1][1]))
            p2 = (int(trail[i][0]),   int(trail[i][1]))
            cv2.line(frame, p1, p2, color,
                     max(1, int(alpha * 3)), cv2.LINE_AA)
        return frame

    # ── Event banner ─────────────────────────────────────────────────────────

    def _draw_event_banner(self, frame: np.ndarray, event) -> np.ndarray:
        text = f"  {event.event_type.name}"
        if event.player_id is not None:
            text += f" | Player #{event.player_id}"
        text += "  "

        font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 0.8, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 20), (tw + 20, 20 + th + 16),
                      COLOR_BANNER, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, text, (10, 20 + th + 6),
                    font, scale, COLOR_TEXT, thick, cv2.LINE_AA)
        return frame

    # ── Possession bar ───────────────────────────────────────────────────────

    def _draw_possession_bar(
        self,
        frame:      np.ndarray,
        possession: Dict,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        bar_w, bar_h = 300, 24
        x, y = w // 2 - bar_w // 2, 10

        a_pct = possession.get("Team A", 50) / 100
        a_w   = int(bar_w * a_pct)

        # Background
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h),
                      (50, 50, 50), -1)
        # Team A portion
        cv2.rectangle(frame, (x, y), (x + a_w, y + bar_h),
                      COLOR_TEAM_A, -1)
        # Team B portion
        cv2.rectangle(frame, (x + a_w, y), (x + bar_w, y + bar_h),
                      COLOR_TEAM_B, -1)
        # Border
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h),
                      COLOR_TEXT, 1)

        # Labels
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
        cv2.putText(frame,
            f"A {possession.get('Team A', 50):.0f}%",
            (x + 4, y + 17), font, scale, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame,
            f"{possession.get('Team B', 50):.0f}% B",
            (x + bar_w - 60, y + 17), font, scale, COLOR_TEXT, 1, cv2.LINE_AA)

        return frame

    # ── Mini radar ───────────────────────────────────────────────────────────

    def _build_radar(
        self,
        player_positions: Dict,
        ball_center:      Optional[Tuple],
        id_to_team:       Optional[Dict],
        pitch_w: float = 105.0,
        pitch_h: float =  68.0,
    ) -> np.ndarray:
        s  = self._radar_size
        rh = int(s * 68 / 105)
        rw = s

        radar = np.full((rh, rw, 3), COLOR_RADAR_BG, dtype=np.uint8)

        # Pitch lines
        cv2.rectangle(radar, (0, 0), (rw-1, rh-1), (255,255,255), 1)
        cv2.line(radar, (rw//2, 0), (rw//2, rh), (255,255,255), 1)

        def to_radar(wx, wy):
            px = int(np.clip(wx / pitch_w * rw, 0, rw-1))
            py = int(np.clip(wy / pitch_h * rh, 0, rh-1))
            return px, py

        for sid, (wx, wy) in player_positions.items():
            team = id_to_team.get(sid, -1) if id_to_team else -1
            if team == 0:
                col = COLOR_TEAM_A
            elif team == 1:
                col = COLOR_TEAM_B
            else:
                col = COLOR_UNKNOWN
            px, py = to_radar(wx, wy)
            cv2.circle(radar, (px, py), 4, col, -1)

        if ball_center:
            bx, by = ball_center
            px, py = to_radar(
                bx / 640 * pitch_w,
                by / 360 * pitch_h
            )
            cv2.circle(radar, (px, py), 5, COLOR_BALL, -1)

        return radar

    def _overlay_radar(
        self, frame: np.ndarray, radar: np.ndarray
    ) -> np.ndarray:
        h, w   = frame.shape[:2]
        rh, rw = radar.shape[:2]
        margin = 10
        x1 = w - rw - margin
        y1 = h - rh - margin

        overlay = frame.copy()
        cv2.rectangle(overlay,
            (x1-4, y1-4), (x1+rw+4, y1+rh+4), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        frame[y1:y1+rh, x1:x1+rw] = radar
        return frame