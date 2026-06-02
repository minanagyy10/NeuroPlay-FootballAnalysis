"""
analytics/speed.py
──────────────────
Speed & distance analytics per player.

Converts pixel trajectories → real-world metres → speed in km/h.

Key idea
--------
Given two consecutive world positions (x1, y1) and (x2, y2) at times t1, t2:

    distance = sqrt((x2-x1)² + (y2-y1)²)   metres
    speed    = distance / (t2 - t1)          m/s
    speed_kh = speed * 3.6                   km/h

We apply a rolling median to suppress noise (occlusion, detection jitter).
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from calibration.homography import HomographyCalibrator
from tracking.tracker import TrackState


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class SpeedConfig:
    smooth_window:  int   = 5      # rolling median window for speed noise removal
    max_speed_kmh:  float = 32.0   # discard spikes above this (tracking errors)
    min_dt:         float = 0.02   # minimum seconds between samples (avoids /0)


# ─── Per-player accumulator ───────────────────────────────────────────────────

@dataclass
class PlayerSpeedStats:
    track_id:         int
    current_speed_kmh: float = 0.0
    max_speed_kmh:     float = 0.0
    avg_speed_kmh:     float = 0.0
    total_distance_m:  float = 0.0
    speed_history:     List[float] = field(default_factory=list)

    # Sprint zones (FIFA standard)
    @property
    def sprint_time_s(self) -> float:
        """Seconds spent at > 25 km/h."""
        # Approximate: each sample ~ 1/fps seconds
        return sum(1 for s in self.speed_history if s >= 25.0) / 25.0  # assume 25 fps

    @property
    def high_intensity_time_s(self) -> float:
        """Seconds at > 19.8 km/h."""
        return sum(1 for s in self.speed_history if s >= 19.8) / 25.0


# ─── SpeedAnalyzer ────────────────────────────────────────────────────────────

class SpeedAnalyzer:
    """
    Maintains per-player speed stats, updated incrementally frame by frame.

    Usage
    -----
    calib   = HomographyCalibrator(...)
    analyzer = SpeedAnalyzer(calib)

    for tf in tracker.iter(...):
        analyzer.update_from_tracked_frame(tf, calib)

    stats = analyzer.get_stats(player_id=7)
    print(stats.max_speed_kmh)
    """

    def __init__(
        self,
        calibrator: HomographyCalibrator,
        config: SpeedConfig = SpeedConfig(),
    ) -> None:
        self.calib  = calibrator
        self.cfg    = config

        # {track_id: deque of (world_x, world_y, timestamp_s)}
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.cfg.smooth_window + 2)
        )
        self._stats: Dict[int, PlayerSpeedStats] = {}

        # Rolling speed buffers for smoothing
        self._speed_buf: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.cfg.smooth_window)
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        track_id:   int,
        pixel_cx:   float,
        pixel_cy:   float,
        timestamp_s: float,
    ) -> Optional[float]:
        """
        Feed one new detection for a player.
        Returns smoothed speed in km/h (None if not enough history).
        """
        if self.calib.is_calibrated:
            wx, wy = self.calib.to_world(pixel_cx, pixel_cy)
        else:
            # Fallback: pixel distance (won't be in km/h, but relative)
            wx, wy = pixel_cx, pixel_cy

        hist = self._history[track_id]
        hist.append((wx, wy, timestamp_s))

        if len(hist) < 2:
            return None

        raw_speed = self._compute_raw_speed(list(hist)[-2:])
        if raw_speed is None:
            return None

        # Smooth with rolling median
        buf = self._speed_buf[track_id]
        buf.append(raw_speed)
        smoothed = float(np.median(list(buf)))

        # Update persistent stats
        self._update_stats(track_id, smoothed)
        return smoothed

    def get_stats(self, track_id: int) -> Optional[PlayerSpeedStats]:
        return self._stats.get(track_id)

    def all_stats(self) -> Dict[int, PlayerSpeedStats]:
        return dict(self._stats)

    def summary_table(self) -> List[dict]:
        rows = []
        for tid, s in sorted(self._stats.items()):
            rows.append({
                "player_id":        tid,
                "current_speed_kmh": round(s.current_speed_kmh, 1),
                "max_speed_kmh":    round(s.max_speed_kmh, 1),
                "avg_speed_kmh":    round(s.avg_speed_kmh, 1),
                "distance_m":       round(s.total_distance_m, 1),
                "sprint_time_s":    round(s.sprint_time_s, 1),
            })
        return rows

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _compute_raw_speed(
        self, two_points: list
    ) -> Optional[float]:
        (x1, y1, t1), (x2, y2, t2) = two_points
        dt = t2 - t1
        if dt < self.cfg.min_dt:
            return None

        dist_m  = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        speed_ms = dist_m / dt
        speed_kmh = speed_ms * 3.6

        if speed_kmh > self.cfg.max_speed_kmh:
            return None   # tracking error spike

        return speed_kmh

    def _update_stats(self, track_id: int, speed_kmh: float) -> None:
        if track_id not in self._stats:
            self._stats[track_id] = PlayerSpeedStats(track_id=track_id)

        s = self._stats[track_id]
        s.current_speed_kmh = speed_kmh
        s.speed_history.append(speed_kmh)
        s.max_speed_kmh = max(s.max_speed_kmh, speed_kmh)
        s.avg_speed_kmh = float(np.mean(s.speed_history))

        # Accumulate distance from the last two world positions
        hist = list(self._history[track_id])
        if len(hist) >= 2:
            x1, y1, _ = hist[-2]
            x2, y2, _ = hist[-1]
            s.total_distance_m += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)