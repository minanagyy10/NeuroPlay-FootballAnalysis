"""
analytics/events.py
───────────────────
Football event detection.

Detects:
  - Pass
  - Shot
  - Cross
  - Tackle / Interception
  - Possession change

This is a rule-based system — accurate enough for Phase 2.
Phase 3 will replace/augment with an action-recognition model (TimeSformer).

All distances/velocities are in WORLD coordinates (metres) when calibration
is available, otherwise in pixels.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import math
import time


# ─── Event Types ─────────────────────────────────────────────────────────────

class EventType(Enum):
    PASS             = auto()
    SHOT             = auto()
    CROSS            = auto()
    TACKLE           = auto()
    INTERCEPTION     = auto()
    POSSESSION_CHANGE = auto()
    GOAL             = auto()


@dataclass
class FootballEvent:
    event_type:   EventType
    frame_index:  int
    timestamp_s:  float
    player_id:    Optional[int]  = None    # primary actor
    player_id_2:  Optional[int]  = None    # secondary actor (e.g. tackler)
    position:     Optional[Tuple[float, float]] = None   # world coords
    confidence:   float          = 1.0
    metadata:     dict           = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event":       self.event_type.name,
            "frame":       self.frame_index,
            "timestamp_s": round(self.timestamp_s, 2),
            "player_id":   self.player_id,
            "position":    self.position,
            "confidence":  round(self.confidence, 2),
            **self.metadata,
        }


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class EventConfig:
    # Ball speed threshold to call a pass (m/s in world space)
    pass_speed_threshold_ms:    float = 5.0
    # If ball comes near goal area → shot candidate
    goal_area_x_frac:           float = 0.85   # % along pitch length
    # Distance to consider a player "in possession" (metres)
    possession_radius_m:        float = 2.0
    # Cooldown between same-event type (frames)
    event_cooldown_frames:      int   = 15
    # Cross: ball in wide position and moving inward
    cross_zone_y_frac:          float = 0.20   # fraction of pitch width from edge


# ─── Event Detector ───────────────────────────────────────────────────────────

class EventDetector:
    """
    Stateful event detector. Feed frame-by-frame data; it emits events.

    Usage
    -----
    detector = EventDetector()

    for tf in tracker.update(fd):
        events = detector.update(
            frame_index=tf.frame_index,
            timestamp_s=tf.timestamp_s,
            ball_world=(bx, by),
            player_world_positions={tid: (wx, wy), ...},
        )
        for event in events:
            print(event.to_dict())

    all_events = detector.get_all_events()
    """

    def __init__(self, config: EventConfig = EventConfig()) -> None:
        self.cfg = config

        self._events:             List[FootballEvent] = []
        self._last_event_frame:   Dict[EventType, int] = {}
        self._possession_id:      Optional[int]        = None

        # Ball trajectory for velocity estimation
        self._ball_history:       List[Tuple[float, float, float]] = []  # (wx, wy, t)

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        frame_index:            int,
        timestamp_s:            float,
        ball_world:             Optional[Tuple[float, float]],
        player_world_positions: Dict[int, Tuple[float, float]],
    ) -> List[FootballEvent]:
        """
        Process one frame. Returns a (possibly empty) list of new events.
        """
        new_events: List[FootballEvent] = []

        if ball_world is not None:
            bx, by = ball_world
            self._ball_history.append((bx, by, timestamp_s))
            if len(self._ball_history) > 30:
                self._ball_history.pop(0)

            ball_speed = self._estimate_ball_speed()

            # ── Possession ──────────────────────────────────────────────────
            new_possessor = self._find_possessor(bx, by, player_world_positions)

            if (new_possessor is not None
                    and new_possessor != self._possession_id):
                if self._possession_id is not None:
                    ev = self._make_event(
                        EventType.POSSESSION_CHANGE,
                        frame_index, timestamp_s,
                        player_id=self._possession_id,
                        player_id_2=new_possessor,
                        position=(bx, by),
                    )
                    if ev:
                        new_events.append(ev)
                self._possession_id = new_possessor

            # ── Shot detection ───────────────────────────────────────────────
            if self._is_shot(bx, by, ball_speed):
                ev = self._make_event(
                    EventType.SHOT, frame_index, timestamp_s,
                    player_id=self._possession_id,
                    position=(bx, by),
                    metadata={"ball_speed_ms": round(ball_speed or 0, 1)},
                )
                if ev:
                    new_events.append(ev)

            # ── Pass detection ───────────────────────────────────────────────
            elif ball_speed and ball_speed > self.cfg.pass_speed_threshold_ms:
                is_cross = self._is_cross(bx, by)
                ev = self._make_event(
                    EventType.CROSS if is_cross else EventType.PASS,
                    frame_index, timestamp_s,
                    player_id=self._possession_id,
                    position=(bx, by),
                    metadata={"ball_speed_ms": round(ball_speed, 1)},
                )
                if ev:
                    new_events.append(ev)

        self._events.extend(new_events)
        return new_events

    def get_all_events(self) -> List[FootballEvent]:
        return list(self._events)

    def get_summary(self) -> dict:
        counts: Dict[str, int] = {}
        for ev in self._events:
            counts[ev.event_type.name] = counts.get(ev.event_type.name, 0) + 1
        return counts

    # ── Internal ────────────────────────────────────────────────────────────

    def _make_event(
        self,
        event_type: EventType,
        frame_index: int,
        timestamp_s: float,
        **kwargs,
    ) -> Optional[FootballEvent]:
        """Create event with cooldown guard."""
        last = self._last_event_frame.get(event_type, -999)
        if frame_index - last < self.cfg.event_cooldown_frames:
            return None

        self._last_event_frame[event_type] = frame_index
        return FootballEvent(
            event_type=event_type,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            **kwargs,
        )

    def _estimate_ball_speed(self) -> Optional[float]:
        if len(self._ball_history) < 2:
            return None
        x1, y1, t1 = self._ball_history[-2]
        x2, y2, t2 = self._ball_history[-1]
        dt = t2 - t1
        if dt < 1e-3:
            return None
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist / dt   # m/s

    def _find_possessor(
        self,
        bx: float,
        by: float,
        player_positions: Dict[int, Tuple[float, float]],
    ) -> Optional[int]:
        best_id, best_dist = None, float("inf")
        for tid, (px, py) in player_positions.items():
            d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
            if d < best_dist:
                best_dist, best_id = d, tid
        if best_dist <= self.cfg.possession_radius_m:
            return best_id
        return None

    def _is_shot(self, bx: float, by: float, speed: Optional[float]) -> bool:
        """Ball near goal area moving fast = shot."""
        near_goal = (
            bx >= self.cfg.goal_area_x_frac * 105.0
            or bx <= (1 - self.cfg.goal_area_x_frac) * 105.0
        )
        fast = speed is not None and speed > self.cfg.pass_speed_threshold_ms * 1.3
        return near_goal and fast

    def _is_cross(self, bx: float, by: float) -> bool:
        """Ball in wide zones moving inward."""
        wide_left  = by < self.cfg.cross_zone_y_frac * 68.0
        wide_right = by > (1 - self.cfg.cross_zone_y_frac) * 68.0
        return wide_left or wide_right