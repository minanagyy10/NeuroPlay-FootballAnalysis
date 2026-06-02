from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv

from detection.detector import FrameDetections


@dataclass
class TrackState:
    track_id:    int
    positions:   List[Tuple[float, float]] = field(default_factory=list)
    timestamps:  List[float]               = field(default_factory=list)
    frame_idxs:  List[int]                = field(default_factory=list)
    is_ball:     bool                      = False

    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None

    def update(self, cx: float, cy: float,
               timestamp: float, frame_idx: int) -> None:
        self.positions.append((cx, cy))
        self.timestamps.append(timestamp)
        self.frame_idxs.append(frame_idx)


@dataclass
class TrackedFrame:
    players:        sv.Detections
    ball:           sv.Detections
    frame_index:    int
    timestamp_s:    float
    player_centers: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    ball_center:    Optional[Tuple[float, float]]  = None


class FootballTracker:

    def __init__(
        self,
        frame_rate:   int   = 25,
        match_thresh: float = 0.8,
        track_buffer: int   = 90,
    ):
        self._player_tracker = sv.ByteTrack(
            minimum_matching_threshold=match_thresh,
            lost_track_buffer=track_buffer,
            minimum_consecutive_frames=2,
            frame_rate=frame_rate,
        )
        self._ball_tracker = sv.ByteTrack(
            minimum_matching_threshold=match_thresh,
            lost_track_buffer=track_buffer,
            minimum_consecutive_frames=1,
            frame_rate=frame_rate,
        )

        self.player_states: Dict[int, TrackState] = {}
        self.ball_states:   Dict[int, TrackState] = {}

    def update(self, fd: FrameDetections) -> TrackedFrame:
        players = self._player_tracker.update_with_detections(fd.players)
        ball    = self._ball_tracker.update_with_detections(fd.ball)

        player_centers = self._record(
            players, self.player_states,
            fd.timestamp_s, fd.frame_index, is_ball=False
        )
        ball_centers = self._record(
            ball, self.ball_states,
            fd.timestamp_s, fd.frame_index, is_ball=True
        )

        ball_center: Optional[tuple] = (
            next(iter(ball_centers.values())) if ball_centers else None
        )

        return TrackedFrame(
            players=players,
            ball=ball,
            frame_index=fd.frame_index,
            timestamp_s=fd.timestamp_s,
            player_centers=player_centers,
            ball_center=ball_center,
        )

    def get_player_history(self, track_id: int) -> Optional[TrackState]:
        return self.player_states.get(track_id)

    def all_player_positions(self) -> Dict[int, List[Tuple[float, float]]]:
        return {tid: s.positions for tid, s in self.player_states.items()}

    def ball_trajectory(self) -> List[Tuple[float, float]]:
        positions = []
        for s in self.ball_states.values():
            positions.extend(s.positions)
        return positions

    @staticmethod
    def _record(
        detections:  sv.Detections,
        states:      Dict[int, TrackState],
        timestamp_s: float,
        frame_idx:   int,
        is_ball:     bool,
    ) -> Dict[int, Tuple[float, float]]:
        centers: Dict[int, Tuple[float, float]] = {}

        if detections.tracker_id is None:
            return centers

        for box, tid in zip(detections.xyxy, detections.tracker_id):
            cx = float((box[0] + box[2]) / 2)
            cy = float((box[1] + box[3]) / 2)

            if tid not in states:
                states[tid] = TrackState(track_id=int(tid), is_ball=is_ball)

            states[tid].update(cx, cy, timestamp_s, frame_idx)
            centers[int(tid)] = (cx, cy)

        return centers