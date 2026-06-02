from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PlayerSignature:
    """
    Visual signature of a player extracted from their bounding box.
    Used to re-identify players after they disappear and reappear.
    """
    track_id:       int
    color_hist:     np.ndarray      # HSV color histogram of jersey
    avg_position:   Tuple[float, float]  # average position on pitch
    last_seen_frame: int
    appearances:    int = 1


class ReIDSystem:
    """
    Player Re-Identification system.

    When a new player ID appears, checks if it matches any previously
    seen player by comparing jersey color and last known position.
    If a match is found, the old ID is reused instead of creating a new one.

    This reduces tracking ID explosion from thousands to ~22-30 stable IDs.

    Usage
    -----
    reid = ReIDSystem()

    # In your pipeline loop:
    id_mapping = reid.update(
        frame        = bgr_frame,
        detections   = tracked_detections,
        frame_index  = frame_index,
    )

    # id_mapping = {new_id: stable_id}
    # Use stable_id everywhere instead of new_id
    """

    def __init__(
        self,
        color_threshold:    float = 0.35,   # max histogram distance to match
        position_threshold: float = 150.0,  # max pixel distance to match
        max_lost_frames:    int   = 150,    # forget player after this many frames
        min_appearances:    int   = 3,      # minimum frames to build signature
    ):
        self.color_thresh    = color_threshold
        self.pos_thresh      = position_threshold
        self.max_lost_frames = max_lost_frames
        self.min_appearances = min_appearances

        # known players: {stable_id: PlayerSignature}
        self._signatures:  Dict[int, PlayerSignature] = {}

        # mapping: {current_tracker_id: stable_id}
        self._id_map:      Dict[int, int] = {}

        # active IDs in current frame
        self._active_ids:  set = set()

        self._next_stable_id = 1

    # ── Public API ──────────────────────────────────────────────────────────

    def update(
        self,
        frame:       np.ndarray,
        boxes_xyxy:  np.ndarray,     # (N, 4) bounding boxes
        tracker_ids: np.ndarray,     # (N,)   tracker IDs from ByteTrack
        frame_index: int,
    ) -> Dict[int, int]:
        """
        Process one frame. Returns {tracker_id: stable_id} mapping.

        Parameters
        ----------
        frame       : BGR video frame
        boxes_xyxy  : bounding boxes from tracker
        tracker_ids : raw tracker IDs from ByteTrack
        frame_index : current frame number

        Returns
        -------
        dict mapping raw tracker_id → stable re-identified ID
        """
        if len(boxes_xyxy) == 0:
            return {}

        self._active_ids = set()
        id_mapping: Dict[int, int] = {}

        for box, tid in zip(boxes_xyxy, tracker_ids):
            tid = int(tid)
            cx  = float((box[0] + box[2]) / 2)
            cy  = float((box[1] + box[3]) / 2)

            # Extract color signature from this box
            color_hist = self._extract_color_hist(frame, box)

            if tid in self._id_map:
                # Known tracker ID — update its signature
                stable_id = self._id_map[tid]
                self._update_signature(stable_id, color_hist, (cx, cy), frame_index)

            else:
                # New tracker ID — try to match with lost players
                stable_id = self._match_lost_player(
                    color_hist, (cx, cy), frame_index
                )

                if stable_id is None:
                    # No match — assign new stable ID
                    stable_id = self._next_stable_id
                    self._next_stable_id += 1
                    self._signatures[stable_id] = PlayerSignature(
                        track_id      = stable_id,
                        color_hist    = color_hist,
                        avg_position  = (cx, cy),
                        last_seen_frame = frame_index,
                    )

                self._id_map[tid] = stable_id

            self._active_ids.add(stable_id)
            id_mapping[tid] = stable_id

        # Mark inactive players
        self._cleanup_lost_players(frame_index)

        return id_mapping

    def get_stable_id(self, tracker_id: int) -> Optional[int]:
        return self._id_map.get(tracker_id)

    def total_unique_players(self) -> int:
        return len(self._signatures)

    # ── Color Extraction ────────────────────────────────────────────────────

    def _extract_color_hist(
        self, frame: np.ndarray, box: np.ndarray
    ) -> np.ndarray:
        """
        Extract HSV color histogram from the UPPER HALF of bounding box.
        Upper half = jersey area (avoids shorts/legs which vary less).
        """
        x1, y1, x2, y2 = map(int, box)
        h = y2 - y1

        # Clamp to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        mid_y = max(0, min(frame.shape[0], y1 + h // 2))

        crop = frame[y1:mid_y, x1:x2]

        if crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # H: 32 bins, S: 32 bins, V: 64 bins → 128 total
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256])

        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()

        # Normalise
        norm = hist.sum()
        if norm > 0:
            hist /= norm

        return hist.astype(np.float32)

    # ── Matching ─────────────────────────────────────────────────────────────

    def _match_lost_player(
        self,
        color_hist:  np.ndarray,
        position:    Tuple[float, float],
        frame_index: int,
    ) -> Optional[int]:
        """
        Try to find a lost player that matches this color + position.
        Returns stable_id if match found, None otherwise.
        """
        best_id    = None
        best_score = float("inf")

        for stable_id, sig in self._signatures.items():
            if stable_id in self._active_ids:
                continue  # player is currently tracked, skip

            frames_lost = frame_index - sig.last_seen_frame
            if frames_lost > self.max_lost_frames:
                continue  # too long ago, forget

            # Color distance (histogram correlation — lower = more similar)
            color_dist = cv2.compareHist(
                color_hist, sig.color_hist,
                cv2.HISTCMP_BHATTACHARYYA
            )

            # Position distance
            px, py = position
            sx, sy = sig.avg_position
            pos_dist = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)

            # Combined score
            if color_dist < self.color_thresh and pos_dist < self.pos_thresh:
                score = color_dist * 0.7 + (pos_dist / self.pos_thresh) * 0.3
                if score < best_score:
                    best_score = best_id if best_id else score
                    best_id    = stable_id

        return best_id

    # ── Signature Updates ────────────────────────────────────────────────────

    def _update_signature(
        self,
        stable_id:   int,
        color_hist:  np.ndarray,
        position:    Tuple[float, float],
        frame_index: int,
    ) -> None:
        sig = self._signatures.get(stable_id)
        if sig is None:
            return

        # Exponential moving average for color
        alpha = 0.1
        sig.color_hist = (1 - alpha) * sig.color_hist + alpha * color_hist

        # Update average position
        px, py = position
        sx, sy = sig.avg_position
        sig.avg_position    = (sx * 0.9 + px * 0.1, sy * 0.9 + py * 0.1)
        sig.last_seen_frame = frame_index
        sig.appearances    += 1

    def _cleanup_lost_players(self, frame_index: int) -> None:
        """Remove tracker ID mappings for very old lost players."""
        to_remove = [
            tid for tid, stable_id in self._id_map.items()
            if stable_id not in self._active_ids
            and stable_id in self._signatures
            and frame_index - self._signatures[stable_id].last_seen_frame
                > self.max_lost_frames * 2
        ]
        for tid in to_remove:
            del self._id_map[tid]