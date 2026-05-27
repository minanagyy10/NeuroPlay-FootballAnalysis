"""
analytics/heatmap.py
────────────────────
Heatmap & positional analytics.

Generates:
  - Per-player position heatmaps
  - Team average position map
  - Detected formation (4-3-3, 4-2-3-1, etc.)
  - Ball trajectory heatmap
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2


# ─── Standard pitch dimensions (metres) ──────────────────────────────────────
PITCH_W = 105.0
PITCH_H =  68.0


class HeatmapGenerator:
    """
    Creates heatmaps from accumulated player/ball pixel or world positions.

    Usage
    -----
    gen = HeatmapGenerator(output_size=(1050, 680))  # pixels per metre × 10

    # Feed positions over the match
    for frame_positions in ...:
        for track_id, (wx, wy) in frame_positions.items():
            gen.add_position(track_id, wx, wy)

    # Render
    heatmap_img = gen.render_player_heatmap(track_id=7)
    team_img    = gen.render_team_heatmap([1,2,3,4,5,6,7,8,9,10,11])
    gen.save_all("outputs/heatmaps/")
    """

    def __init__(
        self,
        output_size:  Tuple[int, int] = (1050, 680),   # (width, height) pixels
        sigma:        float = 20.0,                     # Gaussian blur radius
        colormap:     int   = cv2.COLORMAP_JET,
    ) -> None:
        self.W, self.H   = output_size
        self.sigma       = sigma
        self.colormap    = colormap

        # {track_id: list of (wx, wy) in world metres}
        self._positions: Dict[int, List[Tuple[float, float]]] = {}

    # ── Data ingestion ───────────────────────────────────────────────────────

    def add_position(self, track_id: int, wx: float, wy: float) -> None:
        """Add one world-space (metres) position for a player."""
        self._positions.setdefault(track_id, []).append((wx, wy))

    def add_pixel_position(
        self,
        track_id: int,
        px: float,
        py: float,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Add a pixel-space position (auto-converts to world metres)."""
        wx = (px / frame_w) * PITCH_W
        wy = (py / frame_h) * PITCH_H
        self.add_position(track_id, wx, wy)

    def load_from_tracker(
        self,
        all_positions: Dict[int, List[Tuple[float, float]]],
    ) -> None:
        """Bulk-load from tracker.all_player_positions()."""
        for tid, pos_list in all_positions.items():
            for wx, wy in pos_list:
                self.add_position(tid, wx, wy)

    # ── Rendering ────────────────────────────────────────────────────────────

    def render_player_heatmap(self, track_id: int) -> Optional[np.ndarray]:
        """Return a coloured heatmap image for one player."""
        positions = self._positions.get(track_id)
        if not positions:
            return None
        return self._build_heatmap(positions)

    def render_team_heatmap(
        self, track_ids: List[int]
    ) -> np.ndarray:
        """Aggregate heatmap for a team (list of track IDs)."""
        all_pos = []
        for tid in track_ids:
            all_pos.extend(self._positions.get(tid, []))
        if not all_pos:
            return self._blank_pitch()
        return self._build_heatmap(all_pos)

    def render_average_positions(
        self, track_ids: List[int]
    ) -> np.ndarray:
        """
        Draw dots at each player's average position on a pitch outline.
        Useful for formation detection visualisation.
        """
        img = self._blank_pitch(alpha=True)
        for tid in track_ids:
            positions = self._positions.get(tid)
            if not positions:
                continue
            positions = np.array(positions)
            avg_x = float(positions[:, 0].mean())
            avg_y = float(positions[:, 1].mean())
            px, py = self._world_to_pixel(avg_x, avg_y)

            cv2.circle(img, (px, py), 12, (255, 255, 255, 255), -1)
            cv2.circle(img, (px, py),  9, (30, 144, 255, 255), -1)
            cv2.putText(
                img, str(tid), (px - 6, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1,
                cv2.LINE_AA,
            )
        return img

    # ── Formation Detection ──────────────────────────────────────────────────

    def detect_formation(self, track_ids: List[int]) -> str:
        """
        Heuristic formation detection from average x-positions.

        Clusters players into horizontal defensive lines and counts them.
        Returns string like '4-3-3' or '4-2-3-1'.
        """
        if len(track_ids) < 10:
            return "Unknown"

        avg_xs = []
        for tid in track_ids:
            positions = self._positions.get(tid)
            if positions:
                xs = [p[0] for p in positions]
                avg_xs.append((tid, float(np.mean(xs))))

        if not avg_xs:
            return "Unknown"

        # Sort by x (depth in field)
        avg_xs.sort(key=lambda t: t[1])
        n = len(avg_xs)

        # Exclude goalkeeper (deepest player)
        outfield = avg_xs[1:]   # skip the most defensive player

        if n >= 10:
            formations = _cluster_to_formation(outfield)
            return formations

        return "Unknown"

    # ── Save Utilities ───────────────────────────────────────────────────────

    def save_all(self, output_dir: str, track_ids: Optional[List[int]] = None) -> None:
        """Save heatmap images for all (or specified) players."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ids = track_ids or list(self._positions.keys())

        for tid in ids:
            img = self.render_player_heatmap(tid)
            if img is not None:
                path = f"{output_dir}/player_{tid}_heatmap.jpg"
                cv2.imwrite(path, img)
                print(f"[Heatmap] Saved {path}")

        team_img = self.render_team_heatmap(ids)
        cv2.imwrite(f"{output_dir}/team_heatmap.jpg", team_img)
        print(f"[Heatmap] Saved team heatmap")

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _build_heatmap(
        self, positions: List[Tuple[float, float]]
    ) -> np.ndarray:
        canvas = np.zeros((self.H, self.W), dtype=np.float32)

        for wx, wy in positions:
            px, py = self._world_to_pixel(wx, wy)
            if 0 <= px < self.W and 0 <= py < self.H:
                canvas[py, px] += 1.0

        # Gaussian blur
        k = int(self.sigma * 3) | 1   # odd kernel size
        blurred = cv2.GaussianBlur(canvas, (k, k), self.sigma)

        # Normalise and apply colourmap
        if blurred.max() > 0:
            norm = (blurred / blurred.max() * 255).astype(np.uint8)
        else:
            norm = blurred.astype(np.uint8)

        coloured = cv2.applyColorMap(norm, self.colormap)

        # Blend with pitch outline
        pitch = self._draw_pitch_lines()
        blended = cv2.addWeighted(coloured, 0.75, pitch, 0.25, 0)
        return blended

    def _world_to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        px = int(np.clip((wx / PITCH_W) * self.W, 0, self.W - 1))
        py = int(np.clip((wy / PITCH_H) * self.H, 0, self.H - 1))
        return px, py

    def _draw_pitch_lines(self) -> np.ndarray:
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        img[:] = (34, 85, 34)   # dark green pitch

        c = (255, 255, 255)
        t = 2

        # Outer boundary
        cv2.rectangle(img, (0, 0), (self.W - 1, self.H - 1), c, t)

        # Halfway line
        cv2.line(img, (self.W // 2, 0), (self.W // 2, self.H), c, t)

        # Centre circle (radius ~9.15 m)
        r = int((9.15 / PITCH_W) * self.W)
        cv2.circle(img, (self.W // 2, self.H // 2), r, c, t)

        # Penalty areas (simplified)
        pa_w = int((16.5 / PITCH_W) * self.W)
        pa_h = int((40.32 / PITCH_H) * self.H)
        pa_top = (self.H - pa_h) // 2

        cv2.rectangle(img, (0, pa_top), (pa_w, pa_top + pa_h), c, t)
        cv2.rectangle(img, (self.W - pa_w, pa_top),
                      (self.W - 1, pa_top + pa_h), c, t)

        return img

    def _blank_pitch(self, alpha: bool = False) -> np.ndarray:
        if alpha:
            img = np.zeros((self.H, self.W, 4), dtype=np.uint8)
            img[:, :, :3] = np.array([34, 85, 34], dtype=np.uint8)
            img[:, :,  3] = 200
        else:
            img = self._draw_pitch_lines()
        return img


# ─── Formation helpers ────────────────────────────────────────────────────────

def _cluster_to_formation(outfield: List[Tuple[int, float]]) -> str:
    """Cluster outfield players by x-position into defensive lines."""
    positions = np.array([x for _, x in outfield]).reshape(-1, 1)

    # Simple percentile split into 3–4 lines
    p25, p50, p75 = np.percentile(positions, [25, 50, 75])

    line1 = sum(1 for _, x in outfield if x <= p25)
    line2 = sum(1 for _, x in outfield if p25 < x <= p50)
    line3 = sum(1 for _, x in outfield if p50 < x <= p75)
    line4 = sum(1 for _, x in outfield if x > p75)

    parts = [str(l) for l in [line1, line2, line3, line4] if l > 0]
    return "-".join(parts) if len(parts) >= 3 else "Unknown"