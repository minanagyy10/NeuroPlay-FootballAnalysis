"""
calibration/homography.py
─────────────────────────
Camera calibration via homography.

Maps pixel coordinates → real-world pitch coordinates (meters).

Standard pitch dimensions (FIFA):
  Width:  105 m  (touchline)
  Height:  68 m  (goal line)

Usage
-----
# 1. Define 4+ corresponding point pairs:
#    pixel_pts  → where field landmarks appear in your video frame
#    world_pts  → their known positions in meters on a standard pitch

pixel_pts = np.array([
    [120, 680], [1800, 680],   # bottom left/right corners
    [120,  40], [1800,  40],   # top    left/right corners
], dtype=np.float32)

world_pts = np.array([
    [0,  68], [105,  68],      # same corners in real-world meters
    [0,   0], [105,   0],
], dtype=np.float32)

calib = HomographyCalibrator()
calib.calibrate(pixel_pts, world_pts)

real_xy = calib.to_world(px=540, py=360)
# → (52.5, 34.0)  metres from bottom-left corner
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2


# ─── Standard FIFA pitch in metres ───────────────────────────────────────────
PITCH_WIDTH_M  = 105.0
PITCH_HEIGHT_M =  68.0


class HomographyCalibrator:
    """
    Estimates and applies a homography between pixel space and pitch space.

    Methods
    -------
    calibrate(pixel_pts, world_pts)  → compute H
    to_world(px, py)                 → (mx, my)  pitch metres
    to_pixel(mx, my)                 → (px, py)  image pixels
    project_detections(boxes)        → array of (mx, my) for each box centre
    """

    def __init__(self) -> None:
        self._H:     Optional[np.ndarray] = None   # 3×3 pixel→world
        self._H_inv: Optional[np.ndarray] = None   # 3×3 world→pixel
        self.reprojection_error: float    = float("inf")

    # ── Calibration ─────────────────────────────────────────────────────────

    def calibrate(
        self,
        pixel_pts: np.ndarray,   # shape (N, 2)  float32
        world_pts: np.ndarray,   # shape (N, 2)  float32
        method: int = cv2.RANSAC,
    ) -> None:
        """
        Compute homography from N ≥ 4 point correspondences.

        Parameters
        ----------
        pixel_pts : pixel (x, y) of known field landmarks
        world_pts : real-world (x, y) in metres of those same landmarks
        method    : cv2.RANSAC (robust) or 0 (all points, exact)
        """
        assert len(pixel_pts) >= 4, "Need at least 4 point pairs."

        H, mask = cv2.findHomography(
            pixel_pts.astype(np.float32),
            world_pts.astype(np.float32),
            method,
            ransacReprojThreshold=3.0,
        )

        if H is None:
            raise RuntimeError("Homography estimation failed.")

        self._H     = H
        self._H_inv = np.linalg.inv(H)

        # Compute reprojection error on inliers
        inlier_mask = mask.ravel().astype(bool)
        px_in = pixel_pts[inlier_mask]
        wd_in = world_pts[inlier_mask]

        projected = self._apply_H(self._H, px_in)
        self.reprojection_error = float(
            np.mean(np.linalg.norm(projected - wd_in, axis=1))
        )
        n_inliers = int(inlier_mask.sum())
        print(
            f"[Calibrator] H estimated | inliers={n_inliers}/{len(pixel_pts)} | "
            f"reprojection_err={self.reprojection_error:.3f} m"
        )

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    # ── Coordinate Transforms ────────────────────────────────────────────────

    def to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Convert a single pixel point to pitch metres."""
        self._check()
        pt = self._apply_H(self._H, np.array([[px, py]], dtype=np.float32))
        return float(pt[0, 0]), float(pt[0, 1])

    def to_pixel(self, mx: float, my: float) -> Tuple[float, float]:
        """Convert a single pitch-metre point to pixels."""
        self._check()
        pt = self._apply_H(self._H_inv, np.array([[mx, my]], dtype=np.float32))
        return float(pt[0, 0]), float(pt[0, 1])

    def project_detections(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """
        Given an (N, 4) array of [x1, y1, x2, y2] boxes,
        return (N, 2) array of pitch-metre (mx, my) for each box centre.
        """
        self._check()
        if len(boxes_xyxy) == 0:
            return np.empty((0, 2), dtype=np.float32)

        centres = np.stack([
            (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2,
            (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2,
        ], axis=1).astype(np.float32)

        return self._apply_H(self._H, centres)

    # ── Static Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply 3×3 homography to (N, 2) point array."""
        pts_h   = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        result  = (H @ pts_h.T).T
        result /= result[:, 2:3]          # normalise homogeneous coordinates
        return result[:, :2]

    def _check(self) -> None:
        if not self.is_calibrated:
            raise RuntimeError(
                "Calibrator not calibrated. Call .calibrate(pixel_pts, world_pts) first."
            )

    # ── Convenience: auto-detect field corners ───────────────────────────────

    @classmethod
    def from_full_frame_corners(
        cls,
        frame_w: int,
        frame_h: int,
        padding_frac: float = 0.05,
    ) -> "HomographyCalibrator":
        """
        Quick approximation when the full pitch fills the frame.
        Works for static wide-angle cameras (not TV broadcast panning).

        padding_frac: fraction of frame to ignore on each edge
        """
        px = padding_frac
        pixel_pts = np.array([
            [frame_w * px,       frame_h * (1 - px)],
            [frame_w * (1 - px), frame_h * (1 - px)],
            [frame_w * px,       frame_h * px],
            [frame_w * (1 - px), frame_h * px],
        ], dtype=np.float32)

        world_pts = np.array([
            [0,              PITCH_HEIGHT_M],
            [PITCH_WIDTH_M,  PITCH_HEIGHT_M],
            [0,              0],
            [PITCH_WIDTH_M,  0],
        ], dtype=np.float32)

        calib = cls()
        calib.calibrate(pixel_pts, world_pts, method=0)
        return calib