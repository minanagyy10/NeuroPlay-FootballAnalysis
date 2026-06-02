from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans


@dataclass
class TeamInfo:
    team_id:     int          # 0 or 1
    player_ids:  List[int]    = field(default_factory=list)
    color_bgr:   Tuple        = (255, 255, 255)
    label:       str          = "Team"
    possession:  float        = 0.0   # percentage


class TeamSeparator:
    """
    Automatically splits players into two teams using jersey color clustering.

    How it works
    ------------
    1. Collects jersey color samples from each player over many frames
    2. Runs K-Means clustering (k=3: team A, team B, referee)
    3. Assigns each stable player ID to a team
    4. Tracks possession per team

    Usage
    -----
    separator = TeamSeparator()

    # Feed jersey colors every frame
    separator.add_sample(stable_id, jersey_color_bgr)

    # After enough samples (100+ frames), cluster
    separator.cluster()

    # Get team for any player
    team = separator.get_team(stable_id)   # 0 or 1 or -1 (referee)
    color = separator.get_team_color(stable_id)
    """

    # Team colors for annotation
    TEAM_COLORS = [
        (255,  80,  80),   # Team A — blue
        ( 80, 255,  80),   # Team B — green
        (200, 200, 200),   # Referee — gray
    ]

    TEAM_LABELS = ["Team A", "Team B", "Referee"]

    def __init__(
        self,
        n_clusters:     int = 3,     # team A, team B, referee
        min_samples:    int = 50,    # minimum color samples before clustering
        recluster_every: int = 500,  # re-cluster every N frames
    ):
        self.n_clusters      = n_clusters
        self.min_samples     = min_samples
        self.recluster_every = recluster_every

        # {stable_id: list of (H, S, V) color samples}
        self._color_samples: Dict[int, List[np.ndarray]] = {}

        # {stable_id: team_id (0, 1, or 2)}
        self._player_teams:  Dict[int, int] = {}

        # possession tracking
        self._possession_frames: Dict[int, int] = {0: 0, 1: 0}

        self._last_cluster_frame = 0
        self._clustered = False

    # ── Public API ──────────────────────────────────────────────────────────

    def add_sample(
        self,
        stable_id: int,
        frame:     np.ndarray,
        box:       np.ndarray,    # [x1, y1, x2, y2]
    ) -> None:
        """Extract and store jersey color sample for a player."""
        color = self._extract_jersey_color(frame, box)
        if color is None:
            return

        if stable_id not in self._color_samples:
            self._color_samples[stable_id] = []

        samples = self._color_samples[stable_id]
        samples.append(color)

        # Keep max 200 samples per player
        if len(samples) > 200:
            self._color_samples[stable_id] = samples[-200:]

    def cluster(self, frame_index: int = 0) -> bool:
        """
        Run K-Means clustering to assign players to teams.
        Returns True if clustering was successful.
        """
        # Only recluster periodically
        if (self._clustered and
                frame_index - self._last_cluster_frame < self.recluster_every):
            return self._clustered

        # Need enough players with enough samples
        eligible = {
            pid: samples
            for pid, samples in self._color_samples.items()
            if len(samples) >= self.min_samples
        }

        if len(eligible) < 6:
            return False

        # Build feature matrix: mean HSV per player
        player_ids  = list(eligible.keys())
        features    = np.array([
            np.mean(eligible[pid], axis=0)
            for pid in player_ids
        ], dtype=np.float32)

        # K-Means clustering
        kmeans = KMeans(
            n_clusters  = min(self.n_clusters, len(player_ids)),
            n_init      = 10,
            random_state= 42,
        )
        labels = kmeans.fit_predict(features)

        # Assign teams
        for pid, label in zip(player_ids, labels):
            self._player_teams[pid] = int(label)

        # Identify referee cluster (smallest group)
        from collections import Counter
        counts  = Counter(labels)
        ref_cluster = min(counts, key=counts.get)

        # Relabel: largest two clusters = teams, smallest = referee
        sorted_clusters = sorted(counts.keys(), key=lambda k: -counts[k])
        remap = {}
        for new_id, old_id in enumerate(sorted_clusters[:3]):
            remap[old_id] = new_id

        for pid in self._player_teams:
            old_label = self._player_teams[pid]
            self._player_teams[pid] = remap.get(old_label, 2)

        self._clustered          = True
        self._last_cluster_frame = frame_index

        team_counts = Counter(self._player_teams.values())
        print(f"[TeamSeparator] Clustered {len(player_ids)} players → "
              f"Team A: {team_counts.get(0,0)} | "
              f"Team B: {team_counts.get(1,0)} | "
              f"Referee: {team_counts.get(2,0)}")

        return True

    def get_team(self, stable_id: int) -> int:
        """Returns 0 (Team A), 1 (Team B), 2 (Referee), -1 (unknown)."""
        return self._player_teams.get(stable_id, -1)

    def get_team_color(self, stable_id: int) -> Tuple:
        """Returns BGR color tuple for this player's team."""
        team = self.get_team(stable_id)
        if 0 <= team < len(self.TEAM_COLORS):
            return self.TEAM_COLORS[team]
        return (200, 200, 200)  # gray for unknown

    def get_team_label(self, stable_id: int) -> str:
        team = self.get_team(stable_id)
        if 0 <= team < len(self.TEAM_LABELS):
            return self.TEAM_LABELS[team]
        return "Unknown"

    def update_possession(self, possessor_id: int) -> None:
        """Call when a player has the ball."""
        team = self.get_team(possessor_id)
        if team in (0, 1):
            self._possession_frames[team] = \
                self._possession_frames.get(team, 0) + 1

    def get_possession(self) -> Dict[str, float]:
        """Returns possession percentage for each team."""
        total = sum(self._possession_frames.values())
        if total == 0:
            return {"Team A": 50.0, "Team B": 50.0}
        return {
            "Team A": round(self._possession_frames[0] / total * 100, 1),
            "Team B": round(self._possession_frames[1] / total * 100, 1),
        }

    def get_team_players(self, team_id: int) -> List[int]:
        """Returns list of player IDs for a team."""
        return [
            pid for pid, tid in self._player_teams.items()
            if tid == team_id
        ]

    def is_clustered(self) -> bool:
        return self._clustered

    # ── Color Extraction ────────────────────────────────────────────────────

    def _extract_jersey_color(
        self,
        frame: np.ndarray,
        box:   np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extract dominant jersey color from upper-middle of bounding box."""
        x1, y1, x2, y2 = map(int, box)
        h = y2 - y1
        w = x2 - x1

        if h < 10 or w < 5:
            return None

        # Take middle 60% width, upper 40% height (jersey area)
        mx1 = x1 + w // 5
        mx2 = x2 - w // 5
        my1 = y1 + h // 10
        my2 = y1 + int(h * 0.45)

        # Clamp to frame
        mx1 = max(0, min(mx1, frame.shape[1]))
        mx2 = max(0, min(mx2, frame.shape[1]))
        my1 = max(0, min(my1, frame.shape[0]))
        my2 = max(0, min(my2, frame.shape[0]))

        crop = frame[my1:my2, mx1:mx2]
        if crop.size == 0:
            return None

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean = hsv.mean(axis=(0, 1))
        return mean