"""
tactical_ai/coach.py
────────────────────
Tactical AI: xG, xT, and AI coach recommendations.

Phase 3 of the system.

Includes
--------
xGModel      — probability a shot results in a goal given position + context
xTModel      — expected threat value of any pitch position with the ball
TacticalCoach — natural-language coaching suggestions via Claude API

The xG and xT models here are grid/formula-based approximations.
In production: train on StatsBomb open data for a proper ML model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import json
import httpx   # lightweight HTTP client (pip install httpx)


# ─── Standard pitch ───────────────────────────────────────────────────────────
PITCH_W = 105.0
PITCH_H =  68.0
GOAL_W  =   7.32
GOAL_Y  =  (PITCH_H - GOAL_W) / 2   # left goal post y
GOAL_X  =   0.0                      # attacking goal at x=105


# ─── xG Model ─────────────────────────────────────────────────────────────────

class xGModel:
    """
    Simple formula-based Expected Goals model.

    Real-world: train on StatsBomb data with logistic regression or XGBoost.

    Formula
    -------
    xG ≈ logistic(β0 + β1·dist + β2·angle + β3·header)

    Coefficients below are approximate and should be calibrated on data.
    """

    # Logistic regression coefficients (approximate — retrain on real data)
    _B0     = 0.076
    _B_DIST = -0.095   # further = lower xG
    _B_ANG  =  0.05    # wider angle = higher xG
    _B_HEAD = -0.3     # headers less likely to score

    def predict(
        self,
        shot_x: float,       # metres from own goal (0=own goal, 105=opp goal)
        shot_y: float,       # metres from bottom touchline
        is_header: bool = False,
    ) -> float:
        """Return xG ∈ [0, 1]."""
        dist_m = math.sqrt(
            (PITCH_W - shot_x) ** 2 + (shot_y - PITCH_H / 2) ** 2
        )
        angle_rad = self._shot_angle(shot_x, shot_y)

        z = (
            self._B0
            + self._B_DIST * dist_m
            + self._B_ANG  * math.degrees(angle_rad)
            + self._B_HEAD * int(is_header)
        )
        return round(self._sigmoid(z), 3)

    def _shot_angle(self, x: float, y: float) -> float:
        """
        Angle subtended by the goal as seen from (x, y).
        Larger = better shooting position.
        """
        gx = PITCH_W     # goal on x=105 side
        g1 = GOAL_Y
        g2 = GOAL_Y + GOAL_W

        # Law of cosines
        a  = math.sqrt((gx - x) ** 2 + (g1 - y) ** 2)
        b  = math.sqrt((gx - x) ** 2 + (g2 - y) ** 2)
        c  = GOAL_W

        cos_theta = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b + 1e-8)
        return math.acos(max(-1.0, min(1.0, cos_theta)))

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))


# ─── xT Model ─────────────────────────────────────────────────────────────────

class xTModel:
    """
    Expected Threat (xT) — value of having the ball at position (x, y).

    Based on the original Karun Singh 16×12 grid.
    Here we approximate with a smooth formula.

    Higher near the opponent's goal, lower near own goal and wide areas.
    """

    def predict(self, x: float, y: float) -> float:
        """Return xT ∈ [0, 1]."""
        # Normalise pitch position
        nx = x / PITCH_W          # 0 (own goal) → 1 (opp goal)
        ny = abs(y - PITCH_H / 2) / (PITCH_H / 2)   # 0=centre, 1=wide

        # Threat increases exponentially towards goal
        threat = (nx ** 3) * (1 - 0.3 * ny)
        return round(max(0.0, min(1.0, threat)), 4)

    def best_position_in_area(
        self, candidates: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Given several possible positions, return the one with highest xT."""
        return max(candidates, key=lambda p: self.predict(*p))


# ─── Tactical Coach ───────────────────────────────────────────────────────────

@dataclass
class TacticalSituation:
    """Snapshot of a tactical situation to analyse."""
    ball_position:        Tuple[float, float]        # (wx, wy) metres
    possessor_id:         int
    teammate_positions:   List[Tuple[float, float]]
    opponent_positions:   List[Tuple[float, float]]
    frame_index:          int
    timestamp_s:          float
    current_xg:           float = 0.0
    current_xt:           float = 0.0


@dataclass
class TacticalAdvice:
    situation_description: str
    recommended_action:    str
    reason:               str
    alternatives:         List[str]
    xg_if_shot:           float
    xt_current:           float
    confidence:           float


class TacticalCoach:
    """
    Generates tactical advice for a given match situation.

    Uses Claude API (via Anthropic) for natural language generation.
    Falls back to rule-based advice if API is unavailable.

    Usage
    -----
    coach = TacticalCoach()
    situation = TacticalSituation(...)
    advice = await coach.analyse(situation)
    print(advice.recommended_action)
    """

    CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
    MODEL          = "claude-sonnet-4-20250514"

    def __init__(self) -> None:
        self.xg_model = xGModel()
        self.xt_model = xTModel()

    # ── Public API ──────────────────────────────────────────────────────────

    def analyse_sync(self, situation: TacticalSituation) -> TacticalAdvice:
        """Rule-based tactical analysis (no API call required)."""
        bx, by = situation.ball_position

        xg   = self.xg_model.predict(bx, by)
        xt   = self.xt_model.predict(bx, by)

        teammate_xts = [
            (pos, self.xt_model.predict(*pos))
            for pos in situation.teammate_positions
        ]
        best_pass = max(teammate_xts, key=lambda t: t[1]) if teammate_xts else None

        # Decision logic
        if xg > 0.25:
            action = "SHOOT"
            reason = f"High-value shooting position (xG={xg:.2f}). Take the shot."
        elif best_pass and best_pass[1] > xt * 1.3:
            bpx, bpy = best_pass[0]
            action = f"PASS to ({bpx:.0f}m, {bpy:.0f}m)"
            reason = (
                f"A teammate is in a much higher threat position "
                f"(xT={best_pass[1]:.3f} vs current {xt:.3f}). "
                f"Through pass increases goal probability significantly."
            )
        elif self._in_wide_position(bx, by):
            action = "CROSS or DRIBBLE inside"
            reason = "Wide position limits shooting angle. Look for cutback or cross."
        else:
            action = "RETAIN POSSESSION"
            reason = (
                f"Current xT={xt:.3f} and xG={xg:.2f} are low. "
                f"Recycle possession and wait for better opportunity."
            )

        alternatives = self._generate_alternatives(
            bx, by, xg, xt, teammate_xts
        )

        return TacticalAdvice(
            situation_description=self._describe_situation(situation, xg, xt),
            recommended_action=action,
            reason=reason,
            alternatives=alternatives,
            xg_if_shot=xg,
            xt_current=xt,
            confidence=0.75,
        )

    async def analyse_with_llm(
        self, situation: TacticalSituation, api_key: str
    ) -> TacticalAdvice:
        """
        Enhanced analysis using Claude for natural language explanation.
        Requires a valid Anthropic API key.
        """
        rule_advice = self.analyse_sync(situation)

        prompt = self._build_prompt(situation, rule_advice)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    self.CLAUDE_API_URL,
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.MODEL,
                        "max_tokens": 300,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                llm_text = resp.json()["content"][0]["text"]
                rule_advice.reason = llm_text
                rule_advice.confidence = 0.90
        except Exception as e:
            print(f"[TacticalCoach] LLM unavailable ({e}), using rule-based advice.")

        return rule_advice

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _in_wide_position(x: float, y: float) -> bool:
        return y < 15.0 or y > 53.0

    def _generate_alternatives(
        self,
        bx: float,
        by: float,
        xg: float,
        xt: float,
        teammate_xts: List,
    ) -> List[str]:
        alts = []
        if xg > 0.05:
            alts.append(f"Shoot (xG={xg:.2f})")
        if teammate_xts:
            top3 = sorted(teammate_xts, key=lambda t: -t[1])[:3]
            for pos, t_xt in top3:
                alts.append(f"Pass to ({pos[0]:.0f}m, {pos[1]:.0f}m) — xT={t_xt:.3f}")
        alts.append("Dribble to create space")
        return alts[:4]

    def _describe_situation(
        self, s: TacticalSituation, xg: float, xt: float
    ) -> str:
        bx, by = s.ball_position
        return (
            f"Ball at ({bx:.0f}m, {by:.0f}m) | "
            f"xG={xg:.2f} | xT={xt:.3f} | "
            f"{len(s.teammate_positions)} teammates visible | "
            f"{len(s.opponent_positions)} opponents nearby"
        )

    def _build_prompt(
        self, situation: TacticalSituation, advice: TacticalAdvice
    ) -> str:
        bx, by = situation.ball_position
        return (
            f"You are an elite football tactical analyst. "
            f"A player has the ball at position ({bx:.0f}m, {by:.0f}m) on a 105×68m pitch. "
            f"Expected Goals if shot now: {advice.xg_if_shot:.2f}. "
            f"Expected Threat of current position: {advice.xt_current:.3f}. "
            f"There are {len(situation.teammate_positions)} visible teammates and "
            f"{len(situation.opponent_positions)} nearby opponents. "
            f"The rule-based system suggests: {advice.recommended_action}. "
            f"In 2-3 concise sentences, explain the tactical reasoning as a coach would."
        )