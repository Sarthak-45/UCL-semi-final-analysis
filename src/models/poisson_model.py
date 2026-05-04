"""
Poisson Goal Model (Dixon-Coles framework).

Football goals are well-modelled by independent Poisson processes:
    λ_home = attack_home × defense_away × home_advantage × league_average
    λ_away = attack_away × defense_home × league_average

Attack and defense ratings are estimated from UCL historical data using a
weighted method-of-moments estimator (more recent seasons carry more weight).

These λ values replace the static config defaults in the Monte Carlo
simulator, giving predictions that are grounded in each team's actual
scoring / conceding rates rather than hand-tuned constants.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from src.data_ingestion.historical_data import UCL_HISTORY

LEAGUE_AVG_GOALS = 1.35   # mean goals per team per UCL game (approx 2020-25)
HOME_ADVANTAGE   = 1.12   # UCL home teams score ~12 % more goals on average


class PoissonGoalModel:
    """
    Poisson-based expected-goals estimator.

    Workflow:
        model = PoissonGoalModel().fit()
        lh, la = model.lambda_pair("PSG", "Bayern Munich")
    """

    def __init__(self) -> None:
        self._attack:  Dict[str, float] = {}
        self._defense: Dict[str, float] = {}
        self._fitted  = False

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, last_n: int = 3) -> "PoissonGoalModel":
        """
        Estimate attack / defense strengths from the last `last_n` UCL seasons.
        Linear ramp weighting — most recent season has weight `last_n`,
        oldest has weight 1.
        """
        ramp = list(range(1, last_n + 1))   # [1, 2, 3]

        for team, all_seasons in UCL_HISTORY.items():
            # Only count seasons where the team actually played
            played = [
                s for s in all_seasons
                if s["ucl_wins"] + s["ucl_draws"] + s["ucl_losses"] > 0
            ]
            recent = played[-last_n:]
            if not recent:
                self._attack[team]  = 1.0
                self._defense[team] = 1.0
                continue

            w = ramp[-len(recent):]
            total_w = sum(w)

            gf_wt = sum(
                (s["goals_scored"]   / (s["ucl_wins"] + s["ucl_draws"] + s["ucl_losses"])) * wt
                for s, wt in zip(recent, w)
            ) / total_w

            ga_wt = sum(
                (s["goals_conceded"] / (s["ucl_wins"] + s["ucl_draws"] + s["ucl_losses"])) * wt
                for s, wt in zip(recent, w)
            ) / total_w

            self._attack[team]  = round(gf_wt / LEAGUE_AVG_GOALS, 4)
            self._defense[team] = round(ga_wt / LEAGUE_AVG_GOALS, 4)

        self._fitted = True
        return self

    # ── Prediction ─────────────────────────────────────────────────────────────

    def lambda_pair(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Compute Poisson λ for both teams in a second-leg fixture.

        Returns (λ_home, λ_away).
        """
        if not self._fitted:
            self.fit()

        att_h = self._attack.get(home_team,  1.0)
        def_a = self._defense.get(away_team, 1.0)
        att_a = self._attack.get(away_team,  1.0)
        def_h = self._defense.get(home_team, 1.0)

        lh = att_h * def_a * HOME_ADVANTAGE * LEAGUE_AVG_GOALS
        la = att_a * def_h * LEAGUE_AVG_GOALS

        # Clamp to a sensible UCL range
        lh = round(float(np.clip(lh, 0.30, 4.0)), 3)
        la = round(float(np.clip(la, 0.30, 4.0)), 3)
        return lh, la

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def strength_table(self) -> List[dict]:
        """Sorted list of attack / defense ratings for all teams."""
        if not self._fitted:
            self.fit()
        return sorted(
            [
                {
                    "Team":           t,
                    "Attack rating":  round(self._attack.get(t, 1.0), 3),
                    "Defense rating": round(self._defense.get(t, 1.0), 3),
                    "Net strength":   round(
                        self._attack.get(t, 1.0) - self._defense.get(t, 1.0), 3
                    ),
                }
                for t in self._attack
            ],
            key=lambda x: x["Net strength"],
            reverse=True,
        )

    @property
    def attack_ratings(self) -> Dict[str, float]:
        if not self._fitted:
            self.fit()
        return dict(self._attack)

    @property
    def defense_ratings(self) -> Dict[str, float]:
        if not self._fitted:
            self.fit()
        return dict(self._defense)
