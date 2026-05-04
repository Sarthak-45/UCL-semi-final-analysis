"""
Feature engineering pipeline.

Converts raw historical / live data into a normalised feature vector that is
consumed by both the logistic-regression classifier and the Monte Carlo engine.

Feature vector (per semifinal matchup):
  [0]  agg_diff           – aggregate goal difference going into second leg
                            (positive = leading_team ahead)
  [1]  home_xg            – expected goals per game for second-leg home team
  [2]  away_xg            – expected goals per game for second-leg away team
  [3]  form_diff          – UCL form score difference (leading − trailing, −1..1)
  [4]  h2h_win_rate        – H2H win rate of the leading team  (0..1)
  [5]  league_strength_diff – domestic-league tier gap (leading − trailing)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.data_ingestion.historical_data import (
    compute_h2h_stats,
    compute_ucl_form_score,
    get_league_form,
)
from src.data_ingestion.football_api import get_team_current_form, get_team_ucl_stats

# League strength ratings (higher = harder league / more UCL pedigree)
LEAGUE_STRENGTH = {
    "Premier League": 9.2,
    "La Liga":        9.0,
    "Bundesliga":     8.5,
    "Ligue 1":        7.8,
    "Serie A":        8.3,
    "Eredivisie":     7.0,
}

# football-data.org competition → team fd_id mapping (for UCL xG lookup)
# We fall back to embedded averages when the live API is unavailable.
EMBEDDED_UCL_AVG = {
    "Arsenal":          {"avg_scored": 2.17, "avg_conceded": 0.75},
    "Atletico Madrid":  {"avg_scored": 1.33, "avg_conceded": 0.67},
    "Bayern Munich":    {"avg_scored": 2.50, "avg_conceded": 0.92},
    "PSG":              {"avg_scored": 2.33, "avg_conceded": 0.75},
}

# Home-field advantage multiplier (industry standard ≈ +10 % goals)
HOME_ADVANTAGE = 1.10
AWAY_PENALTY   = 0.92


@dataclass
class MatchupFeatures:
    """All engineered features for one semifinal matchup."""

    # Identifiers
    home_team:  str
    away_team:  str
    leading_team:   str
    trailing_team:  str

    # Raw / computed
    first_leg_home_goals: int
    first_leg_away_goals: int
    agg_diff:       int        # leading_team_goals − trailing_team_goals (always ≥ 0)
    h2h_stats:      dict = field(default_factory=dict)

    # Feature vector values
    home_xg:            float = 1.20
    away_xg:            float = 1.20
    form_diff:          float = 0.0
    h2h_win_rate:       float = 0.5
    league_strength_diff: float = 0.0

    # Derived lambda values for Monte Carlo (may be overridden by UI)
    lambda_home: float = 1.20
    lambda_away: float = 1.20

    def to_array(self) -> np.ndarray:
        """Return feature vector as a 1-D numpy array (matches training schema)."""
        return np.array([
            self.agg_diff,
            self.home_xg,
            self.away_xg,
            self.form_diff,
            self.h2h_win_rate,
            self.league_strength_diff,
        ], dtype=float)


def _ucl_avg(team: str) -> dict:
    """Try live API first; fall back to embedded average."""
    live = get_team_ucl_stats(team, season="2024")
    if live and live["played"] >= 4:
        return {
            "avg_scored":    live["avg_scored"],
            "avg_conceded":  live["avg_conceded"],
        }
    return EMBEDDED_UCL_AVG.get(team, {"avg_scored": 1.5, "avg_conceded": 1.0})


def _league_form(team: str) -> dict:
    """Try live API; fall back to embedded league form."""
    live = get_team_current_form(team)
    if live:
        return live
    return get_league_form(team)


def build_features(
    home_team: str,
    away_team: str,
    first_leg_home_goals: int,
    first_leg_away_goals: int,
    leading_team: Optional[str] = None,
) -> MatchupFeatures:
    """
    Build the full MatchupFeatures object for a given second-leg fixture.

    Parameters
    ----------
    home_team             : Team playing at home in the second leg.
    away_team             : Team playing away in the second leg.
    first_leg_home_goals  : Goals scored by the first-leg home team.
    first_leg_away_goals  : Goals scored by the first-leg away team.
    leading_team          : Team with the aggregate lead (auto-detected if None).
    """
    # ── Determine leading / trailing teams ────────────────────────────────────
    # First-leg home = second-leg away (they swapped venues).
    first_leg_away_team = home_team   # second-leg home team was away in leg 1
    first_leg_home_team = away_team   # second-leg away team was home in leg 1

    # Aggregate from leading team's perspective
    leading_agg  = first_leg_away_goals   # first-leg away team's goals = second-leg home team's leg-1 tally?
    # Let's clarify: first_leg_home_goals are scored by first_leg_home_team (= away in 2nd leg)
    agg_second_leg_home  = first_leg_away_goals  # goals scored by 2nd-leg-home in leg 1
    agg_second_leg_away  = first_leg_home_goals  # goals scored by 2nd-leg-away in leg 1

    diff_2nd_home = agg_second_leg_home - agg_second_leg_away

    if leading_team is None:
        leading_team  = home_team if diff_2nd_home >= 0 else away_team
        trailing_team = away_team if diff_2nd_home >= 0 else home_team
    else:
        trailing_team = away_team if leading_team == home_team else home_team

    agg_diff = abs(diff_2nd_home)

    # ── H2H features ──────────────────────────────────────────────────────────
    h2h_key1 = f"{leading_team}_vs_{trailing_team}"
    h2h_key2 = f"{trailing_team}_vs_{leading_team}"
    h2h = compute_h2h_stats(h2h_key1)
    if h2h["played"] == 0:
        # Try reverse direction and invert win rate
        h2h = compute_h2h_stats(h2h_key2)
        h2h_win_rate = 1.0 - h2h["win_rate"]
    else:
        h2h_win_rate = h2h["win_rate"]

    # ── UCL form ───────────────────────────────────────────────────────────────
    leading_form  = compute_ucl_form_score(leading_team,  last_n=3)
    trailing_form = compute_ucl_form_score(trailing_team, last_n=3)
    form_diff     = round(leading_form - trailing_form, 3)

    # ── Expected goals (λ) for second leg ─────────────────────────────────────
    home_ucl = _ucl_avg(home_team)
    away_ucl = _ucl_avg(away_team)

    # home team's scoring rate adjusted for away team's defensive record
    raw_home_xg = home_ucl["avg_scored"] * HOME_ADVANTAGE
    raw_away_xg = away_ucl["avg_scored"] * AWAY_PENALTY

    # Defensive suppression: scale attacker's xG by opponent avg conceded / league avg
    league_avg_conceded = 1.10
    home_defence_factor = home_ucl["avg_conceded"] / league_avg_conceded
    away_defence_factor = away_ucl["avg_conceded"] / league_avg_conceded

    lambda_home = round(raw_home_xg * away_defence_factor, 3)
    lambda_away = round(raw_away_xg * home_defence_factor, 3)

    # ── League strength ───────────────────────────────────────────────────────
    from config import TEAMS as TEAM_CFG
    leading_league  = TEAM_CFG.get(leading_team,  {}).get("league", "Premier League")
    trailing_league = TEAM_CFG.get(trailing_team, {}).get("league", "Premier League")
    ls_diff = LEAGUE_STRENGTH.get(leading_league, 8.0) - LEAGUE_STRENGTH.get(trailing_league, 8.0)

    return MatchupFeatures(
        home_team             = home_team,
        away_team             = away_team,
        leading_team          = leading_team,
        trailing_team         = trailing_team,
        first_leg_home_goals  = first_leg_home_goals,
        first_leg_away_goals  = first_leg_away_goals,
        agg_diff              = agg_diff,
        h2h_stats             = h2h,
        home_xg               = raw_home_xg,
        away_xg               = raw_away_xg,
        form_diff             = form_diff,
        h2h_win_rate          = h2h_win_rate,
        league_strength_diff  = round(ls_diff, 2),
        lambda_home           = lambda_home,
        lambda_away           = lambda_away,
    )
