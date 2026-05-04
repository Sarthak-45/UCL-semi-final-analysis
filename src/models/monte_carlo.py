"""
Poisson Monte Carlo simulator for UCL second-leg knockout ties.

Rules applied (UEFA post-2021-22):
  • No away-goals rule.
  • If aggregate is level after 90 min → 30 min extra time.
  • If still level after ET → penalty shoot-out (modelled as 50/50 per team
    but skewed by form/penalty record if available).
  • The team with the higher aggregate at any point advances.

Outputs
-------
  advancement_probs  : dict {team_name: probability}   (sums to 1.0)
  score_distribution : dict {(home_goals, away_goals): count}  (second-leg only)
  aggregate_outcomes : dict {outcome_label: count}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config import N_SIMULATIONS, RANDOM_SEED


@dataclass
class SimulationResult:
    """Container for all Monte Carlo outputs."""

    home_team:   str
    away_team:   str
    n_sims:      int

    # Advancement probabilities (0–1)
    prob_home_advances:  float = 0.0
    prob_away_advances:  float = 0.0

    # How the tie was decided
    pct_normal_time:    float = 0.0   # decided in 90 min
    pct_extra_time:     float = 0.0   # decided in extra time
    pct_penalties:      float = 0.0   # decided on pens (50-50 after ET)

    # Score frequency matrix — shape (MAX_G, MAX_G), [home_goals, away_goals]
    # Stored as numpy array so it survives cache serialization on all platforms
    score_distribution: object = field(default_factory=lambda: np.zeros((8, 8), dtype=np.int32))

    # Most likely second-leg scorelines [(  (hg, ag), count ), ...]
    top_scorelines: list = field(default_factory=list)

    # Aggregate goal totals distribution  {total_goals: count}
    agg_total_distribution: dict = field(default_factory=dict)

    # Expected second-leg goals (means of Poisson draws)
    expected_home_goals: float = 0.0
    expected_away_goals: float = 0.0


class MonteCarlo:
    """
    Run vectorised Poisson simulations for a two-legged UCL tie.

    Parameters
    ----------
    home_team            : Name of the second-leg home team.
    away_team            : Name of the second-leg away team.
    first_leg_home_goals : Goals scored by the first-leg home team.
    first_leg_away_goals : Goals scored by the first-leg away team.
    lambda_home          : Expected goals per game for second-leg home team.
    lambda_away          : Expected goals per game for second-leg away team.
    penalty_edge         : Extra probability edge for home team in shoot-out
                           (positive = home team better on pens).  Default 0.
    n_simulations        : Number of Monte Carlo trials.
    seed                 : Random seed for reproducibility.
    """

    def __init__(
        self,
        home_team:            str,
        away_team:            str,
        first_leg_home_goals: int,
        first_leg_away_goals: int,
        lambda_home:          float,
        lambda_away:          float,
        penalty_edge:         float = 0.0,
        n_simulations:        int   = N_SIMULATIONS,
        seed:                 int   = RANDOM_SEED,
    ) -> None:
        self.home_team            = home_team
        self.away_team            = away_team
        self.first_leg_home_goals = first_leg_home_goals
        self.first_leg_away_goals = first_leg_away_goals
        self.lambda_home          = max(lambda_home, 0.1)
        self.lambda_away          = max(lambda_away, 0.1)
        self.penalty_edge         = np.clip(penalty_edge, -0.3, 0.3)
        self.n_simulations        = n_simulations
        self.seed                 = seed

    # ── First-leg aggregate (from second-leg home team's perspective) ─────────
    @property
    def _agg_home_leg1(self) -> int:
        """Goals scored by the second-leg HOME team in leg 1 (they were AWAY)."""
        return self.first_leg_away_goals

    @property
    def _agg_away_leg1(self) -> int:
        """Goals scored by the second-leg AWAY team in leg 1 (they were HOME)."""
        return self.first_leg_home_goals

    # ── Simulation ────────────────────────────────────────────────────────────

    def run(self) -> SimulationResult:
        """Execute the Monte Carlo simulation and return a SimulationResult."""
        rng = np.random.default_rng(self.seed)

        # Draw second-leg goals from Poisson distributions (vectorised)
        home_goals_2l = rng.poisson(self.lambda_home, self.n_simulations)
        away_goals_2l = rng.poisson(self.lambda_away, self.n_simulations)

        # Aggregate totals
        home_agg = self._agg_home_leg1 + home_goals_2l
        away_agg = self._agg_away_leg1 + away_goals_2l

        # ── 90-minute classification ──────────────────────────────────────────
        home_wins_90   = home_agg > away_agg          # home advances at 90
        away_wins_90   = away_agg > home_agg          # away advances at 90
        goes_to_et     = home_agg == away_agg         # level → extra time

        n_home_90  = int(home_wins_90.sum())
        n_away_90  = int(away_wins_90.sum())
        n_et       = int(goes_to_et.sum())

        # ── Extra-time simulation (30 min ≈ 0.33 × 90-min rates) ─────────────
        et_lambda_home = self.lambda_home * 0.33
        et_lambda_away = self.lambda_away * 0.33

        et_home = rng.poisson(et_lambda_home, n_et)
        et_away = rng.poisson(et_lambda_away, n_et)

        home_wins_et  = (et_home > et_away).sum()
        away_wins_et  = (et_away > et_home).sum()
        n_pens        = n_et - int(home_wins_et) - int(away_wins_et)

        # ── Penalty shoot-out (50/50 ± edge) ─────────────────────────────────
        home_pen_prob = np.clip(0.5 + self.penalty_edge, 0.1, 0.9)
        pen_draws     = rng.random(n_pens)
        home_wins_pen = int((pen_draws < home_pen_prob).sum())
        away_wins_pen = n_pens - home_wins_pen

        # ── Aggregate advancement counts ──────────────────────────────────────
        total_home = n_home_90 + int(home_wins_et) + home_wins_pen
        total_away = n_away_90 + int(away_wins_et) + away_wins_pen

        prob_home = total_home / self.n_simulations
        prob_away = total_away / self.n_simulations

        # ── Score distribution — 2D numpy array [home_goals, away_goals] ────────
        max_g = 8
        bins  = np.arange(max_g + 1) - 0.5          # edges: -0.5, 0.5, … 7.5
        score_matrix, _, _ = np.histogram2d(
            home_goals_2l, away_goals_2l, bins=bins
        )
        score_matrix = score_matrix.astype(np.int32)  # (8, 8)

        # Top 10 scorelines by frequency
        flat_idx = np.argsort(score_matrix.ravel())[::-1][:10]
        top = [
            ((int(idx // max_g), int(idx % max_g)), int(score_matrix.ravel()[idx]))
            for idx in flat_idx
            if score_matrix.ravel()[idx] > 0
        ]

        # Aggregate total goals distribution
        agg_total = home_agg + away_agg
        uniq_t, cnt_t = np.unique(agg_total, return_counts=True)
        agg_total_dist = {int(u): int(c) for u, c in zip(uniq_t, cnt_t)}

        return SimulationResult(
            home_team            = self.home_team,
            away_team            = self.away_team,
            n_sims               = self.n_simulations,
            prob_home_advances   = round(prob_home, 4),
            prob_away_advances   = round(prob_away, 4),
            pct_normal_time      = round((n_home_90 + n_away_90) / self.n_simulations, 4),
            pct_extra_time       = round((int(home_wins_et) + int(away_wins_et)) / self.n_simulations, 4),
            pct_penalties        = round(n_pens / self.n_simulations, 4),
            score_distribution   = score_matrix,
            top_scorelines       = top,
            agg_total_distribution = agg_total_dist,
            expected_home_goals  = round(float(home_goals_2l.mean()), 3),
            expected_away_goals  = round(float(away_goals_2l.mean()), 3),
        )


def simulate_semifinal(semifinal_cfg: dict, overrides: Optional[dict] = None) -> SimulationResult:
    """
    High-level helper used by the Streamlit app.

    Parameters
    ----------
    semifinal_cfg : One entry from config.SEMIFINALS.
    overrides     : Optional dict with keys 'lambda_home' and 'lambda_away'
                    (from the Streamlit sidebar sliders).
    """
    from config import EXPECTED_GOALS

    sid = semifinal_cfg["id"]
    home = semifinal_cfg["second_leg_home"]
    away = semifinal_cfg["second_leg_away"]

    eg = EXPECTED_GOALS.get(sid, {})
    lambda_home = eg.get(home, 1.2)
    lambda_away = eg.get(away, 1.2)

    if overrides:
        lambda_home = overrides.get("lambda_home", lambda_home)
        lambda_away = overrides.get("lambda_away", lambda_away)

    sim = MonteCarlo(
        home_team            = home,
        away_team            = away,
        first_leg_home_goals = semifinal_cfg["first_leg_home_goals"],
        first_leg_away_goals = semifinal_cfg["first_leg_away_goals"],
        lambda_home          = lambda_home,
        lambda_away          = lambda_away,
    )
    return sim.run()
