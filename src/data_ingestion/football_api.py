"""
Live data fetcher using the football-data.org v4 API (free tier).

Provides:
  • Current domestic league standings for all four teams
  • UCL match results for the current season
  • Team season stats (goals scored/conceded, form string)

All methods return None / empty DataFrame silently on failure so the rest
of the app can fall back to the embedded historical data without crashing.
"""

import requests
import pandas as pd
from typing import Optional
from config import FD_BASE_URL, FD_HEADERS, TEAMS

# Map league code → competition id on football-data.org
LEAGUE_COMPETITION_IDS = {
    "PL":  2021,   # Premier League
    "PD":  2014,   # La Liga
    "BL1": 2002,   # Bundesliga
    "FL1": 2015,   # Ligue 1
    "UCL": 2001,   # UEFA Champions League
}

_TIMEOUT = 8  # seconds


def _get(endpoint: str) -> Optional[dict]:
    """Issue a GET request; return JSON dict or None on any error."""
    if not FD_HEADERS.get("X-Auth-Token"):
        return None
    try:
        url = f"{FD_BASE_URL}/{endpoint}"
        resp = requests.get(url, headers=FD_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ── Standings ──────────────────────────────────────────────────────────────────

def get_league_standings(league_code: str) -> Optional[pd.DataFrame]:
    """
    Fetch the current standings table for a domestic league.
    Returns a DataFrame with columns: position, team, played, won, draw, lost,
    goalsFor, goalsAgainst, goalDifference, points.
    """
    comp_id = LEAGUE_COMPETITION_IDS.get(league_code)
    if comp_id is None:
        return None
    data = _get(f"competitions/{comp_id}/standings")
    if not data:
        return None
    try:
        table = data["standings"][0]["table"]
        rows = []
        for entry in table:
            rows.append({
                "position":          entry["position"],
                "team":              entry["team"]["name"],
                "played":            entry["playedGames"],
                "won":               entry["won"],
                "draw":              entry["draw"],
                "lost":              entry["lost"],
                "goalsFor":          entry["goalsFor"],
                "goalsAgainst":      entry["goalsAgainst"],
                "goalDifference":    entry["goalDifference"],
                "points":            entry["points"],
                "form":              entry.get("form", ""),
            })
        return pd.DataFrame(rows)
    except (KeyError, IndexError, TypeError):
        return None


def get_team_current_form(team_name: str) -> Optional[dict]:
    """
    Return a condensed form dict for a team from its domestic league standings:
    {position, played, won, drawn, lost, gf, ga, gd, points, form_string}
    """
    team_cfg = TEAMS.get(team_name)
    if not team_cfg:
        return None
    df = get_league_standings(team_cfg["league_code"])
    if df is None or df.empty:
        return None
    # Match by partial name (API names sometimes differ slightly)
    mask = df["team"].str.contains(
        team_name.split()[0], case=False, na=False
    )
    row = df[mask]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "league":   team_cfg["league"],
        "position": int(r["position"]),
        "played":   int(r["played"]),
        "won":      int(r["won"]),
        "drawn":    int(r["draw"]),
        "lost":     int(r["lost"]),
        "gf":       int(r["goalsFor"]),
        "ga":       int(r["goalsAgainst"]),
        "gd":       int(r["goalDifference"]),
        "points":   int(r["points"]),
        "form_str": r.get("form", ""),
    }


# ── UCL Matches ────────────────────────────────────────────────────────────────

def get_ucl_matches(season: str = "2024") -> Optional[pd.DataFrame]:
    """
    Fetch all UCL matches for a given season year string (e.g. '2024' = 2024-25).
    Returns DataFrame with: matchday, stage, home_team, away_team, home_score,
    away_score, status, utc_date.
    """
    data = _get(f"competitions/2001/matches?season={season}")
    if not data:
        return None
    try:
        rows = []
        for m in data.get("matches", []):
            score = m.get("score", {})
            ft    = score.get("fullTime", {})
            rows.append({
                "matchday":   m.get("matchday"),
                "stage":      m.get("stage"),
                "home_team":  m["homeTeam"]["name"],
                "away_team":  m["awayTeam"]["name"],
                "home_score": ft.get("home"),
                "away_score": ft.get("away"),
                "status":     m.get("status"),
                "utc_date":   m.get("utcDate"),
            })
        return pd.DataFrame(rows)
    except (KeyError, TypeError):
        return None


def get_team_ucl_stats(team_name: str, season: str = "2024") -> Optional[dict]:
    """
    Compute UCL season stats for a team from the match data:
    {played, wins, draws, losses, goals_scored, goals_conceded, xg_avg_scored}
    """
    df = get_ucl_matches(season)
    if df is None or df.empty:
        return None
    df_finished = df[df["status"] == "FINISHED"].copy()
    mask_home = df_finished["home_team"].str.contains(
        team_name.split()[0], case=False, na=False
    )
    mask_away = df_finished["away_team"].str.contains(
        team_name.split()[0], case=False, na=False
    )
    home_games = df_finished[mask_home]
    away_games = df_finished[mask_away]

    def _results(games: pd.DataFrame, is_home: bool):
        if games.empty:
            return 0, 0, 0, 0, 0
        gf = games["home_score"].sum() if is_home else games["away_score"].sum()
        ga = games["away_score"].sum() if is_home else games["home_score"].sum()
        if is_home:
            w = (games["home_score"] > games["away_score"]).sum()
            d = (games["home_score"] == games["away_score"]).sum()
        else:
            w = (games["away_score"] > games["home_score"]).sum()
            d = (games["away_score"] == games["home_score"]).sum()
        l = len(games) - w - d
        return int(w), int(d), int(l), int(gf), int(ga)

    hw, hd, hl, hgf, hga = _results(home_games, True)
    aw, ad, al, agf, aga = _results(away_games, False)

    played = len(home_games) + len(away_games)
    wins   = hw + aw
    draws  = hd + ad
    losses = hl + al
    gf     = hgf + agf
    ga     = hga + aga

    return {
        "played":        played,
        "wins":          wins,
        "draws":         draws,
        "losses":        losses,
        "goals_scored":  gf,
        "goals_conceded":ga,
        "avg_scored":    round(gf / played, 2) if played else 0,
        "avg_conceded":  round(ga / played, 2) if played else 0,
    }
