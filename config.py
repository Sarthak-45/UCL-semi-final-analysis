"""
Central configuration for the UCL 2025-26 Semifinal Predictor.
All match data, team metadata, API settings and simulation parameters live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")

# ── Simulation ─────────────────────────────────────────────────────────────────
N_SIMULATIONS = int(os.getenv("MONTE_CARLO_SIMULATIONS", 100_000))
RANDOM_SEED   = int(os.getenv("RANDOM_SEED", 42))

# ── football-data.org endpoints ────────────────────────────────────────────────
FD_BASE_URL   = "https://api.football-data.org/v4"
FD_HEADERS    = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}

# ── Team Registry ──────────────────────────────────────────────────────────────
# football-data.org team IDs + Wikipedia-sourced crest fallback URLs
TEAMS = {
    "Arsenal": {
        "fd_id":       57,
        "league":      "Premier League",
        "league_code": "PL",
        "country":     "England",
        "color":       "#EF0107",   # Gunners red (branding)
        "chart_color": "#EF0107",   # chart bars/lines
        "color2":      "#FFFFFF",
        "crest_url":   "https://crests.football-data.org/57.png",
    },
    "Atletico Madrid": {
        "fd_id":       78,
        "league":      "La Liga",
        "league_code": "PD",
        "country":     "Spain",
        "color":       "#CE3524",   # Atleti red (branding)
        "chart_color": "#4A90D9",   # steel blue — distinct from Arsenal red in charts
        "color2":      "#FFFFFF",
        "crest_url":   "https://crests.football-data.org/78.png",
    },
    "Bayern Munich": {
        "fd_id":       5,
        "league":      "Bundesliga",
        "league_code": "BL1",
        "country":     "Germany",
        "color":       "#DC052D",   # Bayern red (branding)
        "chart_color": "#DC052D",   # chart bars/lines
        "color2":      "#0066B2",
        "crest_url":   "https://crests.football-data.org/5.png",
    },
    "PSG": {
        "fd_id":       524,
        "league":      "Ligue 1",
        "league_code": "FL1",
        "country":     "France",
        "color":       "#003087",   # PSG navy (branding)
        "chart_color": "#00A0E9",   # PSG sky blue — distinct from Bayern red in charts
        "color2":      "#DA291C",
        "crest_url":   "https://crests.football-data.org/524.png",
    },
}

# ── UCL Trophy / Brand ─────────────────────────────────────────────────────────
UCL_LOGO_URL = "https://upload.wikimedia.org/wikipedia/en/thumb/f/f5/UEFA_Champions_League.svg/200px-UEFA_Champions_League.svg.png"

# ── Semifinal Fixtures ─────────────────────────────────────────────────────────
# second_leg_home = team playing the second leg at home
SEMIFINALS = [
    {
        "id":               "semi1",
        "label":            "Semi-Final 1",
        "home":             "Arsenal",       # first-leg home → second-leg AWAY
        "away":             "Atletico Madrid",  # second-leg HOME
        "first_leg_home_goals": 1,
        "first_leg_away_goals": 1,
        # Second leg: Arsenal travel to Atletico
        "second_leg_home": "Atletico Madrid",
        "second_leg_away": "Arsenal",
    },
    {
        "id":               "semi2",
        "label":            "Semi-Final 2",
        # First leg at Bayern, PSG won 5-4 → second leg at PSG
        "home":             "Bayern Munich",
        "away":             "PSG",
        "first_leg_home_goals": 4,
        "first_leg_away_goals": 5,
        "second_leg_home": "PSG",
        "second_leg_away": "Bayern Munich",
    },
]

# ── Monte Carlo λ (Poisson expected goals for second leg) ──────────────────────
# Derived from: UCL season avg, H2H adjustment, home-away factor, defensive strength
# These are overridable from the Streamlit sidebar.
EXPECTED_GOALS = {
    "semi1": {
        "Atletico Madrid": 1.25,   # Home, defensive side, UCL experience
        "Arsenal":         1.05,   # Away at Atletico, xG tempered by Simeone block
    },
    "semi2": {
        "PSG":             1.35,   # Home at Parc des Princes, protect lead
        "Bayern Munich":   2.15,   # Home 2nd leg, must chase → elevated λ
    },
}

# ── AI Models ─────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"   # primary — fast, generous free-tier quota
OPENAI_MODEL  = "gpt-4o-mini"       # fallback — cheap, strong reasoning
