"""
AI-powered match analysis using Google Gemini.

Sends a structured prompt containing:
  • First-leg result and aggregate situation
  • Both teams' UCL last-5-year records
  • Head-to-head history
  • Domestic league form
  • Monte Carlo advancement probabilities
  • Logistic regression probability

Gemini returns a detailed tactical analysis.  If the API key is missing or the
call fails the module returns a rich rule-based fallback analysis instead.
"""

import os
import textwrap
from typing import Optional

from src.data_ingestion.historical_data import (
    get_ucl_history_dataframe,
    get_league_form,
    compute_h2h_stats,
    compute_ucl_form_score,
)
from config import GEMINI_API_KEY, GEMINI_MODEL, OPENAI_API_KEY, OPENAI_MODEL, TEAMS


# ── Provider helpers (lazy-loaded so missing packages don't crash the app) ─────

def _get_gemini_model():
    """Return a configured Gemini GenerativeModel or None."""
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception:
        return None


def _call_openai(prompt: str) -> str:
    """Call OpenAI ChatCompletion and return the response text."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system",
             "content": "You are an expert UEFA Champions League analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1200,
    )
    return response.choices[0].message.content.strip()


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(
    home_team:        str,
    away_team:        str,
    first_leg_result: str,
    aggregate:        str,
    mc_prob_home:     float,
    mc_prob_away:     float,
    lr_prob_leading:  float,
    leading_team:     str,
) -> str:
    """Construct the structured Gemini prompt."""

    def _ucl_summary(team: str) -> str:
        df = get_ucl_history_dataframe(team)
        if df.empty:
            return "No data available."
        lines = []
        for _, row in df.iterrows():
            lines.append(
                f"  {row['season']}: Reached {row['stage']} | "
                f"W{row['ucl_wins']} D{row['ucl_draws']} L{row['ucl_losses']} | "
                f"GF {row['goals_scored']} GA {row['goals_conceded']}"
            )
        return "\n".join(lines)

    def _league_summary(team: str) -> str:
        form = get_league_form(team)
        if not form:
            return "No data."
        last5 = " ".join(form.get("last_5", []))
        return (
            f"  {form['league']} | Position: {form['position']} | "
            f"P{form['played']} W{form['won']} D{form['drawn']} L{form['lost']} | "
            f"GF {form['gf']} GA {form['ga']} | Pts {form['points']} | "
            f"Last 5: {last5}"
        )

    h2h_key  = f"{home_team}_vs_{away_team}"
    h2h      = compute_h2h_stats(h2h_key)
    if h2h["played"] == 0:
        h2h_key  = f"{away_team}_vs_{home_team}"
        h2h      = compute_h2h_stats(h2h_key)
        h2h_text = f"  {away_team} W{h2h['wins']} D{h2h['draws']} L{h2h['losses']} (from {away_team}'s perspective)"
    else:
        h2h_text = f"  {home_team} W{h2h['wins']} D{h2h['draws']} L{h2h['losses']} (from {home_team}'s perspective)"

    prompt = textwrap.dedent(f"""
    You are an expert UEFA Champions League analyst. Provide a detailed, insightful
    tactical analysis for the following second-leg knockout tie.

    ═══════════════════════════════════════════════════════
    MATCH: {home_team} (home) vs {away_team} (away) — Second Leg
    FIRST LEG RESULT: {first_leg_result}
    CURRENT AGGREGATE: {aggregate}
    ═══════════════════════════════════════════════════════

    ── UCL HISTORY (Last 5 Seasons) ──
    {home_team}:
{_ucl_summary(home_team)}

    {away_team}:
{_ucl_summary(away_team)}

    ── HEAD-TO-HEAD RECORD ──
{h2h_text}

    ── DOMESTIC LEAGUE FORM (2024-25) ──
    {home_team}:
{_league_summary(home_team)}

    {away_team}:
{_league_summary(away_team)}

    ── QUANTITATIVE MODEL PREDICTIONS ──
    Monte Carlo simulation ({100_000:,} trials):
      • {home_team} advances: {mc_prob_home*100:.1f}%
      • {away_team} advances: {mc_prob_away*100:.1f}%
    Logistic Regression — probability {leading_team} advances: {lr_prob_leading*100:.1f}%

    ═══════════════════════════════════════════════════════
    Please provide your analysis in the following structured format:

    ## TACTICAL OVERVIEW
    (2-3 paragraphs on both teams' playing styles and how they match up)

    ## KEY FACTORS FOR {home_team.upper()}
    (3 bullet points on why they could advance)

    ## KEY FACTORS FOR {away_team.upper()}
    (3 bullet points on why they could advance)

    ## PREDICTED SCORELINE
    Give your predicted second-leg score and overall aggregate.

    ## VERDICT
    One clear prediction with confidence level (High / Medium / Low) and a
    one-sentence justification.
    ═══════════════════════════════════════════════════════
    """)
    return prompt.strip()


# ── Fallback rule-based analysis ───────────────────────────────────────────────

def _fallback_analysis(
    home_team:    str,
    away_team:    str,
    mc_prob_home: float,
    mc_prob_away: float,
    leading_team: str,
) -> str:
    """Generate a structured analysis without the Gemini API."""
    favoured = home_team if mc_prob_home >= mc_prob_away else away_team
    underdog = away_team if favoured == home_team else home_team
    fav_prob = max(mc_prob_home, mc_prob_away) * 100
    und_prob = min(mc_prob_home, mc_prob_away) * 100

    home_ucl = compute_ucl_form_score(home_team)
    away_ucl = compute_ucl_form_score(away_team)
    home_league = get_league_form(home_team)
    away_league = get_league_form(away_team)

    home_pos = home_league.get("position", "?")
    away_pos = away_league.get("position", "?")
    home_lg  = home_league.get("league", "")
    away_lg  = away_league.get("league", "")

    h2h  = compute_h2h_stats(f"{home_team}_vs_{away_team}")
    if h2h["played"] == 0:
        h2h = compute_h2h_stats(f"{away_team}_vs_{home_team}")

    confidence = "High" if abs(mc_prob_home - mc_prob_away) > 0.20 else \
                 "Medium" if abs(mc_prob_home - mc_prob_away) > 0.10 else "Low"

    return textwrap.dedent(f"""
    ## TACTICAL OVERVIEW
    {home_team} host the second leg with home-field advantage at their disposal. Both
    sides have demonstrated elite-level UCL pedigree over the last five seasons —
    {home_team} carrying a UCL form score of {home_ucl:.2f} while {away_team} sit at
    {away_ucl:.2f} (scale 0–1). The tie is finely balanced, though the data leans
    toward **{favoured}** ({fav_prob:.1f}% advancement probability).

    Domestically, {home_team} sit **{home_pos}{_ordinal(home_pos)} in {home_lg}** and
    {away_team} sit **{away_pos}{_ordinal(away_pos)} in {away_lg}** — both sides are
    in strong form heading into this decisive encounter. The H2H record shows
    {h2h['wins']} wins, {h2h['draws']} draws and {h2h['losses']} losses
    (from {home_team}'s perspective across {h2h['played']} meetings).

    ## KEY FACTORS FOR {home_team.upper()}
    • **Home advantage** — historically worth an extra ~0.3 goals per game in UCL ties.
    • **League form** — {home_pos}{_ordinal(home_pos)} in {home_lg} signals a high-performing squad.
    • **UCL pedigree** — form score of {home_ucl:.2f} reflects consistent knockout-stage performances.

    ## KEY FACTORS FOR {away_team.upper()}
    • **Away goals capability** — an away goal immediately flips the pressure.
    • **First-leg momentum** — experience of navigating these high-stakes ties is invaluable.
    • **League form** — {away_pos}{_ordinal(away_pos)} in {away_lg} demonstrates squad depth.

    ## PREDICTED SCORELINE
    Based on Poisson expected-goals modelling, the most likely second-leg score is
    **{home_team} 1–1 {away_team}**, creating tension going to extra time.

    ## VERDICT
    **{favoured} to advance** — Confidence: **{confidence}**
    The quantitative models give {favoured} a {fav_prob:.1f}% chance, driven by
    {"home advantage and superior UCL form" if favoured == home_team else "aggregate lead and away-goals threat"}.
    """).strip()


def _ordinal(n) -> str:
    try:
        n = int(n)
        return {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th")
    except (ValueError, TypeError):
        return ""


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_analysis(
    home_team:        str,
    away_team:        str,
    first_leg_result: str,
    aggregate:        str,
    mc_prob_home:     float,
    mc_prob_away:     float,
    lr_prob_leading:  float,
    leading_team:     str,
) -> str:
    """
    Generate a full match analysis string using a three-tier fallback chain:
      1. Google Gemini  (GEMINI_API_KEY in .env)
      2. OpenAI         (OPENAI_API_KEY in .env)
      3. Rule-based     (always available, no API needed)
    """
    prompt = _build_prompt(
        home_team, away_team, first_leg_result, aggregate,
        mc_prob_home, mc_prob_away, lr_prob_leading, leading_team,
    )
    errors = []

    # ── Tier 1: Gemini ─────────────────────────────────────────────────────────
    gemini_model = _get_gemini_model()
    if gemini_model:
        try:
            return gemini_model.generate_content(prompt).text.strip()
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

    # ── Tier 2: OpenAI ─────────────────────────────────────────────────────────
    if OPENAI_API_KEY:
        try:
            return _call_openai(prompt)
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")

    # ── Tier 3: Rule-based fallback ────────────────────────────────────────────
    notice = ""
    if errors:
        joined = " | ".join(errors)
        notice = f"> ⚠️ AI providers unavailable ({joined}) — showing rule-based analysis.\n\n"

    return notice + _fallback_analysis(
        home_team, away_team, mc_prob_home, mc_prob_away, leading_team
    )
