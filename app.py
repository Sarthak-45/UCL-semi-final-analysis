"""
UCL 2025-26 Semifinal Predictor — Streamlit Dashboard
======================================================
Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import base64

from config import SEMIFINALS, TEAMS
from src.feature_engineering.features import build_features
from src.models.ensemble_model import UCLEnsembleModel
from src.models.monte_carlo import simulate_semifinal
from src.models.poisson_model import PoissonGoalModel
from src.data_ingestion.historical_data import (
    get_ucl_history_dataframe,
    get_league_form,
    compute_h2h_stats,
    H2H_RECORDS,
)
from src.ai_feedback.reasoning import generate_analysis

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UCL 2025-26 Semifinal Predictor",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — UCL night-sky theme ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', Arial, sans-serif;
}
.stApp {
    background: radial-gradient(ellipse at 15% 40%, #0d1b3e 0%, #050d1f 40%, #020810 100%);
    background-attachment: fixed;
    min-height: 100vh;
    color: #e8eaf6;
}

/* ── Animated gold stars ── */
@keyframes twinkle {
  0%,100% { opacity:0.15; transform:scale(1);   }
  50%      { opacity:1.0;  transform:scale(1.4); }
}
.star {
    position: fixed;
    border-radius: 50%;
    background: #FFD700;
    animation: twinkle linear infinite;
    pointer-events: none;
    z-index: 0;
}

/* ── Header banner ── */
.ucl-header {
    background: linear-gradient(135deg, #0a1628 0%, #1a2d5a 50%, #0a1628 100%);
    border: 1px solid #FFD700;
    border-radius: 16px;
    padding: 28px 40px;
    text-align: center;
    margin-bottom: 32px;
    box-shadow: 0 0 40px rgba(255,215,0,0.15);
}
.ucl-header h1 {
    color: #FFD700;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 10px 0 4px;
    text-shadow: 0 0 20px rgba(255,215,0,0.5);
}
.ucl-header p {
    color: #9db4d8;
    font-size: 1.0rem;
    margin: 0;
}

/* ── Semifinal cards ── */
.semi-card {
    background: linear-gradient(160deg, #0d1f3c 0%, #0a1628 100%);
    border: 1px solid #1e3a6e;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    position: relative;
    z-index: 1;
}
.semi-title {
    color: #FFD700;
    font-size: 1.2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 16px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Team name headings ── */
.team-name {
    font-size: 1.3rem;
    font-weight: 800;
    text-align: center;
    margin: 8px 0 2px;
}
.team-league {
    font-size: 0.78rem;
    text-align: center;
    color: #9db4d8;
    margin-bottom: 6px;
}

/* ── Score badge ── */
.score-badge {
    background: linear-gradient(135deg, #1a3a6e, #0d2a54);
    border: 2px solid #FFD700;
    border-radius: 40px;
    padding: 10px 28px;
    font-size: 1.8rem;
    font-weight: 900;
    color: #FFD700;
    text-align: center;
    letter-spacing: 4px;
    margin: 12px auto;
    display: inline-block;
    box-shadow: 0 0 20px rgba(255,215,0,0.2);
}
.score-label {
    font-size: 0.72rem;
    color: #9db4d8;
    text-align: center;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Probability bars ── */
.prob-bar-container {
    background: #0a1628;
    border-radius: 8px;
    height: 18px;
    margin: 6px 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* ── Stat pills ── */
.stat-pill {
    display: inline-block;
    background: #0d2240;
    border: 1px solid #1e3a6e;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #9db4d8;
    margin: 3px;
}
.stat-pill strong { color: #FFD700; }

/* ── AI analysis box ── */
.ai-box {
    background: linear-gradient(160deg, #0a1f3c 0%, #071428 100%);
    border: 1px solid #2a4a8a;
    border-left: 5px solid #FFD700;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 12px;
}
.ai-h2 {
    color: #FFD700 !important;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 18px 0 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid #1e3a6e;
    letter-spacing: 0.5px;
}
.ai-h3 {
    color: #9db4d8 !important;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 14px 0 6px;
}
.ai-p {
    color: #d0daf0;
    line-height: 1.85;
    margin: 8px 0;
    font-size: 0.91rem;
}
.ai-list {
    margin: 6px 0 14px 18px;
    padding: 0;
}
.ai-list li {
    color: #d0daf0;
    margin-bottom: 7px;
    line-height: 1.7;
    font-size: 0.91rem;
}
.ai-verdict {
    background: rgba(255,215,0,0.07);
    border: 1px solid rgba(255,215,0,0.25);
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 14px;
    color: #FFD700;
    font-weight: 600;
    font-size: 0.95rem;
}
/* ── Predicted finalists banner ── */
.finalist-card {
    background: linear-gradient(135deg, #0d1f3c, #0a1628);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    transition: box-shadow 0.3s;
}
.finalist-card:hover { box-shadow: 0 0 30px rgba(255,215,0,0.2); }
/* ── Model confidence panel ── */
.model-conf-card {
    background: #071428;
    border: 1px solid #1e3a6e;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 4px 0;
}
.conf-metric { text-align:center; }
.conf-metric-label { color:#9db4d8; font-size:0.68rem; text-transform:uppercase; letter-spacing:1px; }
.conf-metric-value { color:#FFD700; font-size:1.25rem; font-weight:800; }
.conf-metric-sub   { color:#9db4d8; font-size:0.68rem; }
.agree-badge {
    display:inline-block; border-radius:20px; padding:5px 16px;
    font-size:0.8rem; font-weight:700; letter-spacing:0.5px;
    background: rgba(40,167,69,0.15); border:1px solid rgba(40,167,69,0.4);
    color:#28a745;
}
.brier-badge {
    display:inline-block; border-radius:20px; padding:5px 16px;
    font-size:0.78rem; font-weight:600;
    background: rgba(255,215,0,0.08); border:1px solid rgba(255,215,0,0.3);
    color:#FFD700;
}
/* ── Why explanation panel ── */
.why-box {
    background: #071428;
    border: 1px solid #1e3a6e;
    border-left: 4px solid #4A90D9;
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 4px;
}
.why-bullet {
    color: #d0daf0;
    font-size: 0.88rem;
    line-height: 1.8;
    margin: 6px 0;
    padding-left: 8px;
}

/* ── Section dividers ── */
.section-divider {
    border: none;
    border-top: 1px solid #1e3a6e;
    margin: 28px 0 20px;
}

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: #0d1f3c;
    border: 1px solid #1e3a6e;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #9db4d8 !important; }
[data-testid="stMetricValue"] { color: #FFD700 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #050d1f;
    border-right: 1px solid #1e3a6e;
}
[data-testid="stSidebar"] .css-1d391kg { color: #e8eaf6; }

/* ── Tables ── */
.dataframe { background: #0a1628 !important; color: #e8eaf6 !important; }
.dataframe th { background: #1a3a6e !important; color: #FFD700 !important; }

/* hide default streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>

<!-- Animated background stars -->
<script>
(function() {
  const n = 80;
  for (let i = 0; i < n; i++) {
    const s = document.createElement("div");
    s.className = "star";
    const sz = Math.random() * 3 + 1;
    s.style.cssText = [
      `width:${sz}px`, `height:${sz}px`,
      `top:${Math.random()*100}vh`,
      `left:${Math.random()*100}vw`,
      `animation-duration:${Math.random()*4+2}s`,
      `animation-delay:${Math.random()*5}s`,
    ].join(";");
    document.body.appendChild(s);
  }
})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached heavy computations
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _ucl_logo_data_uri() -> str:
    """Load UCL logo from local assets and return a base64 data URI."""
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "ucl_logo.png")
    try:
        with open(logo_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def _get_ensemble_model():
    model = UCLEnsembleModel()
    model.train()
    return model


@st.cache_resource(show_spinner=False)
def _get_poisson_model():
    return PoissonGoalModel().fit()


@st.cache_resource(show_spinner=False)
def _run_simulation(semi_id: str, lambda_home: float, lambda_away: float):
    cfg = next(s for s in SEMIFINALS if s["id"] == semi_id)
    return simulate_semifinal(cfg, overrides={"lambda_home": lambda_home, "lambda_away": lambda_away})


@st.cache_data(show_spinner=False)
def _get_ai_analysis(home, away, fl_result, aggregate, mch, mca, lr_lead, leading):
    return generate_analysis(home, away, fl_result, aggregate, mch, mca, lr_lead, leading)


# ══════════════════════════════════════════════════════════════════════════════
# Helper renderers
# ══════════════════════════════════════════════════════════════════════════════

def _team_logo_html(team: str, size: int = 80) -> str:
    url = TEAMS[team]["crest_url"]
    return (
        f'<div style="text-align:center;">'
        f'<img src="{url}" width="{size}" height="{size}" '
        f'style="object-fit:contain; filter:drop-shadow(0 0 8px rgba(255,215,0,0.4));">'
        f'</div>'
    )


def _probability_bar(team: str, prob: float, color: str) -> str:
    pct = prob * 100
    return (
        f'<div style="margin:6px 0;">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#9db4d8;margin-bottom:3px;">'
        f'<span>{team}</span><span style="color:#FFD700;font-weight:700;">{pct:.1f}%</span>'
        f'</div>'
        f'<div class="prob-bar-container">'
        f'<div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f'</div>'
        f'</div>'
    )


def _score_heatmap(lambda_home: float, lambda_away: float,
                   home_team: str, away_team: str) -> go.Figure:
    """
    Build the second-leg score heatmap directly from Poisson lambdas.
    Generates its own 50k samples — no dependency on cached SimulationResult.
    """
    rng   = np.random.default_rng(42)
    n     = 50_000
    hg    = rng.poisson(max(lambda_home, 0.1), n)
    ag    = rng.poisson(max(lambda_away, 0.1), n)
    max_g = 8
    bins  = np.arange(max_g + 1) - 0.5                  # edges -0.5 … 7.5
    mat, _, _ = np.histogram2d(hg, ag, bins=bins)        # mat[home_g, away_g]
    z = mat.T / n * 100                                  # z[away_g, home_g]

    ticks = list(range(max_g))
    fig = go.Figure(go.Heatmap(
        z=z,
        x=ticks,
        y=ticks,
        colorscale=[
            [0.0, "#050d1f"],
            [0.1, "#0d2240"],
            [0.4, "#1a4a8a"],
            [0.7, "#2a7fff"],
            [1.0, "#FFD700"],
        ],
        showscale=True,
        colorbar=dict(title=dict(text="Prob %", font=dict(color="#9db4d8")), tickfont=dict(color="#9db4d8")),
        hovertemplate=f"{home_team} %{{x}} – %{{y}} {away_team}<br>Probability: %{{z:.2f}}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Second-Leg Score Probability (%)", font=dict(color="#FFD700", size=13)),
        xaxis=dict(title=dict(text=home_team, font=dict(color="#9db4d8")), tickfont=dict(color="#9db4d8"),
                   gridcolor="#1e3a6e", dtick=1),
        yaxis=dict(title=dict(text=away_team, font=dict(color="#9db4d8")), tickfont=dict(color="#9db4d8"),
                   gridcolor="#1e3a6e", dtick=1),
        paper_bgcolor="#050d1f",
        plot_bgcolor="#050d1f",
        margin=dict(l=60, r=20, t=40, b=60),
        height=340,
    )
    return fig


def _ucl_history_chart(team1: str, team2: str) -> go.Figure:
    """Grouped bar chart of UCL wins per season for both teams."""
    df1 = get_ucl_history_dataframe(team1)
    df2 = get_ucl_history_dataframe(team2)

    fig = go.Figure()
    if not df1.empty:
        fig.add_trace(go.Bar(
            name=team1, x=df1["season"], y=df1["ucl_wins"],
            marker_color=TEAMS[team1].get("chart_color", TEAMS[team1]["color"]),
            hovertemplate="%{x}<br>Wins: %{y}<extra></extra>",
        ))
    if not df2.empty:
        fig.add_trace(go.Bar(
            name=team2, x=df2["season"], y=df2["ucl_wins"],
            marker_color=TEAMS[team2].get("chart_color", TEAMS[team2]["color"]),
            hovertemplate="%{x}<br>Wins: %{y}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="UCL Wins per Season (Last 5 Years)", font=dict(color="#FFD700", size=13)),
        barmode="group",
        paper_bgcolor="#050d1f",
        plot_bgcolor="#050d1f",
        legend=dict(font=dict(color="#e8eaf6"), bgcolor="#0a1628"),
        xaxis=dict(tickfont=dict(color="#9db4d8"), gridcolor="#1e3a6e"),
        yaxis=dict(tickfont=dict(color="#9db4d8"), gridcolor="#1e3a6e",
                   title=dict(text="UCL Wins", font=dict(color="#9db4d8"))),
        margin=dict(l=40, r=20, t=40, b=60),
        height=280,
    )
    return fig


def _ucl_goals_chart(team1: str, team2: str) -> go.Figure:
    """Line chart of UCL goals scored per season for both teams."""
    df1 = get_ucl_history_dataframe(team1)
    df2 = get_ucl_history_dataframe(team2)

    fig = go.Figure()
    for df, team in [(df1, team1), (df2, team2)]:
        if not df.empty:
            c = TEAMS[team].get("chart_color", TEAMS[team]["color"])
            fig.add_trace(go.Scatter(
                name=team, x=df["season"],
                y=df["goals_scored"],
                mode="lines+markers",
                line=dict(color=c, width=2),
                marker=dict(size=7, color=c),
            ))
    fig.update_layout(
        title=dict(text="UCL Goals Scored per Season", font=dict(color="#FFD700", size=13)),
        paper_bgcolor="#050d1f",
        plot_bgcolor="#050d1f",
        legend=dict(font=dict(color="#e8eaf6"), bgcolor="#0a1628"),
        xaxis=dict(tickfont=dict(color="#9db4d8"), gridcolor="#1e3a6e"),
        yaxis=dict(tickfont=dict(color="#9db4d8"), gridcolor="#1e3a6e",
                   title=dict(text="Goals Scored", font=dict(color="#9db4d8"))),
        margin=dict(l=40, r=20, t=40, b=60),
        height=260,
    )
    return fig


def _advancement_donut(home_team: str, away_team: str,
                        prob_home: float, prob_away: float) -> go.Figure:
    """Donut chart showing advancement probabilities."""
    colors = [TEAMS[home_team]["color"], TEAMS[away_team]["color"]]
    fig = go.Figure(go.Pie(
        labels=[home_team, away_team],
        values=[prob_home * 100, prob_away * 100],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#050d1f", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#e8eaf6", size=11),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#050d1f",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
    )
    return fig


def _contribution_chart(contribs: list, leading_team: str, trailing_team: str, semi_id: str) -> go.Figure:
    """Horizontal bar chart of per-feature contributions to the ensemble prediction."""
    labels = [c[0] for c in contribs]
    values = [c[3] for c in contribs]          # raw signed contributions
    colors = ["#28a745" if v > 0 else "#dc3545" for v in values]
    hover = [
        f"{c[0]}<br>{'Favors ' + leading_team if c[2]=='favors_leading' else 'Favors ' + trailing_team}"
        f"<br>Magnitude: {c[1]:.3f}<extra></extra>"
        for c in contribs
    ]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        hovertemplate=hover,
    ))
    fig.add_vline(x=0, line_color="#9db4d8", line_width=1, line_dash="dot")
    fig.update_layout(
        title=dict(
            text=f"Feature Contributions  ·  green = {leading_team}  ·  red = {trailing_team}",
            font=dict(color="#9db4d8", size=11),
        ),
        paper_bgcolor="#050d1f", plot_bgcolor="#050d1f",
        xaxis=dict(tickfont=dict(color="#9db4d8"), gridcolor="#1e3a6e", zeroline=False),
        yaxis=dict(tickfont=dict(color="#e8eaf6"), gridcolor="#1e3a6e"),
        margin=dict(l=160, r=20, t=40, b=20),
        height=260,
    )
    return fig


def _brier_color(score: float) -> str:
    if score is None:
        return "#9db4d8"
    if score < 0.15:
        return "#28a745"
    if score < 0.20:
        return "#7ec850"
    if score < 0.25:
        return "#ffc107"
    return "#dc3545"


def _calibration_chart(cal_data: dict, semi_id: str) -> go.Figure:
    """Reliability diagram: mean predicted probability vs actual win rate per bin."""
    if not cal_data or not cal_data.get("mean_predicted"):
        return go.Figure()

    mean_p  = cal_data["mean_predicted"]
    actual  = cal_data["mean_actual"]
    counts  = cal_data["counts"]

    fig = go.Figure()
    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#1e3a6e", width=1, dash="dash"),
        name="Perfect calibration",
        hoverinfo="skip",
    ))
    # Actual calibration
    fig.add_trace(go.Scatter(
        x=mean_p, y=actual,
        mode="lines+markers",
        line=dict(color="#FFD700", width=2),
        marker=dict(
            size=[max(8, c * 2) for c in counts],
            color="#FFD700",
            line=dict(color="#050d1f", width=1),
        ),
        name="Ensemble (LOO)",
        hovertemplate=(
            "Predicted: %{x:.2f}<br>Actual: %{y:.2f}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(
            text="Calibration Reliability Diagram (LOO)",
            font=dict(color="#FFD700", size=13),
        ),
        xaxis=dict(
            title=dict(text="Mean predicted probability", font=dict(color="#9db4d8")),
            tickfont=dict(color="#9db4d8"),
            range=[0, 1],
            gridcolor="#1e3a6e",
        ),
        yaxis=dict(
            title=dict(text="Actual win rate", font=dict(color="#9db4d8")),
            tickfont=dict(color="#9db4d8"),
            range=[0, 1],
            gridcolor="#1e3a6e",
        ),
        paper_bgcolor="#050d1f",
        plot_bgcolor="#050d1f",
        legend=dict(font=dict(color="#e8eaf6"), bgcolor="#0a1628"),
        margin=dict(l=60, r=20, t=40, b=60),
        height=300,
    )
    return fig


def _league_form_html(team: str) -> str:
    form = get_league_form(team)
    if not form:
        return "<p>No league data available.</p>"
    last5_html = ""
    color_map = {"W": "#28a745", "D": "#ffc107", "L": "#dc3545"}
    for r in form.get("last_5", []):
        bg = color_map.get(r, "#666")
        last5_html += (
            f'<span style="background:{bg};color:white;border-radius:4px;'
            f'padding:3px 8px;margin:2px;font-size:0.78rem;font-weight:700;">{r}</span>'
        )

    return (
        f'<div style="font-size:0.85rem;line-height:1.9;">'
        f'<b style="color:#FFD700;">{form["league"]}</b> — '
        f'<b style="color:#e8eaf6;">Position {form["position"]}</b><br>'
        f'<span class="stat-pill">P <strong>{form["played"]}</strong></span>'
        f'<span class="stat-pill">W <strong>{form["won"]}</strong></span>'
        f'<span class="stat-pill">D <strong>{form["drawn"]}</strong></span>'
        f'<span class="stat-pill">L <strong>{form["lost"]}</strong></span>'
        f'<span class="stat-pill">GF <strong>{form["gf"]}</strong></span>'
        f'<span class="stat-pill">GA <strong>{form["ga"]}</strong></span>'
        f'<span class="stat-pill">Pts <strong>{form["points"]}</strong></span><br>'
        f'Last 5: {last5_html}'
        f'</div>'
    )


def _h2h_html(team_a: str, team_b: str) -> str:
    key = f"{team_a}_vs_{team_b}"
    records = H2H_RECORDS.get(key, H2H_RECORDS.get(f"{team_b}_vs_{team_a}", []))
    if not records:
        return "<p style='color:#9db4d8;font-size:0.85rem;'>No head-to-head records found.</p>"

    rows_html = ""
    color_map = {"W": "#28a745", "D": "#ffc107", "L": "#dc3545"}
    for r in records:
        bg = color_map.get(r.get("result", "D"), "#666")
        badge = f'<span style="background:{bg};color:white;border-radius:4px;padding:2px 8px;font-size:0.75rem;font-weight:700;">{r.get("result","?")}</span>'
        rows_html += (
            f'<tr style="border-bottom:1px solid #1e3a6e;">'
            f'<td style="padding:5px 8px;color:#9db4d8;">{r.get("season","")}</td>'
            f'<td style="padding:5px 8px;color:#e8eaf6;">{r.get("competition","")}</td>'
            f'<td style="padding:5px 8px;color:#9db4d8;">{r.get("venue","")}</td>'
            f'<td style="padding:5px 8px;text-align:center;font-weight:700;color:#FFD700;">'
            f'{r.get("team_a_goals","?")} – {r.get("team_b_goals","?")}</td>'
            f'<td style="padding:5px 8px;text-align:center;">{badge}</td>'
            f'</tr>'
        )
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">'
        f'<thead><tr style="background:#0d2240;">'
        f'<th style="padding:6px 8px;color:#FFD700;text-align:left;">Season</th>'
        f'<th style="padding:6px 8px;color:#FFD700;text-align:left;">Competition</th>'
        f'<th style="padding:6px 8px;color:#FFD700;text-align:left;">Venue</th>'
        f'<th style="padding:6px 8px;color:#FFD700;text-align:center;">Score</th>'
        f'<th style="padding:6px 8px;color:#FFD700;text-align:center;">Result ({team_a})</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


def _md_inline(text: str) -> str:
    """Convert inline markdown bold/italic to HTML."""
    import re
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#e8eaf6;">\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',   r'<em style="color:#c0cfe8;">\1</em>', text)
    return text


def _format_ai_analysis(raw: str) -> str:
    """Convert Gemini markdown to fully styled HTML for the ai-box container."""
    import re
    lines    = raw.split('\n')
    out      = []
    in_list  = False

    for line in lines:
        line = line.rstrip()
        if line.startswith('## '):
            if in_list:
                out.append('</ul>'); in_list = False
            out.append(f'<div class="ai-h2">{_md_inline(line[3:])}</div>')
        elif line.startswith('### '):
            if in_list:
                out.append('</ul>'); in_list = False
            out.append(f'<div class="ai-h3">{_md_inline(line[4:])}</div>')
        elif re.match(r'^[•\-\*] ', line):
            if not in_list:
                out.append('<ul class="ai-list">'); in_list = True
            bullet_text = re.sub(r'^[•\-\*] ', '', line)
            out.append(f'<li>{_md_inline(bullet_text)}</li>')
        elif line == '':
            if in_list:
                out.append('</ul>'); in_list = False
        else:
            if in_list:
                out.append('</ul>'); in_list = False
            # Style VERDICT lines specially
            if line.upper().startswith('**VERDICT') or '**VERDICT' in line.upper():
                out.append(f'<div class="ai-verdict">{_md_inline(line)}</div>')
            elif line:
                out.append(f'<p class="ai-p">{_md_inline(line)}</p>')

    if in_list:
        out.append('</ul>')
    return '\n'.join(out)


run_ai = True  # AI analysis always on; requires GEMINI_API_KEY in .env


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

_logo_uri = _ucl_logo_data_uri()
_logo_html = (
    f'<img src="{_logo_uri}" width="110" '
    f'style="margin-bottom:8px;filter:brightness(0) invert(1) drop-shadow(0 0 14px rgba(255,215,0,0.5));">'
    if _logo_uri else
    # SVG fallback if network unavailable
    '<div style="display:inline-flex;align-items:center;justify-content:center;'
    'width:90px;height:90px;border-radius:50%;margin-bottom:10px;'
    'background:radial-gradient(circle at 35% 35%,#1a3a6e,#050d1f);'
    'border:2px solid #FFD700;box-shadow:0 0 28px rgba(255,215,0,0.45);">'
    '<svg width="56" height="56" viewBox="0 0 56 56" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="28" cy="28" r="26" fill="none" stroke="#FFD700" stroke-width="1.5" opacity="0.4"/>'
    '<polygon points="28,4 31,22 46,14 35,26 52,28 35,30 46,42 31,34 28,52 25,34 10,42 21,30 4,28 21,26 10,14 25,22"'
    ' fill="#FFD700" opacity="0.9"/>'
    '<circle cx="28" cy="28" r="6" fill="#0a1628" stroke="#FFD700" stroke-width="1.2"/>'
    '</svg></div>'
)

st.markdown(
    f"""
    <div class="ucl-header">
        {_logo_html}
        <h1>UEFA CHAMPIONS LEAGUE 2025–26</h1>
        <p style="font-size:0.85rem;color:#FFD700;letter-spacing:3px;
                  text-transform:uppercase;margin-top:4px;">
            Semi-Final &nbsp;·&nbsp; Second Leg Predictor
        </p>
        <p style="color:#7a96b8;font-size:0.78rem;margin-top:6px;">
            Logistic Regression &nbsp;+&nbsp; Monte Carlo Simulation
            &nbsp;+&nbsp; Gemini AI Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Build predictions for both semis
# ══════════════════════════════════════════════════════════════════════════════

ensemble      = _get_ensemble_model()
poisson_model = _get_poisson_model()

predictions = []
for semi in SEMIFINALS:
    mf = build_features(
        home_team             = semi["second_leg_home"],
        away_team             = semi["second_leg_away"],
        first_leg_home_goals  = semi["first_leg_home_goals"],
        first_leg_away_goals  = semi["first_leg_away_goals"],
    )
    lr_pred = ensemble.predict_from_matchup(mf)
    # Use Poisson-model λ values (Dixon-Coles attack/defense ratings)
    lh, la  = poisson_model.lambda_pair(semi["second_leg_home"], semi["second_leg_away"])
    sim     = _run_simulation(semi["id"], lh, la)
    predictions.append({"semi": semi, "mf": mf, "lr": lr_pred, "sim": sim,
                         "lambda_home": lh, "lambda_away": la})


# ══════════════════════════════════════════════════════════════════════════════
# Predicted Finalists — shown at the top before detailed breakdowns
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div style="text-align:center;color:#FFD700;font-size:1.45rem;font-weight:800;'
    'letter-spacing:3px;text-transform:uppercase;margin:28px 0 16px;">'
    '🏆 &nbsp; Predicted Finalists &nbsp; 🏆</div>',
    unsafe_allow_html=True,
)

fin_cols = st.columns(2)
for col, pred in zip(fin_cols, predictions):
    sim_p  = pred["sim"]
    semi_p = pred["semi"]
    lr_p   = pred["lr"]
    if sim_p.prob_home_advances >= sim_p.prob_away_advances:
        finalist    = semi_p["second_leg_home"]
        mc_prob     = sim_p.prob_home_advances
        eliminated  = semi_p["second_leg_away"]
    else:
        finalist    = semi_p["second_leg_away"]
        mc_prob     = sim_p.prob_away_advances
        eliminated  = semi_p["second_leg_home"]

    fc = TEAMS[finalist]["color"]
    ec = TEAMS[eliminated]["color"]
    with col:
        st.markdown(
            f'<div class="finalist-card" style="border:2px solid {fc};">'
            # logos
            f'<div style="display:flex;justify-content:center;align-items:center;gap:20px;margin-bottom:14px;">'
            f'<div style="text-align:center;">'
            + _team_logo_html(finalist, 80)
            + f'<div style="font-size:0.95rem;font-weight:800;color:{fc};margin-top:6px;">{finalist}</div>'
            f'<div style="font-size:0.72rem;color:#28a745;margin-top:2px;">✅ ADVANCES</div>'
            f'</div>'
            f'<div style="color:#1e3a6e;font-size:1.6rem;font-weight:900;">vs</div>'
            f'<div style="text-align:center;opacity:0.5;">'
            + _team_logo_html(eliminated, 60)
            + f'<div style="font-size:0.82rem;font-weight:600;color:{ec};margin-top:6px;">{eliminated}</div>'
            f'<div style="font-size:0.72rem;color:#dc3545;margin-top:2px;">❌ OUT</div>'
            f'</div>'
            f'</div>'
            # stats
            f'<div style="display:flex;justify-content:center;gap:20px;flex-wrap:wrap;">'
            f'<div class="conf-metric">'
            f'<div class="conf-metric-label">Monte Carlo</div>'
            f'<div class="conf-metric-value">{mc_prob*100:.1f}%</div>'
            f'</div>'
            f'<div class="conf-metric">'
            f'<div class="conf-metric-label">Ensemble</div>'
            f'<div class="conf-metric-value">{lr_p["prob_leading_advances"]*100:.1f}%</div>'
            f'</div>'
            f'<div class="conf-metric">'
            f'<div class="conf-metric-label">Agreement</div>'
            f'<div class="conf-metric-value" style="font-size:0.9rem;">{lr_p["agreement"]}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Semi-Final Sections
# ══════════════════════════════════════════════════════════════════════════════

TABS = st.tabs(["🔴 Semi-Final 1: Arsenal vs Atlético", "🔵 Semi-Final 2: Bayern vs PSG"])

for tab, pred in zip(TABS, predictions):
    with tab:
        semi   = pred["semi"]
        mf     = pred["mf"]
        lr     = pred["lr"]
        sim    = pred["sim"]

        home   = semi["second_leg_home"]
        away   = semi["second_leg_away"]
        h_col  = TEAMS[home]["color"]
        a_col  = TEAMS[away]["color"]

        fl_h_goals = semi["first_leg_home_goals"]
        fl_a_goals = semi["first_leg_away_goals"]
        # First-leg home team = second-leg away team
        fl_home_team = away
        fl_away_team = home
        first_leg_str  = f"{fl_home_team} {fl_h_goals}–{fl_a_goals} {fl_away_team}"
        # Aggregate: second-leg home team has fl_a_goals, away team has fl_h_goals
        agg_home_leg1  = fl_a_goals
        agg_away_leg1  = fl_h_goals
        if agg_home_leg1 > agg_away_leg1:
            agg_str = f"{home} leads {agg_home_leg1}–{agg_away_leg1}"
        elif agg_away_leg1 > agg_home_leg1:
            agg_str = f"{away} leads {agg_away_leg1}–{agg_home_leg1}"
        else:
            agg_str = f"Tied {agg_home_leg1}–{agg_away_leg1}"

        # ── Team header row ────────────────────────────────────────────────────
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            st.markdown(_team_logo_html(home, 100), unsafe_allow_html=True)
            st.markdown(
                f'<div class="team-name" style="color:{h_col};">{home}</div>'
                f'<div class="team-league">{TEAMS[home]["league"]} · {TEAMS[home]["country"]}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div style="text-align:center;padding-top:20px;">'
                f'<div class="score-badge">{fl_h_goals} – {fl_a_goals}</div>'
                f'<div class="score-label">First Leg</div>'
                f'<div style="font-size:0.78rem;color:#9db4d8;margin-top:8px;">{fl_home_team} (home)</div>'
                f'<div style="font-size:0.72rem;color:#FFD700;margin-top:4px;">2nd Leg at {home}</div>'
                f'<div style="font-size:0.8rem;color:#9db4d8;margin-top:6px;">{agg_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(_team_logo_html(away, 100), unsafe_allow_html=True)
            st.markdown(
                f'<div class="team-name" style="color:{a_col};">{away}</div>'
                f'<div class="team-league">{TEAMS[away]["league"]} · {TEAMS[away]["country"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Prediction columns ────────────────────────────────────────────────
        left, right = st.columns([3, 2])

        with left:
            st.markdown("#### 📊 Advancement Probabilities")

            prob_home = sim.prob_home_advances
            prob_away = sim.prob_away_advances

            st.markdown(
                _probability_bar(f"{home} (home)", prob_home, h_col) +
                _probability_bar(f"{away} (away)", prob_away, a_col),
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div style="margin:12px 0;font-size:0.82rem;color:#9db4d8;">'
                f'⏱ Decided in 90 min: <b style="color:#e8eaf6;">{sim.pct_normal_time*100:.1f}%</b> &nbsp;|&nbsp; '
                f'Extra Time: <b style="color:#e8eaf6;">{sim.pct_extra_time*100:.1f}%</b> &nbsp;|&nbsp; '
                f'Pens: <b style="color:#e8eaf6;">{sim.pct_penalties*100:.1f}%</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Three-model comparison ────────────────────────────────────────
            st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(
                f'<div class="conf-metric">'
                f'<div class="conf-metric-label">LR Model</div>'
                f'<div class="conf-metric-value">{lr["lr_prob"]*100:.1f}%</div>'
                f'<div class="conf-metric-sub">{lr["leading_team"]}</div></div>',
                unsafe_allow_html=True,
            )
            mc2.markdown(
                f'<div class="conf-metric">'
                f'<div class="conf-metric-label">XGBoost</div>'
                f'<div class="conf-metric-value">{lr["xgb_prob"]*100:.1f}%</div>'
                f'<div class="conf-metric-sub">{lr["leading_team"]}</div></div>',
                unsafe_allow_html=True,
            )
            mc3.markdown(
                f'<div class="conf-metric">'
                f'<div class="conf-metric-label">Monte Carlo</div>'
                f'<div class="conf-metric-value">'
                f'{(sim.prob_home_advances if sim.prob_home_advances > 0.5 else sim.prob_away_advances)*100:.1f}%</div>'
                f'<div class="conf-metric-sub">leading team</div></div>',
                unsafe_allow_html=True,
            )
            brier_val  = lr.get("brier")
            brier_col  = _brier_color(brier_val)
            brier_text = f"{brier_val:.3f}" if brier_val is not None else "N/A"
            mc4.markdown(
                f'<div class="conf-metric">'
                f'<div class="conf-metric-label">Brier Score</div>'
                f'<div class="conf-metric-value" style="color:{brier_col};">{brier_text}</div>'
                f'<div class="conf-metric-sub">lower = better</div></div>',
                unsafe_allow_html=True,
            )
            # Agreement + Brier row
            st.markdown(
                f'<div style="margin-top:10px;">'
                f'<span class="agree-badge">Agreement: {lr["agreement"]}</span>'
                f'&nbsp;&nbsp;<span style="font-size:0.72rem;color:#9db4d8;">'
                f'CV accuracy: {lr["cv_accuracy"]*100:.0f}% &nbsp;·&nbsp; '
                f'LR Brier: {lr.get("brier_lr","N/A")} &nbsp;·&nbsp; '
                f'XGB Brier: {lr.get("brier_xgb","N/A")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Ensemble weighting explanation ────────────────────────────────
            weights = lr.get("model_weights", {})
            w_lr  = weights.get("lr",  0.5)
            w_xgb = weights.get("xgb", 0.5)
            b_lr  = lr.get("brier_lr")
            b_xgb = lr.get("brier_xgb")
            brier_note = ""
            if b_lr is not None and b_xgb is not None:
                better = "LR" if b_lr < b_xgb else "XGBoost"
                brier_note = (
                    f" — <b style='color:#FFD700;'>{better}</b> calibrated better "
                    f"(Brier LR={b_lr:.3f} vs XGB={b_xgb:.3f})"
                )
            st.markdown(
                f'<div style="background:#071428;border:1px solid #1e3a6e;border-left:4px solid #FFD700;'
                f'border-radius:10px;padding:12px 18px;margin:10px 0;font-size:0.82rem;">'
                f'<b style="color:#FFD700;">⚖️ Ensemble Weighting (inverse-Brier optimal)</b>'
                f'<div style="margin-top:8px;display:flex;gap:24px;align-items:center;">'
                f'<div style="text-align:center;">'
                f'<div style="color:#9db4d8;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">LR Weight</div>'
                f'<div style="color:#e8eaf6;font-size:1.2rem;font-weight:800;">{w_lr*100:.0f}%</div>'
                f'</div>'
                f'<div style="color:#1e3a6e;font-size:1.3rem;">+</div>'
                f'<div style="text-align:center;">'
                f'<div style="color:#9db4d8;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">XGBoost Weight</div>'
                f'<div style="color:#e8eaf6;font-size:1.2rem;font-weight:800;">{w_xgb*100:.0f}%</div>'
                f'</div>'
                f'</div>'
                f'<div style="color:#9db4d8;margin-top:6px;font-size:0.75rem;">'
                f'Weight = 1/Brier ÷ Σ(1/Brier). Lower Brier score → higher weight{brier_note}.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Poisson expected goals
            st.markdown(
                f'<div style="font-size:0.82rem;color:#9db4d8;margin-top:6px;">'
                f'Poisson λ (2nd leg) → {home}: <b style="color:#FFD700;">{pred["lambda_home"]:.3f}</b> &nbsp;|&nbsp; '
                f'{away}: <b style="color:#FFD700;">{pred["lambda_away"]:.3f}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Top 5 most-likely second-leg scorelines
            st.markdown("**Most likely second-leg scorelines:**")
            top5_rows = []
            for (hg, ag), cnt in sim.top_scorelines[:5]:
                pct = cnt / sim.n_sims * 100
                top5_rows.append({
                    "Scoreline": f"{home} {hg}–{ag} {away}",
                    "Probability": f"{pct:.2f}%",
                })
            st.dataframe(pd.DataFrame(top5_rows), hide_index=True, use_container_width=True)

        with right:
            st.plotly_chart(
                _advancement_donut(home, away, prob_home, prob_away),
                use_container_width=True,
                key=f"donut_{semi['id']}",
            )
            st.markdown(
                f'<div style="text-align:center;font-size:0.78rem;color:#9db4d8;margin-top:-10px;">'
                f'Monte Carlo n = {sim.n_sims:,} simulations</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Why model favors X ────────────────────────────────────────────────
        st.markdown("#### 🧠 Why the Model Favors This Outcome")
        why_col, feat_col = st.columns([2, 3])
        with why_col:
            bullets_html = "".join(
                f'<div class="why-bullet">• {_md_inline(b)}</div>'
                for b in lr.get("explanation_bullets", [])
            )
            st.markdown(
                f'<div class="why-box">'
                f'<div style="color:#4A90D9;font-weight:700;font-size:0.88rem;'
                f'margin-bottom:10px;text-transform:uppercase;letter-spacing:1px;">'
                f'Prediction: {lr["leading_team"]} advances</div>'
                + bullets_html +
                f'</div>',
                unsafe_allow_html=True,
            )
        with feat_col:
            st.plotly_chart(
                _contribution_chart(
                    lr.get("feature_contributions", []),
                    lr["leading_team"],
                    lr["trailing_team"],
                    semi["id"],
                ),
                use_container_width=True,
                key=f"contrib_{semi['id']}",
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Score heatmap ─────────────────────────────────────────────────────
        st.markdown("#### 🔥 Second-Leg Score Probability Heatmap")
        st.plotly_chart(
            _score_heatmap(pred["lambda_home"], pred["lambda_away"], home, away),
            use_container_width=True,
            key=f"heatmap_{semi['id']}",
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Model calibration ─────────────────────────────────────────────────
        st.markdown("#### 📐 Ensemble Calibration (Reliability Diagram)")
        cal_col, cal_info_col = st.columns([2, 1])
        with cal_col:
            cal_data = ensemble.calibration_data()
            cal_fig  = _calibration_chart(cal_data, semi["id"])
            if cal_fig.data:
                st.plotly_chart(cal_fig, use_container_width=True,
                                key=f"cal_{semi['id']}")
            else:
                st.info("Calibration data not yet available.")
        with cal_info_col:
            n_s = cal_data.get("n_samples", 0)
            brier_ens = lr.get("brier")
            brier_col = _brier_color(brier_ens)
            brier_txt = f"{brier_ens:.3f}" if brier_ens is not None else "N/A"
            st.markdown(
                f'<div style="background:#071428;border:1px solid #1e3a6e;border-radius:10px;'
                f'padding:16px 18px;font-size:0.82rem;line-height:2;">'
                f'<b style="color:#FFD700;">Calibration Guide</b><br>'
                f'<span style="color:#9db4d8;">Points on the dashed line = perfectly calibrated.</span><br>'
                f'<span style="color:#9db4d8;">Above line → model is overconfident.</span><br>'
                f'<span style="color:#9db4d8;">Below line → model is underconfident.</span><br>'
                f'<span style="color:#9db4d8;">Marker size ∝ sample count per bin.</span><br><br>'
                f'<b style="color:#9db4d8;">Training samples: </b>'
                f'<b style="color:#FFD700;">{n_s}</b><br>'
                f'<b style="color:#9db4d8;">Ensemble Brier: </b>'
                f'<b style="color:{brier_col};">{brier_txt}</b>'
                f'<span style="color:#9db4d8;font-size:0.7rem;"> (random = 0.25)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── Poisson strength table ─────────────────────────────────────────────
        with st.expander("⚡ Poisson Attack / Defense Ratings (all teams)"):
            strength_df = pd.DataFrame(poisson_model.strength_table())
            highlight = [home, away]
            st.dataframe(
                strength_df.style.apply(
                    lambda row: [
                        "background-color:#0d2240;font-weight:bold;" if row["Team"] in highlight else ""
                        for _ in row
                    ],
                    axis=1,
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Attack rating > 1.0 = scores more than UCL average · "
                "Defense rating < 1.0 = concedes less than average · "
                "Based on last 3 seasons (linear ramp weighting)."
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── UCL History charts ────────────────────────────────────────────────
        st.markdown("#### 🏆 UCL History — Last 5 Seasons")
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(_ucl_history_chart(home, away), use_container_width=True,
                            key=f"ucl_wins_{semi['id']}")
        with ch2:
            st.plotly_chart(_ucl_goals_chart(home, away), use_container_width=True,
                            key=f"ucl_goals_{semi['id']}")

        # UCL stats table
        df_h = get_ucl_history_dataframe(home).assign(Team=home)
        df_a = get_ucl_history_dataframe(away).assign(Team=away)
        df_ucl = pd.concat([df_h, df_a], ignore_index=True)[
            ["Team", "season", "stage", "ucl_wins", "ucl_draws", "ucl_losses",
             "goals_scored", "goals_conceded"]
        ].rename(columns={
            "season": "Season", "stage": "Best Stage",
            "ucl_wins": "W", "ucl_draws": "D", "ucl_losses": "L",
            "goals_scored": "GF", "goals_conceded": "GA",
        })
        with st.expander("📋 Full UCL Record Table"):
            st.dataframe(df_ucl, hide_index=True, use_container_width=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── League form ───────────────────────────────────────────────────────
        st.markdown("#### 🏟️ Domestic League Form (2024-25)")
        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown(
                f'<div style="background:#0a1628;border:1px solid #1e3a6e;border-radius:10px;padding:16px;">'
                f'<div style="font-size:1rem;font-weight:700;color:{h_col};margin-bottom:10px;">{home}</div>'
                + _league_form_html(home) +
                f'</div>',
                unsafe_allow_html=True,
            )
        with lc2:
            st.markdown(
                f'<div style="background:#0a1628;border:1px solid #1e3a6e;border-radius:10px;padding:16px;">'
                f'<div style="font-size:1rem;font-weight:700;color:{a_col};margin-bottom:10px;">{away}</div>'
                + _league_form_html(away) +
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── H2H record ────────────────────────────────────────────────────────
        st.markdown("#### ⚔️ Head-to-Head Record")
        h2h_stats = compute_h2h_stats(f"{home}_vs_{away}")
        if h2h_stats["played"] == 0:
            h2h_stats = compute_h2h_stats(f"{away}_vs_{home}")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric(f"{home} Wins", h2h_stats["wins"])
        hc2.metric("Draws", h2h_stats["draws"])
        hc3.metric(f"{away} Wins", h2h_stats["losses"])
        st.markdown(_h2h_html(home, away), unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── AI Analysis ───────────────────────────────────────────────────────
        st.markdown("#### 🤖 AI Tactical Analysis (Gemini)")
        if run_ai:
            with st.spinner("Generating AI analysis..."):
                analysis = _get_ai_analysis(
                    home, away,
                    first_leg_str,
                    agg_str,
                    sim.prob_home_advances,
                    sim.prob_away_advances,
                    lr["prob_leading_advances"],
                    lr["leading_team"],
                )
            st.markdown(
                f'<div class="ai-box">{_format_ai_analysis(analysis)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Toggle 'Generate AI Analysis' in the sidebar to enable Gemini reasoning.")

        # ── LR feature importances ────────────────────────────────────────────
        with st.expander("🔬 Model Feature Coefficients (LR)"):
            coefs = ensemble.feature_coefficients
            if coefs:
                df_coef = pd.DataFrame(
                    {"Feature": list(coefs.keys()), "Coefficient": list(coefs.values())}
                ).sort_values("Coefficient", ascending=False)
                fig_coef = px.bar(
                    df_coef, x="Coefficient", y="Feature", orientation="h",
                    color="Coefficient",
                    color_continuous_scale=["#dc3545", "#0a1628", "#28a745"],
                    template="plotly_dark",
                    title="LR Feature Importances",
                )
                fig_coef.update_layout(
                    paper_bgcolor="#050d1f", plot_bgcolor="#050d1f",
                    title_font_color="#FFD700", height=280,
                    margin=dict(l=120, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_coef, use_container_width=True,
                                key=f"lr_coef_{semi['id']}")


st.markdown(
    '<div style="text-align:center;color:#1e3a6e;font-size:0.72rem;margin-top:32px;">'
    'UCL 2025-26 Semifinal Predictor · Built with Streamlit, scikit-learn &amp; Google Gemini · '
    'Statistical model — not financial or betting advice.'
    '</div>',
    unsafe_allow_html=True,
)
