---
title: UCL 2025-26 Semifinal Predictor
emoji: ⚽
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: true
---
<p align="center">
  <a href="https://huggingface.co/spaces/Sarthak-45/UCL-SemiFinal">
    <img src="https://img.shields.io/badge/⚽%20LIVE%20DEMO-UCL%20AI%20PREDICTOR-blue?style=for-the-badge" />
  </a>
</p>

# UCL 2025-26 Semifinal Predictor

A data-driven Python application that predicts which clubs advance from the
2025-26 UEFA Champions League semi-final second legs using three complementary
techniques:

| Layer | What it does |
|---|---|
| **Logistic Regression** | Classifies advancement probability from engineered features (H2H, form, aggregate diff, league strength) |
| **Monte Carlo Simulation** | Draws 100 000 Poisson-sampled second-leg scores and tallies advancement, extra-time, and penalty outcomes |
| **Gemini AI Analysis** | Calls Google Gemini 1.5 Flash to generate a structured tactical breakdown grounded in the statistical inputs |

---

## Fixtures

| Semi-Final | First Leg | Situation |
|---|---|---|
| Arsenal vs Atlético Madrid | 1 – 1 | Tied on aggregate, 2nd leg at Atlético |
| Bayern Munich vs PSG | 4 – 5 (PSG win) | PSG lead 5–4, 2nd leg at PSG |

---

## Project Structure

```
ucl-semifinal-predictor-2026/
│
├── app.py                          ← Streamlit dashboard (entry point)
├── config.py                       ← Central config: fixtures, teams, API keys
├── requirements.txt
├── .env.example                    ← Copy to .env and fill in your keys
│
├── src/
│   ├── data_ingestion/
│   │   ├── historical_data.py      ← Embedded H2H, UCL history, league form
│   │   └── football_api.py         ← Live data from football-data.org (optional)
│   │
│   ├── feature_engineering/
│   │   └── features.py             ← Builds the numerical feature vector
│   │
│   ├── models/
│   │   ├── logistic_regression.py  ← sklearn LR classifier + LOO cross-val
│   │   └── monte_carlo.py          ← Poisson Monte Carlo engine
│   │
│   └── ai_feedback/
│       └── reasoning.py            ← Gemini prompt builder + fallback analyser
│
└── data/
    ├── raw/                        ← API response cache (auto-populated)
    └── processed/                  ← Cleaned datasets (auto-populated)
```

---

## Quick Start

### 1 · Clone / open the project

```bash
cd "UCL SEMI LEG 2"
```

### 2 · Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Configure API keys

Copy the example env file and fill in your keys:

```bash
copy .env.example .env      # Windows
# or
cp .env.example .env        # macOS / Linux
```

Then open `.env` and set:

```env
GEMINI_API_KEY=AIza...          # https://aistudio.google.com/app/apikey  (free)
FOOTBALL_DATA_API_KEY=abc...    # https://www.football-data.org/          (free tier)
```

> **Both keys are optional.**  
> Without `GEMINI_API_KEY` the app uses the rule-based fallback analysis.  
> Without `FOOTBALL_DATA_API_KEY` the app uses the embedded historical data.

### 5 · Run the dashboard

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## API Keys — How to Get Them

### Google Gemini (AI Analysis)
1. Go to <https://aistudio.google.com/app/apikey>
2. Sign in with your Google account
3. Click **Create API key** — it's free with a generous quota
4. Paste the key into `.env` as `GEMINI_API_KEY`

### football-data.org (Live Standings)
1. Go to <https://www.football-data.org/client/register>
2. Register a free account (no credit card)
3. Copy the API token from your dashboard
4. Paste it into `.env` as `FOOTBALL_DATA_API_KEY`

---

## Customising the Simulation

### Via the Streamlit sidebar
Use the λ (expected-goals) sliders in the sidebar to adjust each team's
attacking threat for the second leg and instantly re-run 100 000 simulations.

### Via `config.py`
| Setting | Variable | Default |
|---|---|---|
| Simulations per run | `N_SIMULATIONS` | 100 000 |
| Random seed | `RANDOM_SEED` | 42 |
| Expected goals (home) — Semi 1 | `EXPECTED_GOALS["semi1"]["Atletico Madrid"]` | 1.25 |
| Expected goals (away) — Semi 1 | `EXPECTED_GOALS["semi1"]["Arsenal"]` | 1.05 |
| Expected goals (home) — Semi 2 | `EXPECTED_GOALS["semi2"]["PSG"]` | 1.35 |
| Expected goals (away) — Semi 2 | `EXPECTED_GOALS["semi2"]["Bayern Munich"]` | 2.15 |

---

## How Each Model Works

### Logistic Regression
Trained on 18 historical UCL semi-final results (2015-16 → 2023-24) with 6 features:

| Feature | Description |
|---|---|
| `agg_diff` | First-leg aggregate goal difference |
| `home_xg` / `away_xg` | Expected goals per game (UCL season avg) |
| `form_diff` | UCL form score gap (leading team − trailing team) |
| `h2h_win_rate` | Leading team's historical win rate vs opponent |
| `league_strength_diff` | Domestic league tier gap |

Leave-one-out cross-validation accuracy is displayed on the dashboard.

### Monte Carlo
For each of 100 000 trials:
1. Sample second-leg home goals from `Poisson(λ_home)`
2. Sample second-leg away goals from `Poisson(λ_away)`
3. Add to first-leg aggregate
4. If tied after 90 min → simulate 30 min extra time (`Poisson(0.33 × λ)`)
5. If still tied → penalty shoot-out (50/50 ± optional form edge)
6. Record outcome and scoreline

> **No away-goals rule** — UEFA abolished it from 2021-22 onwards.

### Gemini AI Analysis
A structured prompt is sent to `gemini-1.5-flash` containing both teams'
UCL history, H2H record, domestic league form, and the quantitative model
outputs.  Gemini returns a **Tactical Overview**, key factors for each side,
a predicted scoreline, and a one-sentence verdict.

---

## Tech Stack

| Package | Purpose |
|---|---|
| `streamlit` | Interactive dashboard |
| `plotly` | Score heatmaps, bar/line/donut charts |
| `scikit-learn` | Logistic Regression, StandardScaler, cross-val |
| `numpy` / `scipy` | Poisson sampling, statistics |
| `pandas` | Data manipulation |
| `google-generativeai` | Gemini AI API client |
| `requests` | football-data.org REST calls |
| `python-dotenv` | `.env` key loading |

---

## Disclaimer

This project is for educational and entertainment purposes only.  
All historical statistics are embedded approximations based on public records.  
Predictions are probabilistic and not financial or betting advice.
