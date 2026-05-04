"""
Logistic Regression classifier for UCL semifinal advancement.

Training data  : ~18 historical UCL semifinal results (2015-16 → 2023-24)
                 embedded in historical_data.py
Target         : 1 = leading team advanced, 0 = trailing team came back
Output         : probability that the currently-leading team advances
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from src.data_ingestion.historical_data import get_training_dataframe
from src.feature_engineering.features import MatchupFeatures


FEATURE_COLS = [
    "agg_diff",
    "home_xg",
    "away_xg",
    "form_diff",
    "h2h",
    "ls_diff",
]


class UCLLogisticRegression:
    """Wrapper around sklearn LogisticRegression with a standardisation scaler."""

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(C=C, max_iter=1000,
                                          random_state=random_state,
                                          solver="lbfgs")),
        ])
        self._trained = False
        self._cv_scores = None  # set to np.ndarray after train()

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self) -> "UCLLogisticRegression":
        """Fit the pipeline and compute 5-fold cross-validation accuracy."""
        df = get_training_dataframe()
        X  = df[FEATURE_COLS].values.astype(float)
        y  = df["advanced"].values.astype(int)

        self._pipeline.fit(X, y)
        self._trained = True

        self._cv_scores = cross_val_score(
            self._pipeline, X, y, cv=min(5, len(y)), scoring="accuracy"
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Return (prob_leading_advances, prob_trailing_advances).
        features must be a 1-D array of shape (6,).
        """
        if not self._trained:
            self.train()
        x = features.reshape(1, -1)
        probs = self._pipeline.predict_proba(x)[0]
        classes = list(self._pipeline.named_steps["lr"].classes_)
        # Safely find the index of class label 1 (= leading team advances)
        idx_leading = 0
        for i, c in enumerate(classes):
            if int(c) == 1:
                idx_leading = i
                break
        idx_trailing = 1 - idx_leading
        return float(probs[idx_leading]), float(probs[idx_trailing])

    def predict_from_matchup(self, mf: MatchupFeatures) -> dict:
        """
        Accepts a MatchupFeatures object and returns a prediction dict.
        """
        proba_lead, proba_trail = self.predict_proba(mf.to_array())

        # Compute cv_accuracy into a plain Python float before building the dict
        # to avoid any numpy truth-value ambiguity inside a dict literal.
        cv_scores = self._cv_scores
        if cv_scores is None or len(cv_scores) == 0:
            cv_acc = 0.0
        else:
            cv_acc = round(float(cv_scores.mean()), 3)

        return {
            "leading_team":           mf.leading_team,
            "trailing_team":          mf.trailing_team,
            "prob_leading_advances":  round(proba_lead,  4),
            "prob_trailing_advances": round(proba_trail, 4),
            "cv_accuracy":            cv_acc,
            "features": {
                "agg_diff":             mf.agg_diff,
                "home_xg":              mf.home_xg,
                "away_xg":              mf.away_xg,
                "form_diff":            mf.form_diff,
                "h2h_win_rate":         mf.h2h_win_rate,
                "league_strength_diff": mf.league_strength_diff,
            },
        }

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def cv_accuracy(self) -> float:
        if self._cv_scores is None:
            return 0.0
        return float(self._cv_scores.mean())

    @property
    def feature_coefficients(self) -> dict:
        """Return feature name → coefficient mapping (for explainability)."""
        if not self._trained:
            return {}
        coef = self._pipeline.named_steps["lr"].coef_[0]
        return dict(zip(FEATURE_COLS, coef.tolist()))
