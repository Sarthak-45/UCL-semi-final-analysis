"""
Ensemble model: Calibrated Logistic Regression + XGBoost.

Improvements over the baseline LR:
  • XGBoost added alongside LR — ensemble average reduces single-model variance
  • LOO Brier score for honest calibration measurement on the small dataset
  • Optimal inverse-Brier-score weighting instead of naive 50/50 average
  • Model agreement indicator flags when the two classifiers strongly disagree
  • Feature-contribution layer (LR coefficient × scaled feature) drives the
    "Why model favors X" explanation bullets shown in the dashboard
  • Stores _lr_loo_probs / _xgb_loo_probs / _y_train for calibration chart
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score, LeaveOneOut
from sklearn.metrics import brier_score_loss

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from src.data_ingestion.historical_data import get_training_dataframe
from src.feature_engineering.features import MatchupFeatures


FEATURE_COLS = ["agg_diff", "home_xg", "away_xg", "form_diff", "h2h", "ls_diff"]

FEATURE_LABELS = {
    "agg_diff":  "First-leg aggregate lead",
    "home_xg":   "Home attacking xG",
    "away_xg":   "Away attacking xG",
    "form_diff": "UCL form advantage",
    "h2h":       "Head-to-head record",
    "ls_diff":   "League strength gap",
}


class UCLEnsembleModel:
    """
    Two-model ensemble (LR + XGBoost) with:
      • StandardScaler normalisation
      • Leave-One-Out Brier score for calibration check
      • Inverse-Brier-score optimal weighting (replaces naive 50/50)
      • Feature-contribution analysis for explainability
      • Natural-language "why" bullets
      • Stored LOO predictions for calibration reliability diagram
    """

    def __init__(self, random_state: int = 42) -> None:
        self._scaler = StandardScaler()
        self._lr = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=random_state
        )
        self._xgb = (
            XGBClassifier(
                max_depth=2, n_estimators=40, learning_rate=0.1,
                subsample=0.8, eval_metric="logloss",
                random_state=random_state, verbosity=0,
            )
            if _HAS_XGB else None
        )
        self._trained = False
        self.brier_lr: Optional[float]       = None
        self.brier_xgb: Optional[float]      = None
        self.brier_ensemble: Optional[float] = None
        self.cv_accuracy: float              = 0.0
        self._feature_importances: dict      = {}

        # Stored for calibration chart
        self._lr_loo_probs:  Optional[np.ndarray] = None
        self._xgb_loo_probs: Optional[np.ndarray] = None
        self._y_train:       Optional[np.ndarray] = None

        # Optimal inverse-Brier weights (computed in train)
        self._w_lr:  float = 0.5
        self._w_xgb: float = 0.5

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self) -> "UCLEnsembleModel":
        df   = get_training_dataframe()
        Xraw = df[FEATURE_COLS].values.astype(float)
        y    = df["advanced"].values.astype(int)
        X    = self._scaler.fit_transform(Xraw)

        self._lr.fit(X, y)
        if self._xgb:
            self._xgb.fit(X, y)

        # ── LOO Brier scores ───────────────────────────────────────────────────
        loo = LeaveOneOut()
        lr_probs = cross_val_predict(
            self._lr, X, y, cv=loo, method="predict_proba"
        )[:, 1]
        self.brier_lr      = round(float(brier_score_loss(y, lr_probs)), 4)
        self._lr_loo_probs = lr_probs
        self._y_train      = y

        if self._xgb:
            xgb_probs = cross_val_predict(
                self._xgb, X, y, cv=loo, method="predict_proba"
            )[:, 1]
            self.brier_xgb      = round(float(brier_score_loss(y, xgb_probs)), 4)
            self._xgb_loo_probs = xgb_probs

            # ── Optimal inverse-Brier-score weighting ──────────────────────────
            # w_lr ∝ 1/Brier_lr ; w_xgb ∝ 1/Brier_xgb  (lower Brier → more weight)
            inv_lr  = 1.0 / max(self.brier_lr,  1e-6)
            inv_xgb = 1.0 / max(self.brier_xgb, 1e-6)
            total   = inv_lr + inv_xgb
            self._w_lr  = inv_lr  / total
            self._w_xgb = inv_xgb / total

            ens_probs = self._w_lr * lr_probs + self._w_xgb * xgb_probs
            self.brier_ensemble = round(float(brier_score_loss(y, ens_probs)), 4)
        else:
            self._w_lr  = 1.0
            self._w_xgb = 0.0

        # ── CV accuracy ────────────────────────────────────────────────────────
        cv_scores    = cross_val_score(self._lr, X, y, cv=min(5, len(y)))
        self.cv_accuracy = round(float(cv_scores.mean()), 3)

        # ── Feature importances: normalised blend of LR |coef| + XGB gain ─────
        lr_imp = np.abs(self._lr.coef_[0])
        lr_imp = lr_imp / (lr_imp.max() or 1)
        if self._xgb:
            xgb_imp = self._xgb.feature_importances_
            xgb_imp = xgb_imp / (xgb_imp.max() or 1)
            blended = self._w_lr * lr_imp + self._w_xgb * xgb_imp
        else:
            blended = lr_imp
        self._feature_importances = {
            f: round(float(v), 4) for f, v in zip(FEATURE_COLS, blended)
        }

        self._trained = True
        return self

    def _ensure_trained(self) -> None:
        if not self._trained:
            self.train()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _raw_probs(self, x_scaled: np.ndarray) -> Tuple[float, float]:
        """Return (lr_prob_leading, xgb_prob_leading)."""
        classes  = list(self._lr.classes_)
        idx1     = next((i for i, c in enumerate(classes) if int(c) == 1), 1)
        lr_prob  = float(self._lr.predict_proba(x_scaled)[0][idx1])

        if self._xgb:
            xc   = list(self._xgb.classes_)
            xi   = next((i for i, c in enumerate(xc) if int(c) == 1), 1)
            xgb_prob = float(self._xgb.predict_proba(x_scaled)[0][xi])
        else:
            xgb_prob = lr_prob
        return lr_prob, xgb_prob

    def _contributions(self, x_scaled: np.ndarray) -> List[Tuple]:
        """
        Per-feature contribution = LR_coef × scaled_value.
        Returns list of (label, abs_magnitude, direction, raw_contrib)
        sorted by magnitude descending.
        """
        vals   = x_scaled[0]
        coefs  = self._lr.coef_[0]
        result = []
        for i, feat in enumerate(FEATURE_COLS):
            raw = float(coefs[i] * vals[i])
            direction = "favors_leading" if raw > 0 else "favors_trailing"
            result.append((FEATURE_LABELS[feat], abs(raw), direction, raw))
        return sorted(result, key=lambda t: t[1], reverse=True)

    def _why_bullets(self, contribs: list, mf: MatchupFeatures) -> List[str]:
        """Generate 5 natural-language explanation bullets."""
        bullets = []
        for label, mag, direction, _ in contribs[:5]:
            favored  = mf.leading_team  if direction == "favors_leading"  else mf.trailing_team
            strength = "strongly" if mag > 0.4 else "moderately" if mag > 0.2 else "slightly"

            if "aggregate" in label.lower():
                if mf.agg_diff == 0:
                    bullets.append(
                        f"**{label}:** Tied on aggregate — any goal from either side is"
                        f" immediately decisive. Neutral impact."
                    )
                else:
                    bullets.append(
                        f"**{label}:** {mf.leading_team} hold a {mf.agg_diff}-goal lead —"
                        f" {strength} benefits **{favored}**."
                    )
            elif "home" in label.lower():
                bullets.append(
                    f"**{label}:** {mf.home_team} average **{mf.home_xg:.2f} xG/game**"
                    f" at home in UCL — {strength} favors **{favored}**."
                )
            elif "away" in label.lower():
                bullets.append(
                    f"**{label}:** {mf.away_team} carry **{mf.away_xg:.2f} xG/game**"
                    f" on the road — {strength} favors **{favored}**."
                )
            elif "form" in label.lower():
                if abs(mf.form_diff) < 0.05:
                    bullets.append(
                        f"**{label}:** Both teams are evenly matched in recent UCL form"
                        f" (gap {mf.form_diff:+.2f}) — minimal predictive weight."
                    )
                else:
                    better = mf.leading_team if mf.form_diff > 0 else mf.trailing_team
                    bullets.append(
                        f"**{label}:** {better} have the stronger UCL form record"
                        f" (gap {mf.form_diff:+.2f}) — {strength} favors **{favored}**."
                    )
            elif "head" in label.lower():
                h = mf.h2h_stats
                if h.get("played", 0) < 3:
                    bullets.append(
                        f"**{label}:** Only {h.get('played', 0)} competitive meetings found"
                        f" — low statistical weight, treated as near-neutral."
                    )
                else:
                    bullets.append(
                        f"**{label}:** {h.get('wins',0)}W {h.get('draws',0)}D"
                        f" {h.get('losses',0)}L in {h.get('played',0)} meetings"
                        f" — {strength} favors **{favored}**."
                    )
            elif "league" in label.lower():
                bullets.append(
                    f"**{label}:** {mf.leading_team}'s domestic league is rated"
                    f" {abs(mf.league_strength_diff):.1f} pts higher"
                    f" — {strength} favors **{favored}**."
                )
        return bullets

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict_from_matchup(self, mf: MatchupFeatures) -> dict:
        self._ensure_trained()
        x_raw    = mf.to_array().reshape(1, -1)
        x_scaled = self._scaler.transform(x_raw)

        lr_prob, xgb_prob = self._raw_probs(x_scaled)

        # Weighted ensemble (optimal inverse-Brier weights)
        ens_prob_lead  = self._w_lr * lr_prob + self._w_xgb * xgb_prob
        ens_prob_trail = 1.0 - ens_prob_lead

        diff = abs(lr_prob - xgb_prob)
        agreement = "🟢 High" if diff < 0.08 else ("🟡 Moderate" if diff < 0.15 else "🔴 Models disagree")

        contribs = self._contributions(x_scaled)
        bullets  = self._why_bullets(contribs, mf)

        brier_display = self.brier_ensemble if self.brier_ensemble is not None else self.brier_lr

        return {
            # compatible with old lr["..."] keys used in app.py
            "leading_team":             mf.leading_team,
            "trailing_team":            mf.trailing_team,
            "prob_leading_advances":    round(ens_prob_lead,  4),
            "prob_trailing_advances":   round(ens_prob_trail, 4),
            "cv_accuracy":              self.cv_accuracy,
            # new keys
            "lr_prob":                  round(lr_prob,  4),
            "xgb_prob":                 round(xgb_prob, 4),
            "agreement":                agreement,
            "brier":                    brier_display,
            "brier_lr":                 self.brier_lr,
            "brier_xgb":                self.brier_xgb,
            "feature_contributions":    contribs,
            "explanation_bullets":      bullets,
            "feature_importances":      self._feature_importances,
            # weighting explanation
            "model_weights": {
                "lr":  round(self._w_lr,  3),
                "xgb": round(self._w_xgb, 3),
            },
            "features": {
                "agg_diff":             mf.agg_diff,
                "home_xg":              mf.home_xg,
                "away_xg":              mf.away_xg,
                "form_diff":            mf.form_diff,
                "h2h_win_rate":         mf.h2h_win_rate,
                "league_strength_diff": mf.league_strength_diff,
            },
        }

    def calibration_data(self) -> dict:
        """
        Returns LOO predicted probabilities and true labels for calibration chart.
        Bins predicted probs into deciles; computes mean predicted vs actual win rate.
        """
        self._ensure_trained()
        if self._lr_loo_probs is None or self._y_train is None:
            return {}

        y = self._y_train
        lr_p = self._lr_loo_probs

        if self._xgb_loo_probs is not None:
            ens_p = self._w_lr * lr_p + self._w_xgb * self._xgb_loo_probs
        else:
            ens_p = lr_p

        bins = np.linspace(0, 1, 6)   # 5 bins: 0-0.2, 0.2-0.4, ...
        mean_pred, mean_actual, counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (ens_p >= lo) & (ens_p < hi)
            if mask.sum() > 0:
                mean_pred.append(float(ens_p[mask].mean()))
                mean_actual.append(float(y[mask].mean()))
                counts.append(int(mask.sum()))

        return {
            "mean_predicted": mean_pred,
            "mean_actual":    mean_actual,
            "counts":         counts,
            "n_samples":      int(len(y)),
        }

    @property
    def feature_coefficients(self) -> dict:
        if not self._trained:
            return {}
        return dict(zip(FEATURE_COLS, self._lr.coef_[0].tolist()))

    @property
    def model_weights(self) -> dict:
        self._ensure_trained()
        return {"lr": round(self._w_lr, 3), "xgb": round(self._w_xgb, 3)}
