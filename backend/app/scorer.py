"""Score a single applicant. ONLY loads artifacts from ml/artifacts/, never trains.

Layer boundary (docs/NOTES.md section 5, "never imports from ml/src/train.py"): imports
here are limited to `ml.src.features.build` (FeaturePipeline, the feature CONTRACT, a
pure transform with no training), `ml.src.explain` (pure SHAP inference, no training),
and `ml.src.reason_codes` (pure data and functions). The backend NEVER imports
`ml.src.train`, `ml.src.train_engineered`, or `ml.src.calibrate`, since those three are
the actual TRAINING orchestration.

model.txt and calibrator.pkl carry every learned parameter; the scorer only predicts.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.src.explain import build_explainer, compute_shap_values
from ml.src.features.build import FeaturePipeline

from .reason_codes import build_reason_codes

BASE_DEFAULT_RATE = 0.0807  # observed default rate in application_train, the risk-tier reference


def risk_tier(pd_score: float, base_rate: float = BASE_DEFAULT_RATE) -> str:
    """Risk tier relative to the population base rate rather than an arbitrary threshold.

    The returned labels are Vietnamese product values: they are shown in the UI and stored
    in the audit trail, so they are intentionally not translated.
    """
    if pd_score < base_rate / 2:
        return "Thấp"
    if pd_score < base_rate * 2:
        return "Trung bình"
    return "Cao"


class Scorer:
    def __init__(self, artifacts_dir: Path, model_version: str):
        self.model_version = model_version
        self.pipeline = FeaturePipeline.load(artifacts_dir / "feature_pipeline.pkl")
        self.model = lgb.Booster(model_file=str(artifacts_dir / "model.txt"))
        with open(artifacts_dir / "calibrator.pkl", "rb") as f:
            calibrator = pickle.load(f)
        self.calibrator_method: str = calibrator["method"]
        self.calibrator_model = calibrator["model"]
        with open(artifacts_dir / "feature_names.json") as f:
            self.feature_names: list[str] = json.load(f)
        # Build the explainer ONCE at startup. Walking all ~1400 trees is the expensive
        # part of SHAP, far more than the computation for a single row. This used to be
        # rebuilt on every /api/score request.
        self.explainer = build_explainer(self.model)

    def _predict_calibrated(self, prob_uncalibrated: np.ndarray) -> np.ndarray:
        if self.calibrator_method == "isotonic":
            return self.calibrator_model.predict(prob_uncalibrated)
        return self.calibrator_model.predict_proba(prob_uncalibrated.reshape(-1, 1))[:, 1]

    def score(self, tables: dict[str, pd.DataFrame], top_k_reasons: int = 5) -> dict:
        flat = self.pipeline.transform(tables)
        # ALWAYS reindex by feature_names.json. Booster.predict() matches DataFrame
        # columns BY POSITION, not by name (verified by hand in Block 6, see
        # ml/src/explain.py).
        X = flat[self.feature_names]

        prob_uncalibrated = self.model.predict(X)
        pd_score = float(self._predict_calibrated(prob_uncalibrated)[0])

        shap_values, base_values = compute_shap_values(self.model, X, explainer=self.explainer)
        reasons = build_reason_codes(self.feature_names, shap_values[0], X.iloc[0].values, top_k=top_k_reasons)
        # base_value + raw_margin anchor the frontend waterfall chart. `reasons` only
        # carries the top-K features, so these two numbers are needed to make the
        # "everything else" bar add up correctly.
        base_value = float(base_values[0])
        raw_margin = float(base_value + shap_values[0].sum())

        return {
            "sk_id_curr": int(flat.iloc[0]["SK_ID_CURR"]),
            "pd_uncalibrated": float(prob_uncalibrated[0]),
            "pd_score": pd_score,
            "risk_tier": risk_tier(pd_score),
            "model_version": self.model_version,
            "reasons": reasons,
            "base_value": base_value,
            "raw_margin": raw_margin,
        }
