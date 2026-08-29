"""Chấm điểm 1 applicant — CHỈ load artifact từ ml/artifacts/, KHÔNG train.

Biên giới tầng (docs/NOTES.md mục 4, "KHÔNG import từ ml/src/train.py"): import ở
đây giới hạn trong `ml.src.features.build` (FeaturePipeline — feature
CONTRACT thuần transform, không train), `ml.src.explain` (SHAP inference
thuần, không train) và `ml.src.reason_codes` (dữ liệu/hàm thuần). Backend
KHÔNG BAO GIỜ import `ml.src.train`, `ml.src.train_engineered`, hay
`ml.src.calibrate` — 3 module đó là orchestration TRAINING thật sự.

model.txt/calibrator.pkl tự chứa toàn bộ tham số đã học; scorer chỉ predict.
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

BASE_DEFAULT_RATE = 0.0807  # default_rate quan sát trên application_train — mốc so sánh risk tier


def risk_tier(pd_score: float, base_rate: float = BASE_DEFAULT_RATE) -> str:
    """Risk tier tương đối so với base rate quần thể, không phải ngưỡng tuỳ tiện."""
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
        # Dựng explainer 1 LẦN lúc startup: duyệt toàn bộ ~1400 cây là phần đắt
        # nhất của SHAP, đắt hơn hẳn phép tính cho 1 dòng. Trước đây nó bị dựng
        # lại trong mỗi request /api/score.
        self.explainer = build_explainer(self.model)

    def _predict_calibrated(self, prob_uncalibrated: np.ndarray) -> np.ndarray:
        if self.calibrator_method == "isotonic":
            return self.calibrator_model.predict(prob_uncalibrated)
        return self.calibrator_model.predict_proba(prob_uncalibrated.reshape(-1, 1))[:, 1]

    def score(self, tables: dict[str, pd.DataFrame], top_k_reasons: int = 5) -> dict:
        flat = self.pipeline.transform(tables)
        # LUÔN reindex theo feature_names.json — Booster.predict() khớp cột DataFrame
        # theo VỊ TRÍ chứ không theo tên (đã verify tay ở Block 6, xem ml/src/explain.py).
        X = flat[self.feature_names]

        prob_uncalibrated = self.model.predict(X)
        pd_score = float(self._predict_calibrated(prob_uncalibrated)[0])

        shap_values, base_values = compute_shap_values(self.model, X, explainer=self.explainer)
        reasons = build_reason_codes(self.feature_names, shap_values[0], X.iloc[0].values, top_k=top_k_reasons)
        # base_value + raw_margin: neo cho waterfall chart ở frontend — reasons chỉ
        # có top-K feature, cần 2 số này để cộng dồn "các feature còn lại" cho khớp.
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
