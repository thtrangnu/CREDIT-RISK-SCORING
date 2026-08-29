"""Orchestrator: ghép 7 bảng Home Credit -> 1 bảng phẳng 1 dòng/applicant.

`FeaturePipeline` đóng gói toàn bộ transform (categories cố định lúc fit trên
train, áp lại y hệt lúc transform) — đây là ranh giới duy nhất giữa ml/ và
backend/ (xem docs/NOTES.md mục 4): backend chỉ load `feature_pipeline.pkl`,
KHÔNG import gì từ ml/src/train*.py.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregations import infer_categories, merge_all
from .bureau import BUREAU_CAT_COLS, aggregate_bureau
from .credit_card import CC_CAT_COLS, aggregate_credit_card
from .installments import aggregate_installments
from .pos_cash import POS_CAT_COLS, aggregate_pos_cash
from .previous_application import PREV_CAT_COLS, aggregate_previous_application

ML_DIR = Path(__file__).parent.parent.parent
ROOT_DIR = ML_DIR.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ML_DIR / "artifacts"

ID_COL, TARGET = "SK_ID_CURR", "TARGET"

RAW_FILES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}


def load_raw_tables(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(data_dir / fname) for name, fname in RAW_FILES.items()}


class FeaturePipeline:
    """`fit` trên tập train một lần; `transform` tái dùng y hệt lúc serve 1 applicant."""

    def __init__(
        self,
        app_categories: dict[str, list],
        bureau_categories: dict[str, list],
        prev_categories: dict[str, list],
        pos_categories: dict[str, list],
        cc_categories: dict[str, list],
    ) -> None:
        self.app_categories = app_categories
        self.bureau_categories = bureau_categories
        self.prev_categories = prev_categories
        self.pos_categories = pos_categories
        self.cc_categories = cc_categories
        self.feature_names: list[str] | None = None

    @classmethod
    def fit(cls, tables: dict[str, pd.DataFrame]) -> "FeaturePipeline":
        app = tables["application"]
        app_cat_cols = app.select_dtypes(include="object").columns.tolist()
        return cls(
            app_categories=infer_categories(app, app_cat_cols),
            bureau_categories=infer_categories(tables["bureau"], BUREAU_CAT_COLS),
            prev_categories=infer_categories(tables["previous_application"], PREV_CAT_COLS),
            pos_categories=infer_categories(tables["pos_cash"], POS_CAT_COLS),
            cc_categories=infer_categories(tables["credit_card"], CC_CAT_COLS),
        )

    def transform(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        app = tables["application"].copy()
        # Sentinel lỗi data nổi tiếng của Home Credit: 365243 = "không áp dụng"
        # (~18% dòng, chủ yếu hưu trí) — dọn về NaN, KHÔNG phải impute (không suy
        # đoán giá trị, chỉ sửa placeholder sai thành missing đúng nghĩa).
        if "DAYS_EMPLOYED" in app.columns:
            app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)
        for c, cats in self.app_categories.items():
            app[c] = pd.Categorical(app[c], categories=cats)

        bureau_agg = aggregate_bureau(
            tables["bureau"], tables.get("bureau_balance"), categories=self.bureau_categories
        )
        prev_agg = aggregate_previous_application(
            tables["previous_application"], categories=self.prev_categories
        )
        pos_agg = aggregate_pos_cash(tables["pos_cash"], categories=self.pos_categories)
        instal_agg = aggregate_installments(tables["installments"])
        cc_agg = aggregate_credit_card(tables["credit_card"], categories=self.cc_categories)

        flat = merge_all(app, ID_COL, bureau_agg, prev_agg, pos_agg, instal_agg, cc_agg)

        float_cols = flat.select_dtypes(include=["float64"]).columns
        flat[float_cols] = flat[float_cols].astype("float32")
        return flat

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "FeaturePipeline":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_features(data_dir: Path = DATA_DIR) -> tuple[FeaturePipeline, pd.DataFrame]:
    """Fit pipeline trên train + transform -> (pipeline, bảng phẳng 1 dòng/applicant)."""
    tables = load_raw_tables(data_dir)
    pipeline = FeaturePipeline.fit(tables)
    flat = pipeline.transform(tables)
    pipeline.feature_names = [c for c in flat.columns if c not in (ID_COL, TARGET)]
    return pipeline, flat


def main() -> None:
    pipeline, flat = build_features()
    print(f"flat shape={flat.shape}  #features={len(pipeline.feature_names)}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    pipeline.save(ARTIFACTS_DIR / "feature_pipeline.pkl")
    with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(pipeline.feature_names, f, indent=2, ensure_ascii=False)
    flat.to_parquet(ARTIFACTS_DIR / "train_features.parquet", index=False)
    print(f"Saved pipeline + feature_names.json + train_features.parquet -> {ARTIFACTS_DIR}")


# CỐ TÌNH không có `if __name__ == "__main__":` ở đây. Module này ĐỊNH NGHĨA
# FeaturePipeline — nếu chạy trực tiếp bằng `python -m ml.src.features.build`,
# Python nạp module này AS `__main__`, khiến FeaturePipeline.__module__ bị ghi
# thành "__main__" lúc pickle, và feature_pipeline.pkl KHÔNG unpickle được từ
# bất kỳ entry point nào khác (backend, pytest, notebook...) — lỗi thật đã gặp.
# Dùng `python -m ml.src.run_build_features` (chỉ import main(), không định
# nghĩa class) để build.py luôn được nạp như module bình thường.

#
# 2026-08-28: khối `if __name__ == "__main__": main()` từng bị thêm lại vào
# đây (mâu thuẫn với chính comment ở trên) — đã xoá. ĐỪNG thêm lại.
