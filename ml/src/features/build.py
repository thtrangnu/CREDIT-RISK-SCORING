"""Orchestrator: join the 7 Home Credit tables into one flat table, one row per applicant.

`FeaturePipeline` wraps the whole transform (categories fixed at fit time on train, then
reapplied identically at transform time). This is the only boundary between ml/ and
backend/ (see docs/NOTES.md section 5): the backend just loads `feature_pipeline.pkl` and
imports nothing from ml/src/train*.py.
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
    """`fit` once on train; `transform` is reused verbatim when serving a single applicant."""

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
        # Home Credit's well-known data sentinel: 365243 = "not applicable" (~18% of
        # rows, mostly retirees). Cleaned to NaN, which is NOT imputation: no value is
        # guessed, a wrong placeholder is just turned into a proper missing value.
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
    """Fit the pipeline on train and transform -> (pipeline, flat table, one row per applicant)."""
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


# There is DELIBERATELY no `if __name__ == "__main__":` here. This module DEFINES
# FeaturePipeline, so running it directly via `python -m ml.src.features.build` makes
# Python load it AS `__main__`, which records FeaturePipeline.__module__ as "__main__"
# in the pickle. feature_pipeline.pkl then CANNOT be unpickled from any other entry
# point (backend, pytest, notebooks, ...). This actually happened.
# Use `python -m ml.src.run_build_features` (which only imports main() and defines no
# classes) so build.py is always loaded as a normal module.

#
# 2026-08-28: an `if __name__ == "__main__": main()` block was re-added here at one
# point, contradicting the comment right above it. Removed. DO NOT add it back.
