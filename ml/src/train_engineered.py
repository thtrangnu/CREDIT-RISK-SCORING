"""Blocks 2+3: retrain OOF on multi-table engineered features + monotonic constraints.

Compared directly against the baseline (train.py) on the SAME fold split (same seed,
same n_folds, same row order in application_train.csv) so the measured lift comes from
feature engineering rather than from a different random fold assignment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml
from sklearn.model_selection import StratifiedKFold

from .features.build import ARTIFACTS_DIR, ID_COL, TARGET, build_features
from .metrics import (roc_auc, gini, ks_statistic, average_precision,
                      partial_auc, tpr_at_fpr, brier_score,
                      expected_calibration_error, decile_table)

ML_DIR = Path(__file__).parent.parent
CONFIG_PATH = ML_DIR / "config" / "params.yaml"
FEATURES_CONFIG_PATH = ML_DIR / "config" / "features.yaml"
CACHED_FEATURES_PATH = ARTIFACTS_DIR / "train_features.parquet"
BASELINE_OOF_PATH = ARTIFACTS_DIR / "oof_baseline.npy"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_engineered_features() -> pd.DataFrame:
    """Use the cached parquet if present (build.py already ran) to skip a ~50s rebuild."""
    if CACHED_FEATURES_PATH.exists():
        return pd.read_parquet(CACHED_FEATURES_PATH)
    _, flat = build_features()
    return flat


def build_monotone_constraints(columns: list[str], cat_cols: set[str], declared: dict[str, int]) -> list[int]:
    """Map features.yaml onto a constraint array matching X's column order.

    Categorical features must NOT carry a constraint (LightGBM only applies monotonic
    behaviour to numeric/ordinal splits), so a constraint mistakenly declared for a
    categorical column is silently dropped to 0.
    """
    constraints = []
    for c in columns:
        value = declared.get(c, 0)
        if c in cat_cols and value != 0:
            value = 0
        constraints.append(value)
    return constraints


def load_data(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    y = df[TARGET].values
    X = df.drop(columns=[TARGET, ID_COL])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        if X[c].dtype != "category":
            X[c] = X[c].astype("category")
    return X, y, cat_cols


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    n_folds: int = cfg["n_folds"]
    seed: int = cfg["seed"]
    lgbm_cfg: dict = cfg["lgbm"]

    num_boost_round = lgbm_cfg["num_boost_round"]
    early_stopping_rounds = lgbm_cfg["early_stopping_rounds"]
    params = {k: v for k, v in lgbm_cfg.items()
              if k not in ("num_boost_round", "early_stopping_rounds")}
    params["seed"] = seed

    feat_cfg = load_config(FEATURES_CONFIG_PATH)
    declared_constraints: dict[str, int] = feat_cfg.get("monotone_constraints", {})

    df = load_engineered_features()
    X, y, cat_cols = load_data(df)
    monotone_constraints = build_monotone_constraints(list(X.columns), set(cat_cols), declared_constraints)
    n_declared = sum(1 for v in monotone_constraints if v != 0)
    params["monotone_constraints"] = monotone_constraints
    print(f"shape={X.shape}  default_rate={y.mean():.4f}  #cat={len(cat_cols)}  "
          f"#monotone={n_declared}")

    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        # KNOWN LIMITATION (see model_card.md section 6): `dva` is both the
        # valid_set driving early stopping and the fold supplying oof[va]. The
        # iteration count is therefore chosen USING the very data it is about to
        # predict -> mildly optimistic OOF, not a fully clean estimate. Kept
        # deliberately: baseline and engineered are biased identically so the delta
        # stays fair. For a clean OOF, carve an inner split out of `tr`.
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        model = lgb.train(
            params, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        print(f"  fold {fold}: AUC={roc_auc(y[va], oof[va]):.5f}  "
              f"(best_iter={model.best_iteration})")

    print("\n=== OOF — RANKING (engineered) ===")
    print(f"AUC       : {roc_auc(y, oof):.5f}")
    print(f"Gini      : {gini(y, oof):.5f}")
    print(f"KS        : {ks_statistic(y, oof):.5f}")
    print(f"PR-AUC    : {average_precision(y, oof):.5f}   (baseline = {y.mean():.4f})")
    print(f"pAUC@20%  : {partial_auc(y, oof, max_fpr=0.2):.5f}")
    print(f"TPR@FPR10%: {tpr_at_fpr(y, oof, 0.1):.5f}")

    print("\n=== OOF — CALIBRATION (engineered) ===")
    print(f"Brier     : {brier_score(y, oof):.5f}")
    print(f"ECE(10)   : {expected_calibration_error(y, oof):.5f}")

    print("\n=== DECILE TABLE (OOF, engineered) ===")
    print(decile_table(y, oof).to_string(index=False))

    if BASELINE_OOF_PATH.exists():
        oof_baseline = np.load(BASELINE_OOF_PATH)
        if len(oof_baseline) == len(y):
            auc_engineered, auc_baseline = roc_auc(y, oof), roc_auc(y, oof_baseline)
            print("\n=== DELTA vs BASELINE (Block 1) ===")
            print(f"AUC  baseline={auc_baseline:.5f}  engineered={auc_engineered:.5f}  "
                  f"delta={auc_engineered - auc_baseline:+.5f}")
        else:
            print("\n[!] oof_baseline.npy has a different length; skipping the delta comparison.")
    else:
        print("\n[!] No ml/artifacts/oof_baseline.npy yet; run `python -m ml.src.train` first to compare.")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    np.save(ARTIFACTS_DIR / "oof_engineered.npy", oof)
    print(f"\nSaved → {ARTIFACTS_DIR / 'oof_engineered.npy'}")


if __name__ == "__main__":
    main()
