from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml
from sklearn.model_selection import StratifiedKFold

from .metrics import (roc_auc, gini, ks_statistic, average_precision,
                      partial_auc, tpr_at_fpr, brier_score,
                      expected_calibration_error, decile_table)

ML_DIR = Path(__file__).parent.parent
ROOT_DIR = ML_DIR.parent

CONFIG_PATH = ML_DIR / "config" / "params.yaml"
DATA_PATH = ROOT_DIR / "data" / "application_train.csv"
ARTIFACTS_DIR = ML_DIR / "artifacts"

TARGET, ID_COL = "TARGET", "SK_ID_CURR"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    y = df[TARGET].values
    X = df.drop(columns=[TARGET, ID_COL])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
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
    # No scale_pos_weight / is_unbalance, to keep the baseline calibration clean

    X, y, cat_cols = load_data(DATA_PATH)
    print(f"shape={X.shape}  default_rate={y.mean():.4f}  #cat={len(cat_cols)}")

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

    print("\n=== OOF — RANKING ===")
    print(f"AUC       : {roc_auc(y, oof):.5f}")
    print(f"Gini      : {gini(y, oof):.5f}")
    print(f"KS        : {ks_statistic(y, oof):.5f}")
    print(f"PR-AUC    : {average_precision(y, oof):.5f}   (baseline = {y.mean():.4f})")
    print(f"pAUC@20%  : {partial_auc(y, oof, max_fpr=0.2):.5f}")
    print(f"TPR@FPR10%: {tpr_at_fpr(y, oof, 0.1):.5f}")

    print("\n=== OOF — CALIBRATION ===")
    print(f"Brier     : {brier_score(y, oof):.5f}")
    print(f"ECE(10)   : {expected_calibration_error(y, oof):.5f}")

    print("\n=== DECILE TABLE (OOF) ===")
    print(decile_table(y, oof).to_string(index=False))

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    np.save(ARTIFACTS_DIR / "oof_baseline.npy", oof)
    print(f"\nSaved → {ARTIFACTS_DIR / 'oof_baseline.npy'}")


if __name__ == "__main__":
    main()
