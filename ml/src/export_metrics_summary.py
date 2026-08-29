"""Write ml/artifacts/metrics_summary.json from existing artifacts. Retrains nothing.
The backend insights router reads this file instead of parsing model_card.md.

Four groups:
  1. Ranking quality   - AUC/Gini, baseline vs engineered
  2. Calibration       - Brier/ECE, including quantile binning (which avoids the
                         illusion created when a uniform bin swallows all the mass
                         in the low-PD region)
  3. Policy (Block 9)  - cutoff table: approve X% -> how much default risk drops.
                         THIS is the number that means something to the business.
  4. Segments/fairness - gender and age band (protected attributes under ECOA)
                         plus adverse impact ratios.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .calibrate import nested_calibrate
from .features.build import ARTIFACTS_DIR, TARGET
from .metrics import brier_score, expected_calibration_error, gini, roc_auc
from .policy import (REFERENCE_APPROVAL_RATE, adverse_impact_ratio, age_bands,
                     cutoff_table, segment_report)
from .train_engineered import CACHED_FEATURES_PATH, CONFIG_PATH, load_config

# Columns used for segment reporting: protected attributes in lending.
SEGMENT_COLS = ["CODE_GENDER", "DAYS_BIRTH"]


def json_safe(obj):
    """Recursively map NaN -> None.

    MANDATORY: `json.dump` writes NaN as the literal `NaN`, which is not valid JSON.
    Python's `json.loads` still reads it, so the failure is silent in the ML layer,
    but the browser's `JSON.parse` THROWS and the Insights page goes blank.
    NaN really does occur here: the AUC of a single-class group (CODE_GENDER='XNA',
    4 rows, all TARGET=0).
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def calibration_block(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict:
    """Report ECE under several binning schemes; a conclusion only counts if it holds in all three."""
    out = {
        "brier_raw": brier_score(y, raw),
        "brier_calibrated": brier_score(y, calibrated),
    }
    for key, kwargs in {
        "ece_uniform_10": {"n_bins": 10, "strategy": "uniform"},
        "ece_quantile_10": {"n_bins": 10, "strategy": "quantile"},
        "ece_quantile_50": {"n_bins": 50, "strategy": "quantile"},
    }.items():
        out[f"{key}_raw"] = expected_calibration_error(y, raw, **kwargs)
        out[f"{key}_calibrated"] = expected_calibration_error(y, calibrated, **kwargs)
    # Keep the old key names for backwards compatibility with an already-built frontend.
    out["ece_raw"] = out["ece_uniform_10_raw"]
    out["ece_calibrated"] = out["ece_uniform_10_calibrated"]
    return out


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    n_folds, seed = cfg["n_folds"], cfg["seed"]

    meta = pd.read_parquet(CACHED_FEATURES_PATH, columns=[TARGET] + SEGMENT_COLS)
    y = meta[TARGET].values
    oof_baseline = np.load(ARTIFACTS_DIR / "oof_baseline.npy")
    oof_engineered = np.load(ARTIFACTS_DIR / "oof_engineered.npy")

    calibrated = nested_calibrate(oof_engineered, y, "isotonic", n_folds=n_folds, seed=seed)

    policy = cutoff_table(y, calibrated)
    gender = segment_report(y, calibrated, meta["CODE_GENDER"].astype(str), REFERENCE_APPROVAL_RATE)
    age = segment_report(y, calibrated, age_bands(meta["DAYS_BIRTH"]), REFERENCE_APPROVAL_RATE)

    summary = {
        "n_train_rows": int(len(y)),
        "default_rate": float(y.mean()),
        "n_features": int(pd.read_parquet(CACHED_FEATURES_PATH).shape[1] - 2),
        "baseline": {
            "auc": roc_auc(y, oof_baseline),
            "gini": gini(y, oof_baseline),
        },
        "engineered": {
            "auc": roc_auc(y, oof_engineered),
            "gini": gini(y, oof_engineered),
            **calibration_block(y, oof_engineered, calibrated),
        },
        "auc_delta": roc_auc(y, oof_engineered) - roc_auc(y, oof_baseline),
        "policy": {
            "reference_approval_rate": REFERENCE_APPROVAL_RATE,
            "table": policy.to_dict(orient="records"),
        },
        "fairness": {
            "reference_approval_rate": REFERENCE_APPROVAL_RATE,
            "by_gender": gender.to_dict(orient="records"),
            "by_age_band": age.to_dict(orient="records"),
            "adverse_impact_ratio_gender": adverse_impact_ratio(gender),
            "adverse_impact_ratio_age": adverse_impact_ratio(age),
        },
    }

    with open(ARTIFACTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2, allow_nan=False)

    print("=== CUTOFF POLICY (calibrated OOF) ===")
    print(policy.round(4).to_string(index=False))
    print("\n=== SEGMENTS - GENDER ===")
    print(gender.round(4).to_string(index=False))
    print("\n=== SEGMENTS - AGE BAND ===")
    print(age.round(4).to_string(index=False))
    print(f"\nAdverse impact ratio (4/5ths rule, threshold 0.8): "
          f"gender={summary['fairness']['adverse_impact_ratio_gender']:.4f}  "
          f"age={summary['fairness']['adverse_impact_ratio_age']:.4f}")
    print(f"\nSaved -> {ARTIFACTS_DIR / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
