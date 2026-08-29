"""Block 6: SHAP explainability — global importance + local reason codes.

Runs shap.TreeExplainer on model.txt (no Dataset, no retraining). Works in
margin/log-odds space, which is TreeExplainer's default for LightGBM binary.
Since the sigmoid is monotonic, the sign and relative magnitude of SHAP values
still answer "which feature pushed risk up or down", so there is no need to move
to probability space for reason codes.

IMPORTANT (verified by hand): `Booster.predict()` on a DataFrame matches columns
BY POSITION, not by name. Reorder the columns and you get a different result with
NO error raised. `model.feature_name()` is not reliable for reindexing either,
because LightGBM sanitizes column names containing special characters (spaces,
brackets, ...) when saving model.txt.
=> EVERY place that uses the model (here and in backend/scorer.py) must reindex X
into the exact order of `feature_names.json` before predicting or explaining.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import yaml

from .features.build import ARTIFACTS_DIR
from .reason_codes import build_reason_codes

FEATURES_CONFIG_PATH = Path(__file__).parent.parent / "config" / "features.yaml"
CACHED_FEATURES_PATH = ARTIFACTS_DIR / "train_features.parquet"
GLOBAL_SAMPLE_SIZE = 5000


def load_feature_names(artifacts_dir: Path = ARTIFACTS_DIR) -> list[str]:
    with open(artifacts_dir / "feature_names.json") as f:
        return json.load(f)


def load_model(artifacts_dir: Path = ARTIFACTS_DIR) -> lgb.Booster:
    return lgb.Booster(model_file=str(artifacts_dir / "model.txt"))


def build_explainer(model: lgb.Booster) -> "shap.TreeExplainer":
    """Build the TreeExplainer ONCE and reuse it.

    Constructing an explainer walks every tree (model.txt currently has ~1400),
    which is far more expensive than the SHAP computation for a single row. The
    backend holds one instance for the process lifetime (see backend/app/scorer.py)
    instead of rebuilding it on every request.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(
    model: lgb.Booster, X: pd.DataFrame, explainer: "shap.TreeExplainer | None" = None
) -> tuple[np.ndarray, np.ndarray]:
    """SHAP in margin space. Returns (shap_values [n, n_features], base_values [n]).

    `explainer`: pass a prebuilt instance to avoid reconstruction (the serving path).
    Leave it out and one is built on the spot, which is fine for one-off analysis
    scripts.
    """
    if explainer is None:
        explainer = build_explainer(model)
    exp = explainer(X)
    return np.asarray(exp.values), np.asarray(exp.base_values)


def global_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True))


def explain_applicant(model: lgb.Booster, feature_names: list[str], row: pd.DataFrame, top_k: int = 5) -> dict:
    """SHAP + reason codes for EXACTLY one applicant.

    Reindexes `row` by `feature_names` before predicting or explaining. The caller's
    column order is never trusted, because Booster matches columns positionally (see
    the module docstring). This is validation at the boundary and it blocks a whole
    class of skew bugs.
    """
    row = row[feature_names]
    shap_values, base_values = compute_shap_values(model, row)
    reasons = build_reason_codes(feature_names, shap_values[0], row.iloc[0].values, top_k=top_k)
    raw_margin = float(base_values[0] + shap_values[0].sum())
    return {"raw_margin": raw_margin, "reasons": reasons}


def main() -> None:
    feature_names = load_feature_names()
    model = load_model()
    df = pd.read_parquet(CACHED_FEATURES_PATH)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(df), size=min(GLOBAL_SAMPLE_SIZE, len(df)), replace=False)
    sample = df.iloc[sample_idx]
    X_sample = sample[feature_names]

    print(f"Computing SHAP for {len(X_sample)} sampled rows ({len(feature_names)} features)...")
    shap_values, _ = compute_shap_values(model, X_sample)

    importance = global_importance(shap_values, feature_names)
    print("\n=== TOP 20 most important features (global mean|SHAP|) ===")
    print(importance.head(20).to_string(index=False))

    with open(FEATURES_CONFIG_PATH) as f:
        declared = yaml.safe_load(f).get("monotone_constraints", {})
    top30 = set(importance.head(30)["feature"])
    overlap = top30 & set(declared)
    print(f"\nOverlap between top-30 SHAP and the {len(declared)} monotonic-constrained features (Block 3): "
          f"{len(overlap)} -> {sorted(overlap)}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    importance.to_csv(ARTIFACTS_DIR / "shap_global_importance.csv", index=False)
    print(f"\nSaved -> {ARTIFACTS_DIR / 'shap_global_importance.csv'}")

    print("\n=== Reason code demo: the 3 riskiest applicants in the sample ===")
    raw_pred = model.predict(X_sample, raw_score=True)
    top_risk_pos = np.argsort(-raw_pred)[:3]
    for pos in top_risk_pos:
        sk_id = int(sample.iloc[pos]["SK_ID_CURR"])
        actual = int(sample.iloc[pos]["TARGET"]) if "TARGET" in sample.columns else None
        reasons = build_reason_codes(feature_names, shap_values[pos], X_sample.iloc[pos].values, top_k=5)
        print(f"\nSK_ID_CURR={sk_id}  raw_margin={raw_pred[pos]:.3f}  actual TARGET={actual}")
        for r in reasons:
            flag = "" if r["curated"] else "  [fallback, needs legal review]"
            print(f"  - {r['label']}: {r['direction']} (shap={r['shap']:+.3f}){flag}")


if __name__ == "__main__":
    main()
