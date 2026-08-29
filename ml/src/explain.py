"""Block 6: SHAP explainability — global importance + local reason codes.

Dùng shap.TreeExplainer trên model.txt (không cần Dataset/train lại). Làm việc
ở margin/log-odds space (mặc định của TreeExplainer với LightGBM binary) —
sigmoid đơn điệu nên dấu & độ lớn tương đối của SHAP vẫn đúng ý nghĩa "feature
nào đẩy rủi ro lên/xuống", không cần đổi sang probability space cho mục đích
reason codes.

QUAN TRỌNG (đã verify bằng tay): `Booster.predict()` trên DataFrame khớp cột
THEO VỊ TRÍ, không theo tên — đảo thứ tự cột cho ra kết quả khác mà KHÔNG báo
lỗi. `model.feature_name()` cũng không đáng tin để re-index vì LightGBM tự
sanitize tên cột có ký tự đặc biệt (dấu cách, ngoặc, ...) khi lưu model.txt.
=> MỌI nơi dùng model (ở đây và ở backend/scorer.py) đều phải reindex X theo
đúng thứ tự `feature_names.json` trước khi predict/explain.
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
    """Dựng TreeExplainer MỘT LẦN rồi tái dùng.

    Dựng explainer phải duyệt toàn bộ cây (model.txt hiện có ~1400 cây) — đắt
    hơn nhiều so với chính phép tính SHAP cho 1 dòng. Backend giữ 1 instance
    suốt vòng đời process (xem backend/app/scorer.py) thay vì dựng lại mỗi request.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(
    model: lgb.Booster, X: pd.DataFrame, explainer: "shap.TreeExplainer | None" = None
) -> tuple[np.ndarray, np.ndarray]:
    """SHAP ở margin space. Trả về (shap_values [n, n_features], base_values [n]).

    `explainer`: truyền vào instance đã dựng sẵn để khỏi dựng lại (đường serving).
    Bỏ trống thì dựng tại chỗ — tiện cho script phân tích chạy 1 lần.
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
    """SHAP + reason codes cho ĐÚNG 1 applicant.

    Tự reindex `row` theo `feature_names` trước khi predict/explain — KHÔNG tin
    thứ tự cột của `row` do caller truyền vào, vì Booster khớp cột theo vị trí
    (xem docstring đầu file). Đây là validate-tại-biên, chặn cả lớp bug skew.
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

    print(f"Tính SHAP cho {len(X_sample)} dòng mẫu ({len(feature_names)} feature)...")
    shap_values, _ = compute_shap_values(model, X_sample)

    importance = global_importance(shap_values, feature_names)
    print("\n=== TOP 20 feature quan trọng nhất (SHAP global mean|value|) ===")
    print(importance.head(20).to_string(index=False))

    with open(FEATURES_CONFIG_PATH) as f:
        declared = yaml.safe_load(f).get("monotone_constraints", {})
    top30 = set(importance.head(30)["feature"])
    overlap = top30 & set(declared)
    print(f"\nOverlap top-30 SHAP vs {len(declared)} feature có monotonic constraint (Block 3): "
          f"{len(overlap)} feature -> {sorted(overlap)}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    importance.to_csv(ARTIFACTS_DIR / "shap_global_importance.csv", index=False)
    print(f"\nSaved -> {ARTIFACTS_DIR / 'shap_global_importance.csv'}")

    print("\n=== Demo reason codes: 3 applicant rủi ro cao nhất trong mẫu ===")
    raw_pred = model.predict(X_sample, raw_score=True)
    top_risk_pos = np.argsort(-raw_pred)[:3]
    for pos in top_risk_pos:
        sk_id = int(sample.iloc[pos]["SK_ID_CURR"])
        actual = int(sample.iloc[pos]["TARGET"]) if "TARGET" in sample.columns else None
        reasons = build_reason_codes(feature_names, shap_values[pos], X_sample.iloc[pos].values, top_k=5)
        print(f"\nSK_ID_CURR={sk_id}  raw_margin={raw_pred[pos]:.3f}  TARGET thật={actual}")
        for r in reasons:
            flag = "" if r["curated"] else "  [fallback, cần pháp chế review]"
            print(f"  - {r['label']}: {r['direction']} (shap={r['shap']:+.3f}){flag}")


if __name__ == "__main__":
    main()
