"""Block 4-5: recalibration (isotonic / Platt) + final production artifacts.

Dùng sklearn cho MODEL hiệu chỉnh (IsotonicRegression, LogisticRegression) — đây
là component modeling, không phải metric. Đo lường trước/sau LUÔN dùng
metrics.py tự viết (brier_score, expected_calibration_error), giữ đúng tinh
thần "metric tự viết" của docs/NOTES.md.

Đánh giá bằng nested calibration (K-fold trên chính OOF): fit calibrator trên
K-1 phần OOF, dự đoán phần còn lại — tránh đo lạc quan ảo do calibrator học
lại chính label của điểm nó đang hiệu chỉnh.
"""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from .features.build import ARTIFACTS_DIR
from .metrics import brier_score, expected_calibration_error, roc_auc
from .train_engineered import (CACHED_FEATURES_PATH, CONFIG_PATH, FEATURES_CONFIG_PATH,
                                build_monotone_constraints, load_config, load_data)

METHODS = ("isotonic", "sigmoid")


def fit_calibrator(score: np.ndarray, y: np.ndarray, method: str):
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(score, y)
        return model
    if method == "sigmoid":
        model = LogisticRegression()
        model.fit(score.reshape(-1, 1), y)
        return model
    raise ValueError(f"method không hỗ trợ: {method}")


def predict_calibrated(model, score: np.ndarray, method: str) -> np.ndarray:
    if method == "isotonic":
        return model.predict(score)
    if method == "sigmoid":
        return model.predict_proba(score.reshape(-1, 1))[:, 1]
    raise ValueError(f"method không hỗ trợ: {method}")


def nested_calibrate(oof_pred: np.ndarray, y: np.ndarray, method: str,
                      n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Ước lượng không lạc quan: mỗi điểm được hiệu chỉnh bởi calibrator KHÔNG thấy label của nó."""
    calibrated = np.zeros_like(oof_pred, dtype=float)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for tr, va in skf.split(oof_pred, y):
        model = fit_calibrator(oof_pred[tr], y[tr], method)
        calibrated[va] = predict_calibrated(model, oof_pred[va], method)
    return calibrated


def pick_final_num_boost_round(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str],
                                params: dict, early_stopping_rounds: int,
                                num_boost_round_max: int, seed: int) -> int:
    """1 split early-stopping riêng chỉ để chọn số vòng cho model full-data cuối cùng."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=seed
    )
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_val, y_val, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=num_boost_round_max, valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)],
    )
    return model.best_iteration


def train_final_model(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str],
                       params: dict, num_boost_round: int) -> lgb.Booster:
    dtrain = lgb.Dataset(X, y, categorical_feature=cat_cols)
    return lgb.train(params, dtrain, num_boost_round=num_boost_round)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    n_folds, seed, lgbm_cfg = cfg["n_folds"], cfg["seed"], cfg["lgbm"]
    num_boost_round = lgbm_cfg["num_boost_round"]
    early_stopping_rounds = lgbm_cfg["early_stopping_rounds"]
    params = {k: v for k, v in lgbm_cfg.items()
              if k not in ("num_boost_round", "early_stopping_rounds")}
    params["seed"] = seed

    declared_constraints = load_config(FEATURES_CONFIG_PATH).get("monotone_constraints", {})

    df = pd.read_parquet(CACHED_FEATURES_PATH)
    X, y, cat_cols = load_data(df)
    oof = np.load(ARTIFACTS_DIR / "oof_engineered.npy")
    assert len(oof) == len(y), "oof_engineered.npy không khớp số dòng feature hiện tại"

    print("=== TRƯỚC hiệu chỉnh (raw OOF) ===")
    print(f"AUC   : {roc_auc(y, oof):.5f}  (hiệu chỉnh không đổi thứ hạng/AUC)")
    print(f"Brier : {brier_score(y, oof):.5f}")
    print(f"ECE(10): {expected_calibration_error(y, oof):.5f}")

    print("\n=== SAU hiệu chỉnh (nested CV, không lạc quan) ===")
    results = {}
    for method in METHODS:
        calibrated = nested_calibrate(oof, y, method, n_folds=n_folds, seed=seed)
        brier, ece = brier_score(y, calibrated), expected_calibration_error(y, calibrated)
        results[method] = (brier, ece)
        print(f"{method:8s}  Brier={brier:.5f}  ECE(10)={ece:.5f}")

    best_method = min(results, key=lambda m: results[m][1])
    print(f"\n-> Chọn '{best_method}' (ECE thấp nhất)")

    final_calibrator = fit_calibrator(oof, y, best_method)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(ARTIFACTS_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump({"method": best_method, "model": final_calibrator}, f)

    monotone_constraints = build_monotone_constraints(list(X.columns), set(cat_cols), declared_constraints)
    params_final = {**params, "monotone_constraints": monotone_constraints}
    final_rounds = pick_final_num_boost_round(
        X, y, cat_cols, params_final, early_stopping_rounds, num_boost_round, seed
    )
    print(f"\nFinal model: num_boost_round={final_rounds} (chọn qua 1 split early-stopping riêng)")
    final_model = train_final_model(X, y, cat_cols, params_final, final_rounds)
    final_model.save_model(str(ARTIFACTS_DIR / "model.txt"))

    # model_card.md KHÔNG sinh ở đây nữa. Trước đây template card nằm inline trong
    # hàm này, nghĩa là sửa một câu chữ trong card cũng phải chạy lại toàn bộ
    # training. Giờ card là hàm thuần của artifact -> ml/src/export_model_card.py.
    print(f"\nSaved calibrator.pkl + model.txt -> {ARTIFACTS_DIR}")
    print("Bước tiếp: python -m ml.src.export_metrics_summary && python -m ml.src.export_model_card")


if __name__ == "__main__":
    main()
