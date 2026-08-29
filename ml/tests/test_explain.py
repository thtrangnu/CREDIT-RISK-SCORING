import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from ml.src.explain import compute_shap_values, explain_applicant, global_importance


@pytest.fixture
def toy_model_and_data():
    """A small LightGBM model trained inside the test, so it does not depend on the big artifacts."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "RISKY_FEATURE": rng.uniform(0, 1, n),
        "SAFE_FEATURE": rng.uniform(0, 1, n),
        "NOISE": rng.uniform(0, 1, n),
    })
    y = (X["RISKY_FEATURE"] - X["SAFE_FEATURE"] + rng.normal(0, 0.05, n) > 0.2).astype(int)

    dtrain = lgb.Dataset(X, y)
    model = lgb.train({"objective": "binary", "verbosity": -1, "num_leaves": 7}, dtrain, num_boost_round=30)
    return model, X


def test_global_importance_ranks_features_by_mean_abs_shap():
    shap_values = np.array([
        [0.1, -0.9, 0.0],
        [0.2, 0.8, 0.0],
        [-0.1, -0.7, 0.0],
    ])
    importance = global_importance(shap_values, ["A", "B", "C"])

    assert importance.iloc[0]["feature"] == "B"
    assert importance.iloc[-1]["feature"] == "C"
    assert importance.iloc[-1]["mean_abs_shap"] == 0.0


def test_compute_shap_values_reconstructs_raw_margin_prediction(toy_model_and_data):
    """SHAP additivity: base_value + sum(shap) == the raw margin prediction."""
    model, X = toy_model_and_data
    shap_values, base_values = compute_shap_values(model, X)

    raw_pred = model.predict(X, raw_score=True)
    reconstructed = base_values + shap_values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, raw_pred, atol=1e-6)


def test_explain_applicant_flags_the_dominant_feature(toy_model_and_data):
    model, X = toy_model_and_data
    feature_names = list(X.columns)
    row = X.iloc[[0]]

    result = explain_applicant(model, feature_names, row, top_k=2)

    assert len(result["reasons"]) == 2
    top_feature = result["reasons"][0]["feature"]
    assert top_feature in ("RISKY_FEATURE", "SAFE_FEATURE")
    assert "NOISE" not in [r["feature"] for r in result["reasons"]]


def test_explain_applicant_is_robust_to_caller_passing_shuffled_column_order(toy_model_and_data):
    """Booster matches columns BY POSITION, not by name (verified by hand). explain_applicant
    reindexes by feature_names itself, so the result must NOT change when the caller passes a
    DataFrame with a different column order."""
    model, X = toy_model_and_data
    feature_names = list(X.columns)
    row_correct_order = X.iloc[[0]][feature_names]
    row_shuffled = X.iloc[[0]][list(reversed(feature_names))]

    result_correct = explain_applicant(model, feature_names, row_correct_order, top_k=3)
    result_shuffled = explain_applicant(model, feature_names, row_shuffled, top_k=3)

    assert result_correct["raw_margin"] == pytest.approx(result_shuffled["raw_margin"])
    assert result_correct["reasons"] == result_shuffled["reasons"]
