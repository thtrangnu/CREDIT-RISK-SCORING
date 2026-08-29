import numpy as np

from ml.src.reason_codes import build_reason_codes, direction_phrase, humanize


def test_humanize_curated_app_feature():
    label, curated = humanize("EXT_SOURCE_1")
    assert curated is True
    assert "tín dụng ngoài" in label


def test_humanize_curated_multitable_feature_combines_agg_and_table_label():
    label, curated = humanize("BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM")
    assert curated is True
    assert "tổng" in label
    assert "quá hạn" in label
    assert "tổ chức khác" in label


def test_humanize_fallback_for_uncurated_raw_column_still_readable():
    label, curated = humanize("BUREAU_CNT_CREDIT_PROLONG_MAX")
    assert curated is False
    assert "cao nhất" in label
    assert "cnt credit prolong" in label


def test_humanize_fallback_for_unknown_app_feature():
    label, curated = humanize("SOME_RANDOM_APP_COLUMN")
    assert curated is False
    assert label == "Some Random App Column"


def test_direction_phrase_matches_shap_sign():
    assert "TĂNG" in direction_phrase(0.5)
    assert "GIẢM" in direction_phrase(-0.5)


def test_build_reason_codes_returns_top_k_sorted_by_absolute_shap():
    feature_names = ["A", "B", "C", "D"]
    shap_row = np.array([0.1, -0.9, 0.05, 0.5])
    values = np.array([1.0, 2.0, 3.0, 4.0])

    reasons = build_reason_codes(feature_names, shap_row, values, top_k=2)

    assert [r["feature"] for r in reasons] == ["B", "D"]
    assert reasons[0]["direction"] == "làm GIẢM rủi ro"
    assert reasons[1]["direction"] == "làm TĂNG rủi ro"


def test_build_reason_codes_handles_categorical_string_value_without_crashing():
    """Cột categorical gốc (CODE_GENDER, ORGANIZATION_TYPE...) hoàn toàn có thể lọt
    top-K SHAP của 1 applicant cụ thể — giá trị là string, KHÔNG được ép float()."""
    feature_names = ["CODE_GENDER", "EXT_SOURCE_1"]
    shap_row = np.array([0.4, 0.1])
    values = np.array(["F", 0.65], dtype=object)

    reasons = build_reason_codes(feature_names, shap_row, values, top_k=2)

    assert reasons[0]["feature"] == "CODE_GENDER"
    assert reasons[0]["value"] == "F"


def test_build_reason_codes_converts_nan_value_to_none():
    feature_names = ["BUREAU_AMT_CREDIT_SUM_MEAN"]
    shap_row = np.array([0.2])
    values = np.array([np.nan])

    reasons = build_reason_codes(feature_names, shap_row, values, top_k=1)
    assert reasons[0]["value"] is None


def test_build_reason_codes_respects_top_k_bound():
    feature_names = ["A", "B", "C"]
    shap_row = np.array([0.1, 0.2, 0.3])
    values = np.array([1.0, 2.0, 3.0])

    reasons = build_reason_codes(feature_names, shap_row, values, top_k=10)
    assert len(reasons) == 3
