import json
import math

import pytest

from backend.app.data.assembler import assemble_applicant_tables
from backend.app.scorer import risk_tier
from backend.tests.conftest import APPLICANT_NO_HISTORY, APPLICANT_WITH_HISTORY


def test_score_returns_calibrated_probability_in_valid_range(real_scorer, synthetic_store):
    tables = assemble_applicant_tables(synthetic_store, sk_id_curr=APPLICANT_WITH_HISTORY)
    result = real_scorer.score(tables)

    assert 0.0 <= result["pd_score"] <= 1.0
    assert 0.0 <= result["pd_uncalibrated"] <= 1.0
    assert result["sk_id_curr"] == APPLICANT_WITH_HISTORY
    assert result["risk_tier"] in ("Thấp", "Trung bình", "Cao")


def test_score_raw_margin_matches_uncalibrated_probability_via_sigmoid(real_scorer, synthetic_store):
    """base_value/raw_margin dùng để vẽ waterfall ở frontend — phải nhất quán với
    pd_uncalibrated qua sigmoid (LightGBM objective='binary'), không phải số rời rạc."""
    tables = assemble_applicant_tables(synthetic_store, sk_id_curr=APPLICANT_WITH_HISTORY)
    result = real_scorer.score(tables)

    sigmoid = 1.0 / (1.0 + math.exp(-result["raw_margin"]))
    assert sigmoid == pytest.approx(result["pd_uncalibrated"], abs=1e-6)


def test_score_reasons_are_json_serializable_even_with_categorical_top_feature(real_scorer, synthetic_store):
    """Guard cho bug đã fix: reason value có thể là string (categorical) — không được crash."""
    tables = assemble_applicant_tables(synthetic_store, sk_id_curr=APPLICANT_NO_HISTORY)
    result = real_scorer.score(tables)

    assert len(result["reasons"]) == 5
    json.dumps(result["reasons"])  # không được raise


def test_score_applicant_with_no_bureau_history_still_scores(real_scorer, synthetic_store):
    tables = assemble_applicant_tables(synthetic_store, sk_id_curr=APPLICANT_NO_HISTORY)
    result = real_scorer.score(tables)
    assert result["pd_score"] is not None


def test_score_is_deterministic_for_same_applicant(real_scorer, synthetic_store):
    tables = assemble_applicant_tables(synthetic_store, sk_id_curr=APPLICANT_WITH_HISTORY)
    result_a = real_scorer.score(tables)
    result_b = real_scorer.score(tables)
    assert result_a["pd_score"] == result_b["pd_score"]


def test_risk_tier_bands_relative_to_base_rate():
    base_rate = 0.08
    assert risk_tier(0.01, base_rate) == "Thấp"
    assert risk_tier(0.08, base_rate) == "Trung bình"
    assert risk_tier(0.30, base_rate) == "Cao"
