import numpy as np
import pandas as pd
import pytest

from ml.src.features.bureau import aggregate_bureau, aggregate_bureau_balance


def test_aggregate_bureau_balance_level1_dpd_flag():
    bb = pd.DataFrame({
        "SK_ID_BUREAU": [1, 1, 1, 2, 2],
        "MONTHS_BALANCE": [0, -1, -2, 0, -1],
        "STATUS": ["0", "1", "C", "0", "0"],
    })
    agg = aggregate_bureau_balance(bb)

    assert agg.loc[1, "BB_COUNT"] == 3
    assert agg.loc[1, "BB_DPD_FLAG_MEAN"] == pytest.approx(1 / 3)
    assert agg.loc[2, "BB_DPD_FLAG_MEAN"] == 0.0


def test_aggregate_bureau_two_level_propagates_bureau_balance_into_curr_grain():
    """2-level: bureau_balance -> SK_ID_BUREAU -> merge vào bureau -> SK_ID_CURR."""
    bureau = pd.DataFrame({
        "SK_ID_CURR": [100, 100, 200],
        "SK_ID_BUREAU": [1, 2, 3],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
        "CREDIT_CURRENCY": ["currency 1"] * 3,
        "CREDIT_TYPE": ["Consumer credit"] * 3,
        "AMT_CREDIT_SUM": [1000.0, 2000.0, 500.0],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": [1, 1, 2, 3],
        "MONTHS_BALANCE": [0, -1, 0, 0],
        "STATUS": ["1", "1", "0", "C"],
    })

    out = aggregate_bureau(bureau, bureau_balance)

    # Grain: đúng 1 dòng / SK_ID_CURR, không nổ dòng do merge 1-nhiều.
    assert out.index.name == "SK_ID_CURR"
    assert sorted(out.index.tolist()) == [100, 200]
    assert not out.index.duplicated().any()

    # Curr 100 có 2 khoản bureau (id 1, id 2).
    assert out.loc[100, "BUREAU_COUNT"] == 2
    # id 1 luôn DPD (mean=1.0), id 2 luôn không DPD (mean=0.0)
    # -> mean-của-mean ở mức curr = (1.0 + 0.0) / 2 = 0.5
    assert out.loc[100, "BUREAU_BB_DPD_FLAG_MEAN_MEAN"] == 0.5
    assert out.loc[100, "BUREAU_AMT_CREDIT_SUM_SUM"] == 3000.0


def test_aggregate_bureau_curr_with_no_bureau_balance_history_is_nan_not_zero():
    """applicant có bureau nhưng khoản đó không có dòng bureau_balance nào -> NaN, KHÔNG suy diễn thành 0."""
    bureau = pd.DataFrame({
        "SK_ID_CURR": [300],
        "SK_ID_BUREAU": [9],
        "CREDIT_ACTIVE": ["Active"],
        "CREDIT_CURRENCY": ["currency 1"],
        "CREDIT_TYPE": ["Consumer credit"],
        "AMT_CREDIT_SUM": [100.0],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": [1],
        "MONTHS_BALANCE": [0],
        "STATUS": ["0"],
    })

    out = aggregate_bureau(bureau, bureau_balance)
    assert np.isnan(out.loc[300, "BUREAU_BB_DPD_FLAG_MEAN_MEAN"])
