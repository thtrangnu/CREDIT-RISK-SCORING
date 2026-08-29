import numpy as np
import pandas as pd
import pytest

from ml.src.features.aggregations import (
    aggregate_categorical,
    aggregate_numeric,
    group_size,
    infer_categories,
    merge_all,
)


@pytest.fixture
def numeric_df():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2, 2, 2],
        "AMT": [100.0, 300.0, 10.0, 20.0, 30.0],
    })


def test_aggregate_numeric_matches_manual_computation(numeric_df):
    agg = aggregate_numeric(numeric_df, "SK_ID_CURR", "X", funcs=("mean", "sum", "count"))

    assert agg.loc[1, "X_AMT_MEAN"] == 200.0
    assert agg.loc[1, "X_AMT_SUM"] == 400.0
    assert agg.loc[1, "X_AMT_COUNT"] == 2
    assert agg.loc[2, "X_AMT_MEAN"] == pytest.approx(20.0)
    assert agg.loc[2, "X_AMT_SUM"] == 60.0
    assert agg.loc[2, "X_AMT_COUNT"] == 3


def test_aggregate_numeric_excludes_group_key_and_excluded_cols():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_BUREAU": [10, 11],
        "AMT": [1.0, 2.0],
    })
    agg = aggregate_numeric(df, "SK_ID_CURR", "X", exclude=("SK_ID_BUREAU",))
    assert not any("SK_ID_BUREAU" in c for c in agg.columns)
    assert not any("SK_ID_CURR" in c for c in agg.columns)


def test_group_size_counts_rows_per_group():
    df = pd.DataFrame({"SK_ID_CURR": [1, 1, 1, 2]})
    out = group_size(df, "SK_ID_CURR", "BUREAU")
    assert out.loc[1, "BUREAU_COUNT"] == 3
    assert out.loc[2, "BUREAU_COUNT"] == 1


def test_infer_categories_is_sorted_and_drops_na():
    df = pd.DataFrame({"STATUS": ["b", "a", None, "a", "c"]})
    cats = infer_categories(df, ["STATUS"])
    assert cats == {"STATUS": ["a", "b", "c"]}


def test_aggregate_categorical_shares_sum_to_one_per_group():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1, 2, 2],
        "STATUS": ["Active", "Active", "Closed", "Closed", "Closed"],
    })
    agg = aggregate_categorical(df, "SK_ID_CURR", "BUREAU", cat_cols=["STATUS"])
    share_cols = [c for c in agg.columns if c.startswith("BUREAU_STATUS")]
    row_sums = agg[share_cols].sum(axis=1)
    assert row_sums.tolist() == pytest.approx([1.0, 1.0])
    assert agg.loc[1, "BUREAU_STATUS_Active_SHARE"] == pytest.approx(2 / 3)
    assert agg.loc[2, "BUREAU_STATUS_Closed_SHARE"] == pytest.approx(1.0)


def test_aggregate_categorical_with_fixed_categories_keeps_full_column_set():
    """Anti train-serve skew: an applicant with only one category must still emit every dummy column."""
    fit_domain = {"STATUS": ["Active", "Bad_debt", "Closed"]}

    serve_df = pd.DataFrame({"SK_ID_CURR": [99], "STATUS": ["Closed"]})
    agg = aggregate_categorical(
        serve_df, "SK_ID_CURR", "BUREAU", cat_cols=["STATUS"], categories=fit_domain
    )

    assert set(agg.columns) == {
        "BUREAU_STATUS_Active_SHARE",
        "BUREAU_STATUS_Bad_debt_SHARE",
        "BUREAU_STATUS_Closed_SHARE",
    }
    assert agg.loc[99, "BUREAU_STATUS_Active_SHARE"] == 0.0
    assert agg.loc[99, "BUREAU_STATUS_Closed_SHARE"] == 1.0


def test_merge_all_leaves_nan_when_applicant_has_no_history():
    base = pd.DataFrame({"SK_ID_CURR": [1, 2], "AGE": [30, 40]})
    history = pd.DataFrame({"SK_ID_CURR": [1], "X_AMT_MEAN": [500.0]}).set_index("SK_ID_CURR")

    merged = merge_all(base, "SK_ID_CURR", history)

    assert merged.loc[merged.SK_ID_CURR == 1, "X_AMT_MEAN"].item() == 500.0
    assert np.isnan(merged.loc[merged.SK_ID_CURR == 2, "X_AMT_MEAN"].item())
