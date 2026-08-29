"""bureau + bureau_balance -> one row per SK_ID_CURR.

Two-level aggregation, the hardest part of Block 2:
  bureau_balance (grain = SK_ID_BUREAU x month)
    -> groupby SK_ID_BUREAU                                   [level 1]
    -> merge into bureau (grain = SK_ID_BUREAU)
    -> groupby SK_ID_CURR                                     [level 2]
"""
from __future__ import annotations

import pandas as pd

from .aggregations import aggregate_categorical, aggregate_numeric, group_size

DPD_STATUSES = {"1", "2", "3", "4", "5"}  # '0'=on time, 'C'=closed, 'X'=unknown

# A closed domain per the Home Credit data dictionary. Hardcoded rather than inferred
# from the data, so aggregate_bureau_balance always produces the SAME set of
# BB_STATUS_*_SHARE columns whether the input is all of train or the history of a single
# applicant at serving time (anti-skew).
BB_STATUS_CATEGORIES = ["0", "1", "2", "3", "4", "5", "C", "X"]


def aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Level 1: aggregate each bureau credit's monthly history -> one row per SK_ID_BUREAU."""
    bb = bureau_balance.copy()
    bb["DPD_FLAG"] = bb["STATUS"].isin(DPD_STATUSES).astype(int)

    num_agg = aggregate_numeric(bb, "SK_ID_BUREAU", "BB")
    cat_agg = aggregate_categorical(
        bb, "SK_ID_BUREAU", "BB", cat_cols=["STATUS"], categories={"STATUS": BB_STATUS_CATEGORIES}
    )
    cnt = group_size(bb, "SK_ID_BUREAU", "BB")
    return pd.concat([cnt, num_agg, cat_agg], axis=1)


BUREAU_CAT_COLS = ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]


def aggregate_bureau(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame | None = None,
    categories: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Level 2: attach bb_agg to bureau, then aggregate to SK_ID_CURR -> one row per applicant.

    `categories`: the fit-time domain for BUREAU_CAT_COLS. See
    `aggregations.infer_categories` and the training-serving skew note.
    """
    df = bureau.copy()
    if bureau_balance is not None:
        bb_agg = aggregate_bureau_balance(bureau_balance)
        df = df.merge(bb_agg, how="left", left_on="SK_ID_BUREAU", right_index=True)

    num_agg = aggregate_numeric(df, "SK_ID_CURR", "BUREAU", exclude=("SK_ID_BUREAU",))
    cat_agg = aggregate_categorical(
        df, "SK_ID_CURR", "BUREAU", cat_cols=BUREAU_CAT_COLS, categories=categories
    )
    cnt = group_size(df, "SK_ID_CURR", "BUREAU")
    return pd.concat([cnt, num_agg, cat_agg], axis=1)
