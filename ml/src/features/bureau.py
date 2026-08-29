"""bureau + bureau_balance -> 1 dòng/SK_ID_CURR.

2-level aggregation (phần khó nhất của Block 2):
  bureau_balance (grain = SK_ID_BUREAU x tháng)
    -> groupby SK_ID_BUREAU                                   [level 1]
    -> merge vào bureau (grain = SK_ID_BUREAU)
    -> groupby SK_ID_CURR                                     [level 2]
"""
from __future__ import annotations

import pandas as pd

from .aggregations import aggregate_categorical, aggregate_numeric, group_size

DPD_STATUSES = {"1", "2", "3", "4", "5"}  # '0'=đúng hạn, 'C'=đã đóng, 'X'=không rõ

# Domain đóng theo data dictionary Home Credit — hardcode thay vì suy ra từ data,
# để aggregate_bureau_balance luôn ra ĐÚNG bộ cột BB_STATUS_*_SHARE dù input là
# toàn bộ train hay chỉ lịch sử của 1 applicant lẻ lúc serve (chống skew).
BB_STATUS_CATEGORIES = ["0", "1", "2", "3", "4", "5", "C", "X"]


def aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Level 1: agg lịch sử tháng của mỗi khoản bureau -> 1 dòng/SK_ID_BUREAU."""
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
    """Level 2: gắn bb_agg vào bureau rồi agg về SK_ID_CURR -> 1 dòng/applicant.

    `categories`: domain cố định (lúc fit) cho BUREAU_CAT_COLS — xem
    `aggregations.infer_categories` / chống train-serve skew.
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
