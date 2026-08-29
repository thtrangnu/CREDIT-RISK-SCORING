"""previous_application (đơn vay HC trước đó) -> 1 dòng/SK_ID_CURR."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregations import aggregate_categorical, aggregate_numeric, group_size

# 365243 là sentinel "không áp dụng" của Home Credit cho các cột DAYS_* này.
DAYS_SENTINEL_COLS = (
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
)


# Cardinality thấp/vừa, đủ tín hiệu — bỏ các cột cardinality cao (PRODUCT_COMBINATION,
# NAME_GOODS_CATEGORY, ...) để tránh nổ số cột dummy (MacBook 8GB RAM, xem docs/NOTES.md).
PREV_CAT_COLS = [
    "NAME_CONTRACT_TYPE",
    "NAME_CONTRACT_STATUS",
    "NAME_CLIENT_TYPE",
    "NAME_PORTFOLIO",
    "NAME_YIELD_GROUP",
    "CHANNEL_TYPE",
    "FLAG_LAST_APPL_PER_CONTRACT",
]


def aggregate_previous_application(
    prev: pd.DataFrame, categories: dict[str, list] | None = None
) -> pd.DataFrame:
    df = prev.copy()
    for c in DAYS_SENTINEL_COLS:
        if c in df.columns:
            df[c] = df[c].replace(365243, np.nan)

    num_agg = aggregate_numeric(df, "SK_ID_CURR", "PREV", exclude=("SK_ID_PREV",))
    cat_agg = aggregate_categorical(
        df, "SK_ID_CURR", "PREV", cat_cols=PREV_CAT_COLS, categories=categories
    )
    cnt = group_size(df, "SK_ID_CURR", "PREV")
    return pd.concat([cnt, num_agg, cat_agg], axis=1)
