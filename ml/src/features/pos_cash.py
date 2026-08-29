"""POS_CASH_balance (monthly POS/cash history) -> one row per SK_ID_CURR."""
from __future__ import annotations

import pandas as pd

from .aggregations import aggregate_categorical, aggregate_numeric, group_size


POS_CAT_COLS = ["NAME_CONTRACT_STATUS"]


def aggregate_pos_cash(pos: pd.DataFrame, categories: dict[str, list] | None = None) -> pd.DataFrame:
    num_agg = aggregate_numeric(pos, "SK_ID_CURR", "POS", exclude=("SK_ID_PREV",))
    cat_agg = aggregate_categorical(
        pos, "SK_ID_CURR", "POS", cat_cols=POS_CAT_COLS, categories=categories
    )
    cnt = group_size(pos, "SK_ID_CURR", "POS", out_name="RECORDS")
    n_prev = pos.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique().rename("POS_NUNIQUE_PREV").to_frame()
    return pd.concat([cnt, n_prev, num_agg, cat_agg], axis=1)
