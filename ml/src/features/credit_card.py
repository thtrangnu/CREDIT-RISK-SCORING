"""credit_card_balance (sao kê thẻ theo tháng) -> 1 dòng/SK_ID_CURR."""
from __future__ import annotations

import pandas as pd

from .aggregations import aggregate_categorical, aggregate_numeric, group_size


CC_CAT_COLS = ["NAME_CONTRACT_STATUS"]


def aggregate_credit_card(cc: pd.DataFrame, categories: dict[str, list] | None = None) -> pd.DataFrame:
    num_agg = aggregate_numeric(cc, "SK_ID_CURR", "CC", exclude=("SK_ID_PREV",))
    cat_agg = aggregate_categorical(
        cc, "SK_ID_CURR", "CC", cat_cols=CC_CAT_COLS, categories=categories
    )
    cnt = group_size(cc, "SK_ID_CURR", "CC", out_name="RECORDS")
    n_prev = cc.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique().rename("CC_NUNIQUE_PREV").to_frame()
    return pd.concat([cnt, n_prev, num_agg, cat_agg], axis=1)
