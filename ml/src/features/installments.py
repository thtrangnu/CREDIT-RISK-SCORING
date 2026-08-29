"""installments_payments (finest grain, one row per payment) -> one row per SK_ID_CURR.

Two derived columns are added at ROW level BEFORE aggregating. This has to happen here:
these are the strongest repayment-behaviour signals, and aggregating AMT_INSTALMENT /
AMT_PAYMENT or DAYS_INSTALMENT / DAYS_ENTRY_PAYMENT separately cannot reconstruct the
lateness or shortfall of any individual payment.
  DAYS_LATE     = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT   (>0 = paid late)
  PAYMENT_DIFF  = AMT_INSTALMENT - AMT_PAYMENT           (>0 = underpaid)
"""
from __future__ import annotations

import pandas as pd

from .aggregations import aggregate_numeric, group_size


def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    df = installments.copy()
    df["DAYS_LATE"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]

    num_agg = aggregate_numeric(df, "SK_ID_CURR", "INSTAL", exclude=("SK_ID_PREV",))
    cnt = group_size(df, "SK_ID_CURR", "INSTAL", out_name="RECORDS")
    n_prev = df.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique().rename("INSTAL_NUNIQUE_PREV").to_frame()
    return pd.concat([cnt, n_prev, num_agg], axis=1)
