"""Filter the 7 raw tables down to a single applicant, the input to
FeaturePipeline.transform().

bureau_balance only has SK_ID_BUREAU (no SK_ID_CURR), so it has to be filtered
indirectly through that applicant's set of SK_ID_BUREAU values, following the same
two-level chain used during training (see ml/src/features/bureau.py).
"""
from __future__ import annotations

import pandas as pd

from .source import RawTableStore


def assemble_applicant_tables(store: RawTableStore, sk_id_curr: int) -> dict[str, pd.DataFrame]:
    app = store.tables["application"]
    application = app[app["SK_ID_CURR"] == sk_id_curr].reset_index(drop=True)
    if application.empty:
        raise KeyError(f"SK_ID_CURR={sk_id_curr} is not in the applicant pool")

    bureau = store.tables["bureau"]
    bureau_rows = bureau[bureau["SK_ID_CURR"] == sk_id_curr].reset_index(drop=True)

    bureau_ids = set(bureau_rows["SK_ID_BUREAU"])
    bb = store.tables["bureau_balance"]
    bureau_balance_rows = bb[bb["SK_ID_BUREAU"].isin(bureau_ids)].reset_index(drop=True)

    def _filter_by_curr(table_name: str) -> pd.DataFrame:
        t = store.tables[table_name]
        return t[t["SK_ID_CURR"] == sk_id_curr].reset_index(drop=True)

    return {
        "application": application,
        "bureau": bureau_rows,
        "bureau_balance": bureau_balance_rows,
        "previous_application": _filter_by_curr("previous_application"),
        "pos_cash": _filter_by_curr("pos_cash"),
        "installments": _filter_by_curr("installments"),
        "credit_card": _filter_by_curr("credit_card"),
    }
