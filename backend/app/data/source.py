"""Load the 7 Home Credit tables ONCE at backend startup.

Static Kaggle data, single-user demo (docs/NOTES.md section 1). Reference data does not
need a database; holding it in RAM for the process lifetime is sufficient and simpler
than loading it into MySQL. MySQL (see app/db/) is only for the audit trail, one row per
scoring call, kept separate from this static reference data.

Applicant pool = application_train (it carries the real TARGET, useful for comparison on
the Insights page). TARGET is never fed into the feature transform.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_FILES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}


class RawTableStore:
    """Holds the 7 raw tables in RAM for the backend process lifetime."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.tables = tables

    @classmethod
    def load(cls, data_dir: Path) -> "RawTableStore":
        tables = {name: pd.read_csv(data_dir / fname) for name, fname in RAW_FILES.items()}
        return cls(tables)

    def exists(self, sk_id_curr: int) -> bool:
        return bool((self.tables["application"]["SK_ID_CURR"] == sk_id_curr).any())

    # Empty query -> return a FIXED sample (seed 42) rather than a fresh random one on
    # every call. The search box debounces per keystroke, so a changing sample would make
    # the dropdown jump around.
    SUGGESTION_SEED = 42

    def search_applicants(self, query: str | None, limit: int = 20) -> pd.DataFrame:
        app = self.tables["application"]
        if query:
            mask = app["SK_ID_CURR"].astype(str).str.contains(query.strip(), regex=False)
            result = app[mask]
        else:
            result = app.sample(n=min(limit, len(app)), random_state=self.SUGGESTION_SEED)
        cols = ["SK_ID_CURR", "TARGET", "CODE_GENDER", "AMT_INCOME_TOTAL",
                "AMT_CREDIT", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"]
        return result[cols].head(limit).reset_index(drop=True)
