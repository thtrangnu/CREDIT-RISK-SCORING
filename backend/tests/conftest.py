"""Fixtures built from 2 REAL applicants (SK_ID_CURR 100002 and 100006), filtered
straight out of the real CSVs. No hand-built synthetic DataFrames.

Why: the real FeaturePipeline is fit on the full real dataset (every categorical column
is FIXED at fit time, every numeric column is inferred by aggregate_numeric from the df
it was given). A hand-built table missing any column, categorical OR numeric, makes
feature_names.json disagree with what can actually be computed, which raises KeyError at
`flat[feature_names]`. That happened twice (first a missing app categorical column, then
missing numeric columns from bureau/prev/pos/instal/cc) before switching to this approach.

The fixture is session-scoped, so the cost of scanning 2.5GB of CSV (read in chunks,
keeping only rows matching SK_ID_CURR) is paid once per test run rather than per test.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import ARTIFACTS_DIR, DATA_DIR  # noqa: E402
from backend.app.data.source import RAW_FILES, RawTableStore  # noqa: E402
from backend.app.scorer import Scorer  # noqa: E402

APPLICANT_WITH_HISTORY = 100002  # TARGET=1, has bureau + previous_application + pos + installments
APPLICANT_NO_HISTORY = 100006  # TARGET=0, no history in the 5 satellite tables

CHUNK_SIZE = 200_000


def _read_filtered(path: Path, curr_ids: set[int], curr_col: str = "SK_ID_CURR") -> pd.DataFrame:
    chunks = [
        chunk[chunk[curr_col].isin(curr_ids)]
        for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE)
    ]
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _real_applicant_tables(curr_ids: set[int]) -> dict[str, pd.DataFrame]:
    application = _read_filtered(DATA_DIR / RAW_FILES["application"], curr_ids)
    bureau = _read_filtered(DATA_DIR / RAW_FILES["bureau"], curr_ids)
    bureau_ids = set(bureau["SK_ID_BUREAU"])
    bureau_balance = _read_filtered(DATA_DIR / RAW_FILES["bureau_balance"], bureau_ids, curr_col="SK_ID_BUREAU")

    return {
        "application": application,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": _read_filtered(DATA_DIR / RAW_FILES["previous_application"], curr_ids),
        "pos_cash": _read_filtered(DATA_DIR / RAW_FILES["pos_cash"], curr_ids),
        "installments": _read_filtered(DATA_DIR / RAW_FILES["installments"], curr_ids),
        "credit_card": _read_filtered(DATA_DIR / RAW_FILES["credit_card"], curr_ids),
    }


@pytest.fixture(scope="session")
def synthetic_tables() -> dict[str, pd.DataFrame]:
    return _real_applicant_tables({APPLICANT_WITH_HISTORY, APPLICANT_NO_HISTORY})


@pytest.fixture(scope="session")
def synthetic_store(synthetic_tables) -> RawTableStore:
    return RawTableStore(synthetic_tables)


@pytest.fixture(scope="session")
def real_scorer() -> Scorer:
    """The REAL artifacts (model.txt/feature_pipeline.pkl/calibrator.pkl). Blocks 2-5 must
    already have run (ml/artifacts/ must exist) for this fixture to work."""
    return Scorer(ARTIFACTS_DIR, model_version="test-version")
