import pytest

from backend.app.data.assembler import assemble_applicant_tables
from backend.tests.conftest import APPLICANT_NO_HISTORY, APPLICANT_WITH_HISTORY


def test_assemble_filters_all_tables_to_single_applicant(synthetic_store):
    tables = assemble_applicant_tables(synthetic_store, sk_id_curr=APPLICANT_WITH_HISTORY)

    assert len(tables["application"]) == 1
    assert (tables["bureau"]["SK_ID_CURR"] == APPLICANT_WITH_HISTORY).all()
    assert (tables["previous_application"]["SK_ID_CURR"] == APPLICANT_WITH_HISTORY).all()


def test_assemble_follows_bureau_balance_chain_via_sk_id_bureau(synthetic_store):
    """bureau_balance has no SK_ID_CURR, so it must be filtered indirectly through that
    applicant's SK_ID_BUREAU values, without pulling in bureau_balance history belonging to
    another applicant's credits."""
    tables = assemble_applicant_tables(synthetic_store, sk_id_curr=APPLICANT_WITH_HISTORY)
    assert len(tables["bureau_balance"]) > 0
    assert set(tables["bureau_balance"]["SK_ID_BUREAU"]) <= set(tables["bureau"]["SK_ID_BUREAU"])


def test_assemble_applicant_with_no_bureau_history_returns_empty_bureau_frames(synthetic_store):
    tables = assemble_applicant_tables(synthetic_store, sk_id_curr=APPLICANT_NO_HISTORY)
    assert len(tables["bureau"]) == 0
    assert len(tables["bureau_balance"]) == 0


def test_assemble_unknown_applicant_raises_key_error(synthetic_store):
    with pytest.raises(KeyError):
        assemble_applicant_tables(synthetic_store, sk_id_curr=99999)
