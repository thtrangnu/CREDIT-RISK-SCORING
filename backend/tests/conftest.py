"""Fixture dùng 2 applicant THẬT (SK_ID_CURR 100002, 100006), lọc trực tiếp
từ 7 CSV thật — KHÔNG tự tay dựng DataFrame synthetic.

Lý do: FeaturePipeline thật fit trên toàn bộ data thật (mọi cột categorical
CỐ ĐỊNH lúc fit, mọi cột numeric được aggregate_numeric tự suy ra từ CHÍNH df
truyền vào). Tự dựng bảng tay thiếu bất kỳ cột nào (categorical HAY numeric)
đều làm feature_names.json lệch với cột thật sự tính được -> KeyError khi
`flat[feature_names]` — đã gặp bug này 2 lần (thiếu cột categorical app, rồi
thiếu cột numeric bureau/prev/pos/instal/cc) trước khi đổi sang cách này.

Fixture session-scoped nên chi phí quét 2.5GB CSV (đọc theo chunk, chỉ giữ
dòng khớp SK_ID_CURR) chỉ trả 1 lần cho cả lần chạy test, không phải mỗi test.
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

APPLICANT_WITH_HISTORY = 100002  # TARGET=1, có bureau + previous_application + pos + instalments
APPLICANT_NO_HISTORY = 100006  # TARGET=0, không có lịch sử ở 5 bảng phụ

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
    """Artifact THẬT (model.txt/feature_pipeline.pkl/calibrator.pkl) — Block 2-5 phải
    đã chạy xong (ml/artifacts/ tồn tại) để fixture này pass."""
    return Scorer(ARTIFACTS_DIR, model_version="test-version")
