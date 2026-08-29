import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..data.assembler import assemble_applicant_tables
from ..data.source import RawTableStore
from ..db.crud import create_scoring_record
from ..db.session import get_db
from ..dependencies import get_scorer, get_store
from ..schemas import ApplicantSummary, ScoreResponse
from ..scorer import Scorer

router = APIRouter(prefix="/api", tags=["score"])


@router.get("/applicants", response_model=list[ApplicantSummary])
def search_applicants(q: str | None = None, limit: int = 20, store: RawTableStore = Depends(get_store)):
    limit = max(1, min(limit, 100))
    df = store.search_applicants(q, limit=limit)
    return [
        ApplicantSummary(
            sk_id_curr=int(row.SK_ID_CURR),
            target=None if pd.isna(row.TARGET) else int(row.TARGET),
            code_gender=row.CODE_GENDER,
            amt_income_total=float(row.AMT_INCOME_TOTAL),
            amt_credit=float(row.AMT_CREDIT),
            name_education_type=row.NAME_EDUCATION_TYPE,
            name_family_status=row.NAME_FAMILY_STATUS,
        )
        for row in df.itertuples()
    ]


@router.post("/score/{sk_id_curr}", response_model=ScoreResponse)
def score_applicant(
    sk_id_curr: int,
    store: RawTableStore = Depends(get_store),
    scorer: Scorer = Depends(get_scorer),
    db: Session = Depends(get_db),
):
    if not store.exists(sk_id_curr):
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR={sk_id_curr} không tồn tại")

    tables = assemble_applicant_tables(store, sk_id_curr)
    target_raw = tables["application"].iloc[0].get("TARGET")
    target_actual = None if target_raw is None or pd.isna(target_raw) else int(target_raw)

    result = scorer.score(tables)
    create_scoring_record(db, result)

    return ScoreResponse(**result, target_actual=target_actual)
