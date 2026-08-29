from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db import crud
from ..db.session import get_db
from ..schemas import HistoryEntry, HistoryResponse

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history", response_model=HistoryResponse)
def get_history(
    sk_id_curr: int | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    items = crud.list_history(db, sk_id_curr=sk_id_curr, limit=limit, offset=offset)
    total = crud.count_history(db, sk_id_curr=sk_id_curr)
    return HistoryResponse(
        items=[HistoryEntry.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )
