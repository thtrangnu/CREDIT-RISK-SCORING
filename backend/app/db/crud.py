from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import ScoringHistory


def create_scoring_record(db: Session, score_result: dict) -> ScoringHistory:
    record = ScoringHistory(
        sk_id_curr=score_result["sk_id_curr"],
        model_version=score_result["model_version"],
        pd_uncalibrated=score_result["pd_uncalibrated"],
        pd_score=score_result["pd_score"],
        risk_tier=score_result["risk_tier"],
        reasons=score_result["reasons"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_history(db: Session, sk_id_curr: int | None = None, limit: int = 50, offset: int = 0) -> list[ScoringHistory]:
    stmt = select(ScoringHistory).order_by(ScoringHistory.scored_at.desc())
    if sk_id_curr is not None:
        stmt = stmt.where(ScoringHistory.sk_id_curr == sk_id_curr)
    stmt = stmt.limit(limit).offset(offset)
    return list(db.execute(stmt).scalars().all())


def count_history(db: Session, sk_id_curr: int | None = None) -> int:
    stmt = select(func.count()).select_from(ScoringHistory)
    if sk_id_curr is not None:
        stmt = stmt.where(ScoringHistory.sk_id_curr == sk_id_curr)
    return db.execute(stmt).scalar_one()
