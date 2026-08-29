import datetime

from sqlalchemy import DateTime, Integer, JSON, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base


class ScoringHistory(Base):
    """Audit trail: one row per scoring call. model_version exists for governance, so you
    can tell which model scored a given applicant once newer models exist."""

    __tablename__ = "scoring_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sk_id_curr: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    pd_uncalibrated: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    pd_score: Mapped[float] = mapped_column(Numeric(8, 6), nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    reasons: Mapped[list] = mapped_column(JSON, nullable=False)
    scored_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )
