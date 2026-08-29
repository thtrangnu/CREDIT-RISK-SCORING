"""create scoring_history

Revision ID: 0001
Revises:
Create Date: 2026-08-10
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scoring_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("sk_id_curr", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("pd_uncalibrated", sa.Numeric(8, 6), nullable=False),
        sa.Column("pd_score", sa.Numeric(8, 6), nullable=False),
        sa.Column("risk_tier", sa.String(length=16), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("scored_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_scoring_history_sk_id_curr", "scoring_history", ["sk_id_curr"])


def downgrade() -> None:
    op.drop_index("ix_scoring_history_sk_id_curr", table_name="scoring_history")
    op.drop_table("scoring_history")
