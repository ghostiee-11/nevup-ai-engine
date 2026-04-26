"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-26
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

    op.create_table(
        "traders",
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("profile", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("ground_truth_pathologies", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("description", sa.Text, nullable=True),
    )

    op.create_table(
        "sessions",
        sa.Column("session_id", sa.dialects.postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("traders.user_id"), nullable=False),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("trade_count", sa.Integer, nullable=False),
        sa.Column("win_rate", sa.Float, nullable=False),
        sa.Column("total_pnl", sa.Float, nullable=False),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("raw", sa.JSON, nullable=False),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "trades",
        sa.Column("trade_id", sa.dialects.postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("session_id", sa.dialects.postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("sessions.session_id"), nullable=False),
        sa.Column("asset", sa.String(32), nullable=False),
        sa.Column("asset_class", sa.String(16), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("exit_price", sa.Float, nullable=True),
        sa.Column("quantity", sa.Float, nullable=False),
        sa.Column("entry_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("outcome", sa.String(16), nullable=True),
        sa.Column("pnl", sa.Float, nullable=True),
        sa.Column("plan_adherence", sa.Integer, nullable=True),
        sa.Column("emotional_state", sa.String(16), nullable=True),
        sa.Column("entry_rationale", sa.Text, nullable=True),
        sa.Column("revenge_flag", sa.Boolean, server_default=sa.false()),
    )
    op.create_index("ix_trades_user_id", "trades", ["user_id"])
    op.create_index("ix_trades_session_id", "trades", ["session_id"])
    op.create_index("ix_trades_entry_at", "trades", ["entry_at"])

    op.create_table(
        "session_summaries",
        sa.Column("session_id", sa.dialects.postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("metrics", sa.JSON, nullable=False),
        sa.Column("tags", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_session_summaries_user_id", "session_summaries", ["user_id"])
    op.execute(
        "CREATE INDEX ix_session_summaries_embedding ON session_summaries "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)"
    )


def downgrade() -> None:
    op.drop_table("session_summaries")
    op.drop_table("trades")
    op.drop_table("sessions")
    op.drop_table("traders")
