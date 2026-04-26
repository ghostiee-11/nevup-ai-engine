from datetime import datetime, timezone
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.db import Base


class Trader(Base):
    __tablename__ = "traders"
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    profile: Mapped[dict] = mapped_column(JSON, default=dict)
    ground_truth_pathologies: Mapped[list] = mapped_column(JSON, default=list)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Session(Base):
    __tablename__ = "sessions"
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("traders.user_id"), index=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    trade_count: Mapped[int] = mapped_column(Integer)
    win_rate: Mapped[float] = mapped_column(Float)
    total_pnl: Mapped[float] = mapped_column(Float)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw: Mapped[dict] = mapped_column(JSON)


class Trade(Base):
    __tablename__ = "trades"
    trade_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), index=True)
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("sessions.session_id"), index=True)
    asset: Mapped[str] = mapped_column(String(32))
    asset_class: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quantity: Mapped[float] = mapped_column(Float)
    entry_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    exit_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    outcome: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plan_adherence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    emotional_state: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    entry_rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    revenge_flag: Mapped[Optional[bool]] = mapped_column(default=False)


class SessionSummary(Base):
    __tablename__ = "session_summaries"
    session_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), index=True)
    summary: Mapped[str] = mapped_column(Text)
    metrics: Mapped[dict] = mapped_column(JSON)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    embedding: Mapped[list] = mapped_column(Vector(settings.embedding_dim))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
