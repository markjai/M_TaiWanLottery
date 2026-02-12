"""ML model records and prediction ORM models."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, ARRAY, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from taiwan_lottery.db.base import Base


class MLModelRecord(Base):
    """已訓練模型的記錄."""

    __tablename__ = "ml_model_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(30), nullable=False)  # frequency / lstm / dqn / ensemble
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<MLModelRecord game={self.game_type} type={self.model_type} v={self.version}>"


class MLPrediction(Base):
    """模型預測記錄."""

    __tablename__ = "ml_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(30), nullable=False)
    target_term: Mapped[str | None] = mapped_column(String(20), nullable=True)
    predicted_numbers: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    confidence_scores: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)
    actual_numbers: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    hit_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<MLPrediction game={self.game_type} model={self.model_type} hits={self.hit_count}>"
