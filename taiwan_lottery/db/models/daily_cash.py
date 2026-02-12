"""今彩539 (Daily Cash 5/39) ORM model."""

from datetime import date

from sqlalchemy import Date, Integer, String, ARRAY, Index
from sqlalchemy.orm import Mapped, mapped_column

from taiwan_lottery.db.base import Base


class DailyCashDraw(Base):
    """今彩539開獎記錄 — 從39個號碼中開出5個."""

    __tablename__ = "daily_cash_draws"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    draw_term: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    draw_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Original draw order
    num_1: Mapped[int] = mapped_column(Integer, nullable=False)
    num_2: Mapped[int] = mapped_column(Integer, nullable=False)
    num_3: Mapped[int] = mapped_column(Integer, nullable=False)
    num_4: Mapped[int] = mapped_column(Integer, nullable=False)
    num_5: Mapped[int] = mapped_column(Integer, nullable=False)

    # Sorted array
    numbers_sorted: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)

    # Pre-computed ML features
    sum_total: Mapped[int] = mapped_column(Integer, nullable=False)
    odd_count: Mapped[int] = mapped_column(Integer, nullable=False)
    span: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_daily_cash_numbers_sorted", "numbers_sorted", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<DailyCashDraw term={self.draw_term} numbers={self.numbers_sorted}>"
