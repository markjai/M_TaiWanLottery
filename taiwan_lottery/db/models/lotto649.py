"""大樂透 (Lotto 6/49) ORM model."""

from datetime import date

from sqlalchemy import Date, Integer, String, ARRAY, Index
from sqlalchemy.orm import Mapped, mapped_column

from taiwan_lottery.db.base import Base


class Lotto649Draw(Base):
    """大樂透開獎記錄 — 從49個號碼中開出6個 + 1個特別號."""

    __tablename__ = "lotto649_draws"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    draw_term: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    draw_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Original draw order
    num_1: Mapped[int] = mapped_column(Integer, nullable=False)
    num_2: Mapped[int] = mapped_column(Integer, nullable=False)
    num_3: Mapped[int] = mapped_column(Integer, nullable=False)
    num_4: Mapped[int] = mapped_column(Integer, nullable=False)
    num_5: Mapped[int] = mapped_column(Integer, nullable=False)
    num_6: Mapped[int] = mapped_column(Integer, nullable=False)
    special_num: Mapped[int] = mapped_column(Integer, nullable=False)

    # Sorted array for query convenience
    numbers_sorted: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)

    # Pre-computed ML features
    sum_total: Mapped[int] = mapped_column(Integer, nullable=False)
    odd_count: Mapped[int] = mapped_column(Integer, nullable=False)
    span: Mapped[int] = mapped_column(Integer, nullable=False)  # max - min

    __table_args__ = (
        Index("ix_lotto649_numbers_sorted", "numbers_sorted", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<Lotto649Draw term={self.draw_term} numbers={self.numbers_sorted}>"
