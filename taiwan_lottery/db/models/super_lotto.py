"""威力彩 (Super Lotto) ORM model."""

from datetime import date

from sqlalchemy import Date, Integer, String, ARRAY, Index
from sqlalchemy.orm import Mapped, mapped_column

from taiwan_lottery.db.base import Base


class SuperLottoDraw(Base):
    """威力彩開獎記錄 — 第一區從38個號碼開6個 + 第二區從8個號碼開1個."""

    __tablename__ = "super_lotto_draws"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    draw_term: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    draw_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Zone 1: 6 numbers from 1-38
    zone1_num_1: Mapped[int] = mapped_column(Integer, nullable=False)
    zone1_num_2: Mapped[int] = mapped_column(Integer, nullable=False)
    zone1_num_3: Mapped[int] = mapped_column(Integer, nullable=False)
    zone1_num_4: Mapped[int] = mapped_column(Integer, nullable=False)
    zone1_num_5: Mapped[int] = mapped_column(Integer, nullable=False)
    zone1_num_6: Mapped[int] = mapped_column(Integer, nullable=False)

    # Zone 2: 1 number from 1-8
    zone2_num: Mapped[int] = mapped_column(Integer, nullable=False)

    # Sorted array for zone 1
    zone1_sorted: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)

    # Pre-computed features (zone 1 only)
    sum_zone1: Mapped[int] = mapped_column(Integer, nullable=False)
    odd_count: Mapped[int] = mapped_column(Integer, nullable=False)
    span: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_super_lotto_zone1_sorted", "zone1_sorted", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<SuperLottoDraw term={self.draw_term} zone1={self.zone1_sorted} zone2={self.zone2_num}>"
