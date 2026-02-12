"""CRUD operations for 大樂透 (Lotto 6/49)."""

from datetime import date

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from taiwan_lottery.db.models.lotto649 import Lotto649Draw


async def get_latest(session: AsyncSession) -> Lotto649Draw | None:
    result = await session.execute(
        select(Lotto649Draw).order_by(desc(Lotto649Draw.draw_date)).limit(1)
    )
    return result.scalar_one_or_none()


async def get_by_term(session: AsyncSession, term: str) -> Lotto649Draw | None:
    result = await session.execute(
        select(Lotto649Draw).where(Lotto649Draw.draw_term == term)
    )
    return result.scalar_one_or_none()


async def get_draws(
    session: AsyncSession,
    *,
    page: int = 1,
    page_size: int = 20,
    date_from: date | None = None,
    date_to: date | None = None,
) -> tuple[list[Lotto649Draw], int]:
    query = select(Lotto649Draw)
    count_query = select(func.count(Lotto649Draw.id))

    if date_from:
        query = query.where(Lotto649Draw.draw_date >= date_from)
        count_query = count_query.where(Lotto649Draw.draw_date >= date_from)
    if date_to:
        query = query.where(Lotto649Draw.draw_date <= date_to)
        count_query = count_query.where(Lotto649Draw.draw_date <= date_to)

    total = (await session.execute(count_query)).scalar() or 0

    query = query.order_by(desc(Lotto649Draw.draw_date))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    return list(result.scalars().all()), total


async def get_all_sorted_by_date(session: AsyncSession) -> list[Lotto649Draw]:
    result = await session.execute(
        select(Lotto649Draw).order_by(Lotto649Draw.draw_date)
    )
    return list(result.scalars().all())


async def upsert(session: AsyncSession, draw: dict) -> bool:
    """Insert or ignore a draw record. Returns True if inserted."""
    stmt = insert(Lotto649Draw).values(**draw)
    stmt = stmt.on_conflict_do_nothing(index_elements=["draw_term"])
    result = await session.execute(stmt)
    return result.rowcount > 0


async def bulk_upsert(session: AsyncSession, draws: list[dict]) -> int:
    """Bulk insert draws, skip conflicts. Returns number inserted."""
    if not draws:
        return 0
    inserted = 0
    for draw in draws:
        if await upsert(session, draw):
            inserted += 1
    return inserted
