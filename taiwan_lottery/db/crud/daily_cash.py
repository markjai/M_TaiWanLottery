"""CRUD operations for 今彩539 (Daily Cash 5/39)."""

from datetime import date

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from taiwan_lottery.db.models.daily_cash import DailyCashDraw


async def get_latest(session: AsyncSession) -> DailyCashDraw | None:
    result = await session.execute(
        select(DailyCashDraw).order_by(desc(DailyCashDraw.draw_date)).limit(1)
    )
    return result.scalar_one_or_none()


async def get_by_term(session: AsyncSession, term: str) -> DailyCashDraw | None:
    result = await session.execute(
        select(DailyCashDraw).where(DailyCashDraw.draw_term == term)
    )
    return result.scalar_one_or_none()


async def get_draws(
    session: AsyncSession,
    *,
    page: int = 1,
    page_size: int = 20,
    date_from: date | None = None,
    date_to: date | None = None,
) -> tuple[list[DailyCashDraw], int]:
    query = select(DailyCashDraw)
    count_query = select(func.count(DailyCashDraw.id))

    if date_from:
        query = query.where(DailyCashDraw.draw_date >= date_from)
        count_query = count_query.where(DailyCashDraw.draw_date >= date_from)
    if date_to:
        query = query.where(DailyCashDraw.draw_date <= date_to)
        count_query = count_query.where(DailyCashDraw.draw_date <= date_to)

    total = (await session.execute(count_query)).scalar() or 0

    query = query.order_by(desc(DailyCashDraw.draw_date))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    return list(result.scalars().all()), total


async def get_all_sorted_by_date(session: AsyncSession) -> list[DailyCashDraw]:
    result = await session.execute(
        select(DailyCashDraw).order_by(DailyCashDraw.draw_date)
    )
    return list(result.scalars().all())


async def upsert(session: AsyncSession, draw: dict) -> bool:
    stmt = insert(DailyCashDraw).values(**draw)
    stmt = stmt.on_conflict_do_nothing(index_elements=["draw_term"])
    result = await session.execute(stmt)
    return result.rowcount > 0


async def bulk_upsert(session: AsyncSession, draws: list[dict]) -> int:
    if not draws:
        return 0
    inserted = 0
    for draw in draws:
        if await upsert(session, draw):
            inserted += 1
    return inserted
