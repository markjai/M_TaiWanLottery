"""CRUD operations for 賓果賓果 (Bingo Bingo)."""

from datetime import date

from sqlalchemy import select, func, desc, cast, Date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from taiwan_lottery.db.models.bingo import BingoDraw


async def get_latest(session: AsyncSession) -> BingoDraw | None:
    result = await session.execute(
        select(BingoDraw).order_by(desc(BingoDraw.draw_datetime)).limit(1)
    )
    return result.scalar_one_or_none()


async def get_by_term(session: AsyncSession, term: str) -> BingoDraw | None:
    result = await session.execute(
        select(BingoDraw).where(BingoDraw.draw_term == term)
    )
    return result.scalar_one_or_none()


async def get_draws(
    session: AsyncSession,
    *,
    page: int = 1,
    page_size: int = 20,
    date_from: date | None = None,
    date_to: date | None = None,
) -> tuple[list[BingoDraw], int]:
    query = select(BingoDraw)
    count_query = select(func.count(BingoDraw.id))

    if date_from:
        query = query.where(cast(BingoDraw.draw_datetime, Date) >= date_from)
        count_query = count_query.where(cast(BingoDraw.draw_datetime, Date) >= date_from)
    if date_to:
        query = query.where(cast(BingoDraw.draw_datetime, Date) <= date_to)
        count_query = count_query.where(cast(BingoDraw.draw_datetime, Date) <= date_to)

    total = (await session.execute(count_query)).scalar() or 0

    query = query.order_by(desc(BingoDraw.draw_datetime))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    return list(result.scalars().all()), total


async def get_draws_by_date(session: AsyncSession, target_date: date) -> list[BingoDraw]:
    result = await session.execute(
        select(BingoDraw)
        .where(cast(BingoDraw.draw_datetime, Date) == target_date)
        .order_by(BingoDraw.draw_datetime)
    )
    return list(result.scalars().all())


async def get_all_sorted_by_date(session: AsyncSession) -> list[BingoDraw]:
    result = await session.execute(
        select(BingoDraw).order_by(BingoDraw.draw_datetime)
    )
    return list(result.scalars().all())


async def upsert(session: AsyncSession, draw: dict) -> bool:
    stmt = insert(BingoDraw).values(**draw)
    stmt = stmt.on_conflict_do_update(
        index_elements=["draw_term"],
        set_={"draw_datetime": draw["draw_datetime"]},
    )
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
