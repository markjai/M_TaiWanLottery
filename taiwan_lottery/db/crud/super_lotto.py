"""CRUD operations for 威力彩 (Super Lotto)."""

from datetime import date

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from taiwan_lottery.db.models.super_lotto import SuperLottoDraw


async def get_latest(session: AsyncSession) -> SuperLottoDraw | None:
    result = await session.execute(
        select(SuperLottoDraw).order_by(desc(SuperLottoDraw.draw_date)).limit(1)
    )
    return result.scalar_one_or_none()


async def get_by_term(session: AsyncSession, term: str) -> SuperLottoDraw | None:
    result = await session.execute(
        select(SuperLottoDraw).where(SuperLottoDraw.draw_term == term)
    )
    return result.scalar_one_or_none()


async def get_draws(
    session: AsyncSession,
    *,
    page: int = 1,
    page_size: int = 20,
    date_from: date | None = None,
    date_to: date | None = None,
) -> tuple[list[SuperLottoDraw], int]:
    query = select(SuperLottoDraw)
    count_query = select(func.count(SuperLottoDraw.id))

    if date_from:
        query = query.where(SuperLottoDraw.draw_date >= date_from)
        count_query = count_query.where(SuperLottoDraw.draw_date >= date_from)
    if date_to:
        query = query.where(SuperLottoDraw.draw_date <= date_to)
        count_query = count_query.where(SuperLottoDraw.draw_date <= date_to)

    total = (await session.execute(count_query)).scalar() or 0

    query = query.order_by(desc(SuperLottoDraw.draw_date))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    return list(result.scalars().all()), total


async def get_all_sorted_by_date(session: AsyncSession) -> list[SuperLottoDraw]:
    result = await session.execute(
        select(SuperLottoDraw).order_by(SuperLottoDraw.draw_date)
    )
    return list(result.scalars().all())


async def upsert(session: AsyncSession, draw: dict) -> bool:
    stmt = insert(SuperLottoDraw).values(**draw)
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
