"""Parser for 今彩539 (Daily Cash 5/39) crawler output."""

from datetime import datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.db.crud import daily_cash as crud
from taiwan_lottery.scraper.base import BaseScraper
from taiwan_lottery.scraper.lottery_client import lottery_client


def _parse_row(row: dict) -> dict:
    """Parse a single crawler result row.

    v1.5.1 format:
        {'期別': 114000316, '開獎日期': '2025-12-31T00:00:00',
         '獎號': [8, 10, 11, 26, 35]}
    """
    draw_term = str(row.get("期別", ""))
    draw_date = datetime.fromisoformat(str(row.get("開獎日期", ""))).date()

    numbers = list(row.get("獎號", []))

    numbers_sorted = sorted(numbers[:5])
    sum_total = sum(numbers_sorted)
    odd_count = sum(1 for n in numbers_sorted if n % 2 == 1)
    span = numbers_sorted[-1] - numbers_sorted[0] if numbers_sorted else 0

    return {
        "draw_term": draw_term,
        "draw_date": draw_date,
        "num_1": numbers[0] if len(numbers) > 0 else 0,
        "num_2": numbers[1] if len(numbers) > 1 else 0,
        "num_3": numbers[2] if len(numbers) > 2 else 0,
        "num_4": numbers[3] if len(numbers) > 3 else 0,
        "num_5": numbers[4] if len(numbers) > 4 else 0,
        "numbers_sorted": numbers_sorted,
        "sum_total": sum_total,
        "odd_count": odd_count,
        "span": span,
    }


class DailyCashScraper(BaseScraper):
    game_type = "daily_cash"

    async def fetch_latest(self, session: AsyncSession) -> int:
        now = datetime.now()
        roc_year = now.year - 1911
        return await self.fetch_by_month(session, roc_year, now.month)

    async def fetch_by_month(
        self, session: AsyncSession, year: int, month: int
    ) -> int:
        raw_data = await lottery_client.get_daily_cash(year, month)
        if not raw_data:
            logger.info("No daily_cash data for {}/{}", year, month)
            return 0

        draws = []
        for row in raw_data:
            try:
                draws.append(_parse_row(row))
            except Exception as e:
                logger.warning("Failed to parse daily_cash row: {} - {}", row, e)

        return await crud.bulk_upsert(session, draws)
