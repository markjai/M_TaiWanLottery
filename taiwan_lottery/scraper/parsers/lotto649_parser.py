"""Parser for 大樂透 (Lotto 6/49) crawler output."""

from datetime import datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.db.crud import lotto649 as crud
from taiwan_lottery.scraper.base import BaseScraper
from taiwan_lottery.scraper.lottery_client import lottery_client


def _parse_row(row: dict) -> dict:
    """Parse a single crawler result row into a dict for DB insertion.

    v1.5.1 format:
        {'期別': 114000118, '開獎日期': '2025-12-30T00:00:00',
         '獎號': [5, 10, 15, 20, 24, 29], '特別號': 34}
    """
    draw_term = str(row.get("期別", ""))

    date_str = str(row.get("開獎日期", ""))
    draw_date = datetime.fromisoformat(date_str).date()

    numbers = list(row.get("獎號", []))
    special = int(row.get("特別號", 0))

    numbers_sorted = sorted(numbers[:6])
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
        "num_6": numbers[5] if len(numbers) > 5 else 0,
        "special_num": special,
        "numbers_sorted": numbers_sorted,
        "sum_total": sum_total,
        "odd_count": odd_count,
        "span": span,
    }


class Lotto649Scraper(BaseScraper):
    game_type = "lotto649"

    async def fetch_latest(self, session: AsyncSession) -> int:
        now = datetime.now()
        roc_year = now.year - 1911
        return await self.fetch_by_month(session, roc_year, now.month)

    async def fetch_by_month(
        self, session: AsyncSession, year: int, month: int
    ) -> int:
        raw_data = await lottery_client.get_lotto649(year, month)
        if not raw_data:
            logger.info("No lotto649 data for {}/{}", year, month)
            return 0

        draws = []
        for row in raw_data:
            try:
                draws.append(_parse_row(row))
            except Exception as e:
                logger.warning("Failed to parse lotto649 row: {} - {}", row, e)

        return await crud.bulk_upsert(session, draws)
