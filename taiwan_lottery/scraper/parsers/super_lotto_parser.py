"""Parser for 威力彩 (Super Lotto) crawler output."""

from datetime import datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.db.crud import super_lotto as crud
from taiwan_lottery.scraper.base import BaseScraper
from taiwan_lottery.scraper.lottery_client import lottery_client


def _parse_row(row: dict) -> dict:
    """Parse a single crawler result row.

    v1.5.1 format:
        {'期別': 114000104, '開獎日期': '2025-12-29T00:00:00',
         '第一區': [1, 3, 9, 11, 18, 32], '第二區': 6}
    """
    draw_term = str(row.get("期別", ""))
    draw_date = datetime.fromisoformat(str(row.get("開獎日期", ""))).date()

    zone1 = list(row.get("第一區", []))
    zone2 = int(row.get("第二區", 0))

    zone1_sorted = sorted(zone1[:6])
    sum_zone1 = sum(zone1_sorted)
    odd_count = sum(1 for n in zone1_sorted if n % 2 == 1)
    span = zone1_sorted[-1] - zone1_sorted[0] if zone1_sorted else 0

    return {
        "draw_term": draw_term,
        "draw_date": draw_date,
        "zone1_num_1": zone1[0] if len(zone1) > 0 else 0,
        "zone1_num_2": zone1[1] if len(zone1) > 1 else 0,
        "zone1_num_3": zone1[2] if len(zone1) > 2 else 0,
        "zone1_num_4": zone1[3] if len(zone1) > 3 else 0,
        "zone1_num_5": zone1[4] if len(zone1) > 4 else 0,
        "zone1_num_6": zone1[5] if len(zone1) > 5 else 0,
        "zone2_num": zone2,
        "zone1_sorted": zone1_sorted,
        "sum_zone1": sum_zone1,
        "odd_count": odd_count,
        "span": span,
    }


class SuperLottoScraper(BaseScraper):
    game_type = "super_lotto"

    async def fetch_latest(self, session: AsyncSession) -> int:
        now = datetime.now()
        roc_year = now.year - 1911
        return await self.fetch_by_month(session, roc_year, now.month)

    async def fetch_by_month(
        self, session: AsyncSession, year: int, month: int
    ) -> int:
        raw_data = await lottery_client.get_super_lotto(year, month)
        if not raw_data:
            logger.info("No super_lotto data for {}/{}", year, month)
            return 0

        draws = []
        for row in raw_data:
            try:
                draws.append(_parse_row(row))
            except Exception as e:
                logger.warning("Failed to parse super_lotto row: {} - {}", row, e)

        return await crud.bulk_upsert(session, draws)
