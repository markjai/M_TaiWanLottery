"""Parser for 賓果賓果 (Bingo Bingo) — via Taiwan Lottery JSON API.

Uses the official API at api.taiwanlottery.com/TLCAPIWeB/Lottery/BingoResult
instead of HTML scraping, since the website is a Nuxt.js SPA.
"""

from datetime import datetime, date, timedelta

import ssl

import aiohttp
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.db.crud import bingo as crud
from taiwan_lottery.scraper.base import BaseScraper

BINGO_API_URL = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/BingoResult"
PAGE_SIZE = 100  # max per request


def _compute_sectors(numbers: list[int]) -> tuple[int, int, int, int]:
    """Count numbers in each sector (1-20, 21-40, 41-60, 61-80)."""
    s1 = sum(1 for n in numbers if 1 <= n <= 20)
    s2 = sum(1 for n in numbers if 21 <= n <= 40)
    s3 = sum(1 for n in numbers if 41 <= n <= 60)
    s4 = sum(1 for n in numbers if 61 <= n <= 80)
    return s1, s2, s3, s4


def _parse_api_result(item: dict, draw_date: date) -> dict | None:
    """Parse a single API result item into DB format."""
    try:
        draw_term = str(item.get("drawTerm", ""))
        if not draw_term:
            return None

        # bigShowOrder contains sorted numbers as strings like ["01", "02", ...]
        big_show = item.get("bigShowOrder", [])
        numbers = [int(n) for n in big_show if n.strip().isdigit()]

        if len(numbers) < 20:
            return None

        numbers = numbers[:20]
        s1, s2, s3, s4 = _compute_sectors(numbers)

        # Derive approximate draw time from term number
        # Term format: YYYNNNNNN where YYY=ROC year, NNNNNN=sequential
        # Bingo draws every 5 minutes from 09:00 to 22:00
        # We store draw_date + midnight as datetime
        draw_datetime = datetime(draw_date.year, draw_date.month, draw_date.day)

        return {
            "draw_term": draw_term,
            "draw_datetime": draw_datetime,
            "numbers": sorted(numbers),
            "sum_total": sum(numbers),
            "odd_count": sum(1 for n in numbers if n % 2 == 1),
            "sector_1_count": s1,
            "sector_2_count": s2,
            "sector_3_count": s3,
            "sector_4_count": s4,
        }
    except Exception as e:
        logger.debug("Failed to parse bingo API item: {}", e)
        return None


class BingoScraper(BaseScraper):
    game_type = "bingo"

    def _make_session(self) -> tuple[aiohttp.TCPConnector, dict]:
        """Create SSL-bypassed connector and headers."""
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.taiwanlottery.com/",
            "Origin": "https://www.taiwanlottery.com",
        }
        return conn, headers

    async def _fetch_day(self, target_date: date) -> list[dict]:
        """Fetch all bingo draws for a specific date via API.

        Handles pagination — each page returns up to PAGE_SIZE results.
        """
        conn, headers = self._make_session()
        all_draws = []

        async with aiohttp.ClientSession(connector=conn) as client:
            page = 1
            while True:
                params = {
                    "openDate": target_date.strftime("%Y-%m-%d"),
                    "pageNum": page,
                    "pageSize": PAGE_SIZE,
                }
                try:
                    async with client.get(
                        BINGO_API_URL,
                        params=params,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status != 200:
                            logger.warning("Bingo API returned {}", resp.status)
                            break

                        data = await resp.json()
                        content = data.get("content", {})
                        results = content.get("bingoQueryResult", [])
                        total_size = content.get("totalSize") or 0

                        for item in results:
                            parsed = _parse_api_result(item, target_date)
                            if parsed:
                                all_draws.append(parsed)

                        # Check if we need more pages
                        if len(results) < PAGE_SIZE or len(all_draws) >= total_size:
                            break
                        page += 1

                except Exception as e:
                    logger.warning("Bingo API error for {} page {}: {}", target_date, page, e)
                    break

        return all_draws

    async def fetch_latest(self, session: AsyncSession) -> int:
        """Fetch today's bingo draws."""
        today = date.today()
        draws = await self._fetch_day(today)
        if not draws:
            # Try yesterday if today has no data yet
            yesterday = today - timedelta(days=1)
            draws = await self._fetch_day(yesterday)

        if not draws:
            logger.info("No bingo draws found for today/yesterday")
            return 0

        return await crud.bulk_upsert(session, draws)

    async def fetch_by_month(
        self, session: AsyncSession, year: int, month: int
    ) -> int:
        """Fetch bingo draws for a month by iterating dates.

        year: ROC year (民國年)
        """
        ad_year = year + 1911
        total_inserted = 0

        start = date(ad_year, month, 1)
        if month == 12:
            end = date(ad_year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(ad_year, month + 1, 1) - timedelta(days=1)

        # Don't go past today
        today = date.today()
        if end > today:
            end = today

        current = start
        while current <= end:
            try:
                draws = await self._fetch_day(current)
                if draws:
                    inserted = await crud.bulk_upsert(session, draws)
                    total_inserted += inserted
                    if inserted > 0:
                        logger.info(
                            "[bingo] {}: {} draws inserted",
                            current, inserted,
                        )
            except Exception as e:
                logger.warning("Bingo fetch failed for {}: {}", current, e)
            current += timedelta(days=1)

        return total_inserted
