"""Historical data backfill — import from ROC year 93 (2004) onwards."""

from datetime import datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.scraper.parsers.lotto649_parser import Lotto649Scraper
from taiwan_lottery.scraper.parsers.super_lotto_parser import SuperLottoScraper
from taiwan_lottery.scraper.parsers.daily_cash_parser import DailyCashScraper
from taiwan_lottery.scraper.parsers.bingo_parser import BingoScraper

SCRAPERS = {
    "lotto649": Lotto649Scraper(),
    "super_lotto": SuperLottoScraper(),
    "daily_cash": DailyCashScraper(),
    "bingo": BingoScraper(),
}

# Game start years (ROC year)
GAME_START_YEAR = {
    "lotto649": 93,      # 2004
    "super_lotto": 97,   # 2008
    "daily_cash": 96,    # 2007
    "bingo": 103,        # 2014
}


async def backfill_game(
    session: AsyncSession,
    game_type: str,
    year_start: int | None = None,
    year_end: int | None = None,
) -> dict:
    """Backfill historical data for a specific game.

    Args:
        session: Database session
        game_type: One of lotto649, super_lotto, daily_cash, bingo
        year_start: ROC year to start from (default: game start year)
        year_end: ROC year to end at (default: current ROC year)

    Returns:
        dict with total_inserted and details
    """
    scraper = SCRAPERS.get(game_type)
    if not scraper:
        raise ValueError(f"Unknown game type: {game_type}")

    if year_start is None:
        year_start = GAME_START_YEAR.get(game_type, 93)
    if year_end is None:
        year_end = datetime.now().year - 1911

    logger.info(
        "Backfilling {} from ROC {} to {}",
        game_type, year_start, year_end,
    )

    total_inserted = 0
    errors = []

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            # Skip future months
            now = datetime.now()
            if year + 1911 > now.year or (year + 1911 == now.year and month > now.month):
                continue

            try:
                inserted = await scraper.fetch_by_month(session, year, month)
                total_inserted += inserted
                if inserted > 0:
                    logger.info(
                        "[{}] {}/{}: {} records",
                        game_type, year, month, inserted,
                    )
                # Commit periodically
                await session.commit()
            except Exception as e:
                logger.warning(
                    "[{}] Error for {}/{}: {}",
                    game_type, year, month, e,
                )
                errors.append(f"{year}/{month}: {e}")
                await session.rollback()

    return {
        "game_type": game_type,
        "total_inserted": total_inserted,
        "year_range": f"{year_start}-{year_end}",
        "errors": errors,
    }
