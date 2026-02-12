"""APScheduler cron jobs for automatic lottery data scraping."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from taiwan_lottery.db.engine import async_session_factory

_scheduler: AsyncIOScheduler | None = None


async def _scrape_game(game_type: str):
    """Scrape latest draws for a game type."""
    from taiwan_lottery.scraper.parsers.lotto649_parser import Lotto649Scraper
    from taiwan_lottery.scraper.parsers.super_lotto_parser import SuperLottoScraper
    from taiwan_lottery.scraper.parsers.daily_cash_parser import DailyCashScraper
    from taiwan_lottery.scraper.parsers.bingo_parser import BingoScraper

    scrapers = {
        "lotto649": Lotto649Scraper(),
        "super_lotto": SuperLottoScraper(),
        "daily_cash": DailyCashScraper(),
        "bingo": BingoScraper(),
    }

    scraper = scrapers.get(game_type)
    if not scraper:
        return

    async with async_session_factory() as session:
        try:
            await scraper.run_with_logging(session, action="latest")
            await session.commit()
        except Exception as e:
            logger.error("Scheduled scrape failed for {}: {}", game_type, e)
            await session.rollback()


def start_scheduler():
    """Start the APScheduler with cron jobs for each lottery game."""
    global _scheduler
    if _scheduler is not None:
        return

    _scheduler = AsyncIOScheduler()

    # 大樂透: Tuesday & Friday at 21:00
    _scheduler.add_job(
        _scrape_game, "cron",
        args=["lotto649"],
        day_of_week="tue,fri",
        hour=21, minute=0,
        id="lotto649_scrape",
    )

    # 威力彩: Monday & Thursday at 21:00
    _scheduler.add_job(
        _scrape_game, "cron",
        args=["super_lotto"],
        day_of_week="mon,thu",
        hour=21, minute=0,
        id="super_lotto_scrape",
    )

    # 今彩539: Monday-Saturday at 21:00
    _scheduler.add_job(
        _scrape_game, "cron",
        args=["daily_cash"],
        day_of_week="mon-sat",
        hour=21, minute=0,
        id="daily_cash_scrape",
    )

    # 賓果賓果: Daily batch at 00:30
    _scheduler.add_job(
        _scrape_game, "cron",
        args=["bingo"],
        hour=0, minute=30,
        id="bingo_scrape",
    )

    _scheduler.start()
    logger.info("Scheduler started with {} jobs", len(_scheduler.get_jobs()))


def stop_scheduler():
    """Shutdown the scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")


def get_scheduler_status() -> list[dict]:
    """Get status of all scheduled jobs."""
    if not _scheduler:
        return []

    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": str(job.next_run_time) if job.next_run_time else None,
            "trigger": str(job.trigger),
        })
    return jobs
