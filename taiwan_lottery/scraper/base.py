"""Base scraper abstract class."""

from abc import ABC, abstractmethod
from datetime import datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.db.models.scrape_log import ScrapeLog


class BaseScraper(ABC):
    """Abstract base for all lottery scrapers."""

    game_type: str = ""

    @abstractmethod
    async def fetch_latest(self, session: AsyncSession) -> int:
        """Fetch the latest draw(s). Returns number of records inserted."""
        ...

    @abstractmethod
    async def fetch_by_month(
        self, session: AsyncSession, year: int, month: int
    ) -> int:
        """Fetch draws for a specific month. Returns number inserted."""
        ...

    async def run_with_logging(
        self, session: AsyncSession, action: str = "latest", **kwargs
    ) -> ScrapeLog:
        """Execute a scrape action with logging."""
        log = ScrapeLog(
            game_type=self.game_type,
            status="running",
            started_at=datetime.now(),
        )
        session.add(log)
        await session.flush()

        try:
            if action == "latest":
                inserted = await self.fetch_latest(session)
            elif action == "month":
                inserted = await self.fetch_by_month(
                    session, kwargs["year"], kwargs["month"]
                )
            else:
                raise ValueError(f"Unknown action: {action}")

            log.status = "success"
            log.records_inserted = inserted
            log.records_found = inserted
            log.finished_at = datetime.now()
            logger.info(
                "[{}] {} completed: {} records inserted",
                self.game_type, action, inserted,
            )
        except Exception as e:
            log.status = "error"
            log.error_message = str(e)[:1000]
            log.finished_at = datetime.now()
            logger.error("[{}] {} failed: {}", self.game_type, action, e)

        return log
