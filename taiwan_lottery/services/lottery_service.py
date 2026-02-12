"""Lottery service — orchestrates scraping and data retrieval."""

from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.db.crud import lotto649, super_lotto, daily_cash, bingo
from taiwan_lottery.scraper.parsers.lotto649_parser import Lotto649Scraper
from taiwan_lottery.scraper.parsers.super_lotto_parser import SuperLottoScraper
from taiwan_lottery.scraper.parsers.daily_cash_parser import DailyCashScraper
from taiwan_lottery.scraper.parsers.bingo_parser import BingoScraper

GAME_CRUD = {
    "lotto649": lotto649,
    "super_lotto": super_lotto,
    "daily_cash": daily_cash,
    "bingo": bingo,
}

GAME_SCRAPERS = {
    "lotto649": Lotto649Scraper,
    "super_lotto": SuperLottoScraper,
    "daily_cash": DailyCashScraper,
    "bingo": BingoScraper,
}

VALID_GAMES = set(GAME_CRUD.keys())


def get_crud(game_type: str):
    """Get the CRUD module for a game type."""
    if game_type not in GAME_CRUD:
        raise ValueError(f"Unknown game type: {game_type}. Valid: {VALID_GAMES}")
    return GAME_CRUD[game_type]


def get_scraper(game_type: str):
    """Get a scraper instance for a game type."""
    if game_type not in GAME_SCRAPERS:
        raise ValueError(f"Unknown game type: {game_type}. Valid: {VALID_GAMES}")
    return GAME_SCRAPERS[game_type]()


async def get_latest_draw(session: AsyncSession, game_type: str):
    crud_module = get_crud(game_type)
    return await crud_module.get_latest(session)


async def get_draw_by_term(session: AsyncSession, game_type: str, term: str):
    crud_module = get_crud(game_type)
    return await crud_module.get_by_term(session, term)


async def get_draws_paginated(
    session: AsyncSession,
    game_type: str,
    *,
    page: int = 1,
    page_size: int = 20,
    date_from: date | None = None,
    date_to: date | None = None,
) -> tuple[list, int]:
    crud_module = get_crud(game_type)
    return await crud_module.get_draws(
        session,
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
    )


async def trigger_scrape(session: AsyncSession, game_type: str) -> dict:
    """Manually trigger a scrape for a game type."""
    scraper = get_scraper(game_type)
    log = await scraper.run_with_logging(session, action="latest")
    return {
        "game_type": game_type,
        "status": log.status,
        "records_inserted": log.records_inserted,
        "error_message": log.error_message,
    }
