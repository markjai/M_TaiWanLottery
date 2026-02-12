"""Scraper control API endpoints."""

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.api.deps import get_db
from taiwan_lottery.schemas.lottery import ScrapeRequest, BackfillRequest
from taiwan_lottery.services.lottery_service import VALID_GAMES, trigger_scrape
from taiwan_lottery.scraper.scheduler import get_scheduler_status

router = APIRouter()


@router.post("/trigger")
async def trigger_scrape_endpoint(
    request: ScrapeRequest,
    db: AsyncSession = Depends(get_db),
):
    """手動觸發爬蟲抓取最新開獎資料."""
    if request.game_type not in VALID_GAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid game type. Valid: {VALID_GAMES}",
        )
    result = await trigger_scrape(db, request.game_type)
    return result


@router.post("/backfill")
async def backfill_endpoint(
    request: BackfillRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """回補歷史開獎資料（在背景執行）."""
    if request.game_type not in VALID_GAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid game type. Valid: {VALID_GAMES}",
        )

    from taiwan_lottery.scraper.backfill import backfill_game
    from taiwan_lottery.db.engine import async_session_factory

    async def _run_backfill():
        async with async_session_factory() as session:
            await backfill_game(
                session,
                request.game_type,
                year_start=request.year_start,
                year_end=request.year_end,
            )

    background_tasks.add_task(_run_backfill)

    return {
        "message": f"Backfill started for {request.game_type}",
        "year_start": request.year_start,
        "year_end": request.year_end,
    }


@router.get("/status")
async def scraper_status():
    """取得爬蟲排程狀態."""
    jobs = get_scheduler_status()
    return {"scheduler_jobs": jobs}
