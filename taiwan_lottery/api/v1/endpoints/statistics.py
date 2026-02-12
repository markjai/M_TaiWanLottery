"""Statistics API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.api.deps import get_db
from taiwan_lottery.services.lottery_service import VALID_GAMES
from taiwan_lottery.services import statistics_service as stats
from taiwan_lottery.schemas.statistics import (
    NumberFrequency,
    HotColdAnalysis,
    GapAnalysis,
    PairFrequency,
    BiasReport,
)

router = APIRouter()


def _validate_game(game: str):
    if game not in VALID_GAMES:
        raise HTTPException(status_code=400, detail=f"Invalid game. Valid: {VALID_GAMES}")


@router.get("/{game}/frequency", response_model=list[NumberFrequency])
async def frequency(
    game: str,
    window: int | None = Query(None, description="只看最近N期"),
    db: AsyncSession = Depends(get_db),
):
    """號碼出現頻率分析."""
    _validate_game(game)
    return await stats.get_frequency(db, game, window=window)


@router.get("/{game}/hot-cold", response_model=HotColdAnalysis)
async def hot_cold(
    game: str,
    window: int = Query(30, description="觀察窗口期數"),
    db: AsyncSession = Depends(get_db),
):
    """冷熱號碼分析."""
    _validate_game(game)
    return await stats.get_hot_cold(db, game, window=window)


@router.get("/{game}/gaps", response_model=list[GapAnalysis])
async def gaps(
    game: str,
    db: AsyncSession = Depends(get_db),
):
    """號碼遺漏值分析."""
    _validate_game(game)
    return await stats.get_gaps(db, game)


@router.get("/{game}/pairs", response_model=list[PairFrequency])
async def pairs(
    game: str,
    top_n: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """常見號碼配對分析."""
    _validate_game(game)
    return await stats.get_pairs(db, game, top_n=top_n)


@router.get("/{game}/bias", response_model=BiasReport)
async def bias_detection(
    game: str,
    db: AsyncSession = Depends(get_db),
):
    """偏差檢測分析 — 卡方、序列隨機性、位置偏差."""
    _validate_game(game)
    return await stats.get_bias_report(db, game)
