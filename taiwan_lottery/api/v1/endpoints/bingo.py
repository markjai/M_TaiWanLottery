"""賓果賓果 (Bingo Bingo) API endpoints."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.api.deps import get_db
from taiwan_lottery.db.crud import bingo as crud
from taiwan_lottery.schemas.lottery import BingoDrawSchema, PaginatedResponse

router = APIRouter()


@router.get("/latest", response_model=BingoDrawSchema)
async def get_latest(db: AsyncSession = Depends(get_db)):
    """取得最新一期賓果賓果開獎結果."""
    draw = await crud.get_latest(db)
    if not draw:
        raise HTTPException(status_code=404, detail="No draws found")
    return draw


@router.get("/draws", response_model=PaginatedResponse)
async def get_draws(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    date_from: date | None = None,
    date_to: date | None = None,
    db: AsyncSession = Depends(get_db),
):
    """查詢賓果賓果歷史開獎記錄（分頁）."""
    draws, total = await crud.get_draws(
        db, page=page, page_size=page_size, date_from=date_from, date_to=date_to,
    )
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    return PaginatedResponse(
        items=[BingoDrawSchema.model_validate(d) for d in draws],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/draws/{term}", response_model=BingoDrawSchema)
async def get_by_term(term: str, db: AsyncSession = Depends(get_db)):
    """查詢特定期別的賓果賓果開獎結果."""
    draw = await crud.get_by_term(db, term)
    if not draw:
        raise HTTPException(status_code=404, detail=f"Draw term {term} not found")
    return draw
