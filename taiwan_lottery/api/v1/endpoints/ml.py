"""ML prediction API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.api.deps import get_db
from taiwan_lottery.schemas.ml import (
    TrainRequest,
    TrainResponse,
    PredictionResponse,
    ModelInfo,
    EvaluateRequest,
    EvaluateResponse,
    BacktestRequest,
    BacktestResponse,
)
from taiwan_lottery.services.lottery_service import VALID_GAMES

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    db: AsyncSession = Depends(get_db),
):
    """啟動模型訓練."""
    if request.game_type not in VALID_GAMES:
        raise HTTPException(status_code=400, detail=f"Invalid game. Valid: {VALID_GAMES}")

    from taiwan_lottery.services.ml_service import train_model as do_train
    result = await do_train(
        db, request.game_type, request.model_type,
        pick_count=request.pick_count,
    )
    return result


@router.get("/predict/{game}", response_model=PredictionResponse)
async def predict(
    game: str,
    model_type: str = "ensemble",
    pick_count: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """取得下一期預測號碼. pick_count 可指定預測幾個號碼（賓果賓果用）."""
    if game not in VALID_GAMES:
        raise HTTPException(status_code=400, detail=f"Invalid game. Valid: {VALID_GAMES}")

    # Build the storage model_type with pick_count suffix
    lookup_type = model_type
    if pick_count is not None:
        base = model_type.split("_p")[0]
        lookup_type = f"{base}_p{pick_count}"

    from taiwan_lottery.services.ml_service import get_prediction
    result = await get_prediction(db, game, lookup_type)
    if not result:
        raise HTTPException(status_code=404, detail="No model available. Train a model first.")
    return result


@router.get("/models", response_model=list[ModelInfo])
async def list_models(
    game_type: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """列出已訓練的模型."""
    from taiwan_lottery.db.crud import ml_record as crud
    models = await crud.list_models(db, game_type=game_type)
    return models


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    request: EvaluateRequest,
    db: AsyncSession = Depends(get_db),
):
    """評估模型預測準確度."""
    if request.game_type not in VALID_GAMES:
        raise HTTPException(status_code=400, detail=f"Invalid game. Valid: {VALID_GAMES}")

    from taiwan_lottery.services.ml_service import evaluate_model
    result = await evaluate_model(db, request.game_type, request.model_type)
    return result


@router.post("/backtest", response_model=BacktestResponse)
async def backtest(
    request: BacktestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Walk-forward 回測：在歷史資料上測試模型預測準確度."""
    if request.game_type not in VALID_GAMES:
        raise HTTPException(status_code=400, detail=f"Invalid game. Valid: {VALID_GAMES}")

    from taiwan_lottery.services.ml_service import run_backtest_service
    result = await run_backtest_service(
        db, request.game_type, request.model_type,
        test_size=request.test_size, compare_all=request.compare_all,
        pick_count=request.pick_count,
    )
    return result
