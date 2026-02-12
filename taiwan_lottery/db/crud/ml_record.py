"""CRUD operations for ML model records and predictions."""

from sqlalchemy import select, desc, update
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.db.models.ml_record import MLModelRecord, MLPrediction


# --- MLModelRecord ---

async def create_model_record(session: AsyncSession, record: dict) -> MLModelRecord:
    obj = MLModelRecord(**record)
    session.add(obj)
    await session.flush()
    return obj


async def get_active_model(
    session: AsyncSession, game_type: str, model_type: str
) -> MLModelRecord | None:
    result = await session.execute(
        select(MLModelRecord)
        .where(
            MLModelRecord.game_type == game_type,
            MLModelRecord.model_type == model_type,
            MLModelRecord.is_active == True,  # noqa: E712
        )
        .order_by(desc(MLModelRecord.trained_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def list_models(
    session: AsyncSession, game_type: str | None = None
) -> list[MLModelRecord]:
    query = select(MLModelRecord).order_by(desc(MLModelRecord.trained_at))
    if game_type:
        query = query.where(MLModelRecord.game_type == game_type)
    result = await session.execute(query)
    return list(result.scalars().all())


async def set_active_model(session: AsyncSession, model_id: int) -> None:
    """Set a model as active and deactivate others of the same game/type."""
    model = await session.get(MLModelRecord, model_id)
    if not model:
        return

    # Deactivate others
    await session.execute(
        update(MLModelRecord)
        .where(
            MLModelRecord.game_type == model.game_type,
            MLModelRecord.model_type == model.model_type,
            MLModelRecord.id != model_id,
        )
        .values(is_active=False)
    )
    model.is_active = True


# --- MLPrediction ---

async def create_prediction(session: AsyncSession, prediction: dict) -> MLPrediction:
    obj = MLPrediction(**prediction)
    session.add(obj)
    await session.flush()
    return obj


async def get_latest_prediction(
    session: AsyncSession, game_type: str, model_type: str | None = None
) -> MLPrediction | None:
    query = (
        select(MLPrediction)
        .where(MLPrediction.game_type == game_type)
        .order_by(desc(MLPrediction.created_at))
        .limit(1)
    )
    if model_type:
        query = query.where(MLPrediction.model_type == model_type)
    result = await session.execute(query)
    return result.scalar_one_or_none()


async def list_predictions(
    session: AsyncSession,
    game_type: str,
    *,
    limit: int = 20,
) -> list[MLPrediction]:
    result = await session.execute(
        select(MLPrediction)
        .where(MLPrediction.game_type == game_type)
        .order_by(desc(MLPrediction.created_at))
        .limit(limit)
    )
    return list(result.scalars().all())
