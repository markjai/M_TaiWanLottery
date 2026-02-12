"""Model trainer — orchestrates training pipeline."""

from datetime import datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.config import settings
from taiwan_lottery.ml.models.base_model import BaseLotteryModel
from taiwan_lottery.ml.models.frequency_model import FrequencyModel
from taiwan_lottery.ml.models.lstm_model import LSTMModel
from taiwan_lottery.ml.models.dqn_model import DQNModel
from taiwan_lottery.ml.models.ensemble import EnsembleModel
from taiwan_lottery.ml.training.data_loader import load_history
from taiwan_lottery.db.crud import ml_record as crud


MODEL_CLASSES = {
    "frequency": FrequencyModel,
    "lstm": LSTMModel,
    "dqn": DQNModel,
    "ensemble": EnsembleModel,
}

# File extensions per model type
MODEL_EXT = {
    "frequency": ".json",
    "lstm": ".pt",
    "dqn": ".pt",
    "ensemble": ".ensemble",
}


def _create_model(model_type: str, max_num: int, pick_count: int) -> BaseLotteryModel:
    cls = MODEL_CLASSES.get(model_type)
    if not cls:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type in ("lstm", "dqn", "ensemble"):
        return cls(max_num, pick_count, device=settings.TORCH_DEVICE)
    return cls(max_num, pick_count)


async def train_model(
    session: AsyncSession,
    game_type: str,
    model_type: str,
    pick_count_override: int | None = None,
    **train_kwargs,
) -> tuple[BaseLotteryModel, dict, int]:
    """Train a model and save it.

    Args:
        pick_count_override: Override default pick_count (e.g. bingo 3~10).
            When set, model_type is stored with suffix like "ensemble_p5".

    Returns:
        (model_instance, metrics, model_record_id)
    """
    # Load data
    history, max_num, pick_count = await load_history(session, game_type)
    if len(history) < 10:
        raise ValueError(f"Not enough data: only {len(history)} draws available")

    # Apply pick_count override
    if pick_count_override is not None:
        pick_count = pick_count_override

    # Determine storage model_type (with suffix for custom pick_count)
    base_model_type = model_type.split("_p")[0]  # strip any existing suffix
    storage_model_type = model_type
    if pick_count_override is not None:
        storage_model_type = f"{base_model_type}_p{pick_count}"

    logger.info(
        "Training {} model for {} (pick_count={})",
        storage_model_type, game_type, pick_count,
    )

    # Create and train model
    ext_key = base_model_type  # use base type for extension lookup
    model = _create_model(base_model_type, max_num, pick_count)
    metrics = model.train(history, **train_kwargs)
    metrics["pick_count"] = pick_count

    # Save artifact
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = MODEL_EXT.get(ext_key, ".bin")
    artifact_name = f"{game_type}_{storage_model_type}_{version}{ext}"
    artifact_path = settings.MODEL_ARTIFACTS_DIR / artifact_name
    model.save(artifact_path)

    # Record in DB
    record = await crud.create_model_record(session, {
        "game_type": game_type,
        "model_type": storage_model_type,
        "version": version,
        "artifact_path": str(artifact_path),
        "metrics": metrics,
        "is_active": True,
    })

    # Set as active (deactivate others of same type)
    await crud.set_active_model(session, record.id)

    logger.info(
        "Model trained: {} {} v{} (id={})",
        game_type, storage_model_type, version, record.id,
    )

    return model, metrics, record.id
