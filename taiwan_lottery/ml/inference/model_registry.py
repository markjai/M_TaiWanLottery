"""Model registry — loads and caches trained models."""

from pathlib import Path

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.config import settings
from taiwan_lottery.db.crud import ml_record as crud
from taiwan_lottery.ml.models.base_model import BaseLotteryModel
from taiwan_lottery.ml.models.frequency_model import FrequencyModel
from taiwan_lottery.ml.models.lstm_model import LSTMModel
from taiwan_lottery.ml.models.dqn_model import DQNModel
from taiwan_lottery.ml.models.ensemble import EnsembleModel
from taiwan_lottery.ml.training.data_loader import GAME_CONFIG

MODEL_CLASSES = {
    "frequency": FrequencyModel,
    "lstm": LSTMModel,
    "dqn": DQNModel,
    "ensemble": EnsembleModel,
}

# In-memory cache: (game_type, model_type) -> model instance
_cache: dict[tuple[str, str], BaseLotteryModel] = {}


def _instantiate_model(model_type: str, game_type: str) -> BaseLotteryModel:
    """Create a model instance with correct game-specific parameters.

    Supports suffixed model_type like "ensemble_p5" for custom pick_count.
    """
    # Parse base model type and optional pick_count from suffix
    base_type = model_type
    custom_pick = None
    if "_p" in model_type:
        parts = model_type.rsplit("_p", 1)
        if parts[1].isdigit():
            base_type = parts[0]
            custom_pick = int(parts[1])

    cls = MODEL_CLASSES.get(base_type)
    if not cls:
        raise ValueError(f"Unknown model type: {base_type}")

    config = GAME_CONFIG.get(game_type)
    if not config:
        raise ValueError(f"Unknown game type: {game_type}")

    max_num = config["max_num"]
    pick_count = custom_pick if custom_pick is not None else config["pick_count"]

    if base_type in ("lstm", "dqn", "ensemble"):
        return cls(max_num=max_num, pick_count=pick_count, device=settings.TORCH_DEVICE)
    return cls(max_num=max_num, pick_count=pick_count)


async def get_model(
    session: AsyncSession, game_type: str, model_type: str
) -> BaseLotteryModel | None:
    """Get a trained model, loading from disk if needed."""
    cache_key = (game_type, model_type)

    if cache_key in _cache:
        return _cache[cache_key]

    # Look up active model in DB
    record = await crud.get_active_model(session, game_type, model_type)
    if not record or not record.artifact_path:
        return None

    artifact_path = Path(record.artifact_path)
    # Ensemble saves as .json but DB stores .ensemble path
    if not artifact_path.exists():
        json_path = artifact_path.with_suffix(".json")
        if json_path.exists():
            artifact_path = json_path
        else:
            logger.warning("Model artifact not found: {}", artifact_path)
            return None

    try:
        model = _instantiate_model(model_type, game_type)
        model.load(artifact_path)
        _cache[cache_key] = model
        logger.info("Loaded model: {} {} from {}", game_type, model_type, artifact_path)
        return model
    except Exception as e:
        logger.error("Failed to load model {}: {}", artifact_path, e)
        return None


def invalidate_cache(game_type: str | None = None, model_type: str | None = None):
    """Clear cached models."""
    if game_type and model_type:
        _cache.pop((game_type, model_type), None)
    elif game_type:
        keys = [k for k in _cache if k[0] == game_type]
        for k in keys:
            del _cache[k]
    else:
        _cache.clear()
