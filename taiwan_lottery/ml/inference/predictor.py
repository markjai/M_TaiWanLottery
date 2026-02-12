"""Predictor — generates predictions using trained models."""

import numpy as np
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.ml.inference.model_registry import get_model
from taiwan_lottery.ml.training.data_loader import load_history
from taiwan_lottery.db.crud import ml_record as crud


def compute_confidence_details(
    probs: np.ndarray,
    predicted_numbers: list[int],
    max_num: int,
) -> list[dict]:
    """Compute detailed confidence metrics for predicted numbers.

    Returns list of dicts with: number, raw_probability, lift, percentile, normalized.
    """
    random_prob = 1.0 / max_num
    details = []

    # Compute percentile ranks for all numbers
    sorted_probs = np.sort(probs)
    # Min-max for normalization
    p_min = float(probs.min())
    p_max = float(probs.max())
    p_range = p_max - p_min if p_max > p_min else 1.0

    for num in predicted_numbers:
        raw_p = float(probs[num - 1])
        lift = raw_p / random_prob if random_prob > 0 else 0.0
        # Percentile: what fraction of numbers have lower probability
        rank = float(np.searchsorted(sorted_probs, raw_p, side="left"))
        percentile = rank / max_num * 100.0
        # Normalized: min-max scaling to 0-100
        normalized = (raw_p - p_min) / p_range * 100.0

        details.append({
            "number": num,
            "raw_probability": round(raw_p, 6),
            "lift": round(lift, 3),
            "percentile": round(percentile, 1),
            "normalized": round(normalized, 1),
        })

    return details


async def predict(
    session: AsyncSession,
    game_type: str,
    model_type: str = "ensemble",
    n_sets: int = 1,
) -> dict | None:
    """Generate predictions using a trained model.

    Returns:
        dict with predicted_numbers, confidence_scores, confidence_details, etc.
    """
    model = await get_model(session, game_type, model_type)
    if model is None:
        return None

    history, max_num, _default_pick_count = await load_history(session, game_type)
    if not history:
        return None

    # Use the model's actual pick_count (e.g. 3 for frequency_p3)
    actual_pick_count = getattr(model, "pick_count", _default_pick_count)

    try:
        predictions = model.predict(history, n_sets=n_sets)
        probs = model.get_probabilities(history)

        # Get confidence scores for predicted numbers
        confidence_scores = []
        confidence_details_list = []
        for pred_set in predictions:
            scores = [float(probs[n - 1]) for n in pred_set]
            confidence_scores.append(scores)
            confidence_details_list.append(
                compute_confidence_details(probs, pred_set, max_num)
            )

        # Store prediction in DB
        for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
            await crud.create_prediction(session, {
                "game_type": game_type,
                "model_type": model_type,
                "predicted_numbers": pred,
                "confidence_scores": conf,
            })

        return {
            "game_type": game_type,
            "model_type": model_type,
            "predictions": predictions,
            "confidence_scores": confidence_scores,
            "confidence_details": confidence_details_list,
            "max_num": max_num,
            "pick_count": actual_pick_count,
        }

    except Exception as e:
        logger.error("Prediction failed for {} {}: {}", game_type, model_type, e)
        return None


async def evaluate_predictions(
    session: AsyncSession,
    game_type: str,
    model_type: str | None = None,
) -> dict:
    """Evaluate past predictions against actual results."""
    predictions = await crud.list_predictions(session, game_type, limit=100)

    if not predictions:
        return {
            "game_type": game_type,
            "model_type": model_type or "all",
            "total_predictions": 0,
            "average_hits": 0,
            "hit_distribution": {},
        }

    # Filter by model type if specified
    if model_type:
        predictions = [p for p in predictions if p.model_type == model_type]

    total_hits = 0
    hit_dist: dict[str, int] = {}
    evaluated = 0

    for pred in predictions:
        if pred.actual_numbers and pred.hit_count is not None:
            total_hits += pred.hit_count
            key = str(pred.hit_count)
            hit_dist[key] = hit_dist.get(key, 0) + 1
            evaluated += 1

    avg_hits = total_hits / evaluated if evaluated > 0 else 0

    return {
        "game_type": game_type,
        "model_type": model_type or "all",
        "total_predictions": len(predictions),
        "evaluated_predictions": evaluated,
        "average_hits": round(avg_hits, 4),
        "hit_distribution": hit_dist,
    }
