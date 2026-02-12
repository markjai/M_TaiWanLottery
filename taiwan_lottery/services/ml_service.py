"""ML service — high-level interface for training and prediction."""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from taiwan_lottery.ml.training.trainer import train_model as _train
from taiwan_lottery.ml.inference.predictor import predict as _predict, evaluate_predictions
from taiwan_lottery.ml.inference.model_registry import invalidate_cache
from taiwan_lottery.schemas.ml import (
    TrainResponse, PredictionResponse, ConfidenceDetail, EvaluateResponse,
    BacktestResponse, BacktestSummary,
)


async def train_model(
    session: AsyncSession, game_type: str, model_type: str,
    pick_count: int | None = None,
) -> TrainResponse:
    """Train a model and return response."""
    model, metrics, model_id = await _train(
        session, game_type, model_type, pick_count_override=pick_count,
    )

    # Invalidate cache so new model is used
    storage_type = model_type
    if pick_count is not None:
        base = model_type.split("_p")[0]
        storage_type = f"{base}_p{pick_count}"
    invalidate_cache(game_type, storage_type)

    return TrainResponse(
        model_id=model_id,
        game_type=game_type,
        model_type=model_type,
        version=datetime.now().strftime("%Y%m%d_%H%M%S"),
        metrics=metrics,
        message=f"Successfully trained {model_type} model for {game_type}",
    )


async def get_prediction(
    session: AsyncSession, game_type: str, model_type: str
) -> PredictionResponse | None:
    """Get prediction from a trained model."""
    result = await _predict(session, game_type, model_type)
    if not result:
        return None

    max_num = result.get("max_num")
    pick_count = result.get("pick_count")

    # Compute expected random hit count
    # Bingo: 20 numbers drawn out of 80 each period
    expected_random_hit = None
    if max_num and pick_count:
        if game_type == "bingo":
            draw_count = 20  # bingo draws 20 numbers per period
            expected_random_hit = round(pick_count * draw_count / max_num, 4)
        else:
            expected_random_hit = round(pick_count * pick_count / max_num, 4)

    # Build confidence details
    confidence_details = None
    if result.get("confidence_details"):
        confidence_details = [
            ConfidenceDetail(**d) for d in result["confidence_details"][0]
        ]

    return PredictionResponse(
        game_type=result["game_type"],
        model_type=result["model_type"],
        predicted_numbers=result["predictions"][0],
        confidence_scores=result["confidence_scores"][0] if result["confidence_scores"] else None,
        confidence_details=confidence_details,
        max_num=max_num,
        pick_count=pick_count,
        expected_random_hit=expected_random_hit,
        created_at=datetime.now(),
    )


async def evaluate_model(
    session: AsyncSession, game_type: str, model_type: str | None = None
) -> EvaluateResponse:
    """Evaluate model predictions."""
    result = await evaluate_predictions(session, game_type, model_type)
    return EvaluateResponse(
        game_type=result["game_type"],
        model_type=result["model_type"],
        total_predictions=result["total_predictions"],
        average_hits=result["average_hits"],
        hit_distribution=result.get("hit_distribution", {}),
    )


async def run_backtest_service(
    session: AsyncSession,
    game_type: str,
    model_type: str = "ensemble",
    test_size: int = 100,
    compare_all: bool = False,
    pick_count: int | None = None,
) -> BacktestResponse:
    """Run walk-forward backtest on historical data."""
    from taiwan_lottery.ml.training.data_loader import load_history
    from taiwan_lottery.ml.backtesting import run_backtest, run_comparison_backtest

    history, max_num, default_pick_count = await load_history(session, game_type)
    actual_pick_count = pick_count if pick_count is not None else default_pick_count

    if compare_all:
        result = run_comparison_backtest(
            history, max_num, actual_pick_count, test_size=test_size,
        )
        return BacktestResponse(
            game_type=game_type,
            comparison=result["comparison"],
            message=f"Comparison backtest on {test_size} draws (pick={actual_pick_count}, {len(history)} total)",
        )

    # Pre-compute bias boost for frequency model
    if model_type == "frequency":
        from taiwan_lottery.ml.bias_detector import BiasDetector
        detector = BiasDetector(max_num, actual_pick_count)
        bias_boost = detector.get_bias_boost(history)
    else:
        bias_boost = None

    result = run_backtest(
        history, max_num, actual_pick_count,
        model_type=model_type, test_size=test_size,
        bias_boost=bias_boost,
    )

    # Strip details for API response (too large)
    summary_data = {k: v for k, v in result.items() if k != "details"}
    summary = BacktestSummary(**summary_data)

    return BacktestResponse(
        game_type=game_type,
        backtest=summary,
        message=(
            f"{model_type} backtest: avg_hits={summary.average_hits}, "
            f"lift={summary.lift_vs_random}x vs random"
        ),
    )
