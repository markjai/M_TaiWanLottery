"""Walk-forward backtesting engine for lottery prediction models.

Trains a model on a training window, then evaluates prediction accuracy
on each subsequent draw using all data available up to that point.
"""

from collections import Counter

import numpy as np
from loguru import logger

from taiwan_lottery.config import settings
from taiwan_lottery.ml.models.base_model import BaseLotteryModel
from taiwan_lottery.ml.models.frequency_model import FrequencyModel
from taiwan_lottery.ml.models.lstm_model import LSTMModel
from taiwan_lottery.ml.models.dqn_model import DQNModel
from taiwan_lottery.ml.models.ensemble import EnsembleModel

MODEL_CLASSES = {
    "frequency": FrequencyModel,
    "lstm": LSTMModel,
    "dqn": DQNModel,
    "ensemble": EnsembleModel,
}


def _create_model(model_type: str, max_num: int, pick_count: int) -> BaseLotteryModel:
    cls = MODEL_CLASSES.get(model_type)
    if not cls:
        raise ValueError(f"Unknown model type: {model_type}")
    if model_type in ("lstm", "dqn", "ensemble"):
        return cls(max_num, pick_count, device=settings.TORCH_DEVICE)
    return cls(max_num, pick_count)


def run_backtest(
    history: list[list[int]],
    max_num: int,
    pick_count: int,
    model_type: str = "ensemble",
    test_size: int = 100,
    retrain_every: int | None = None,
) -> dict:
    """Run walk-forward backtest.

    Strategy:
        1. Split history into train / test
        2. Train model on train data
        3. For each test draw, use all data up to that point for inference
        4. Compare top-k predicted numbers with actual draw
        5. Optionally retrain periodically

    Args:
        history: Full chronological draw history (list of sorted number lists)
        max_num: Maximum number in the game (e.g. 49 for lotto649)
        pick_count: How many numbers per draw (e.g. 6 for lotto649)
        model_type: Model to backtest (frequency/lstm/dqn/ensemble)
        test_size: Number of recent draws to test on
        retrain_every: Retrain model every N test draws (None = no retrain)

    Returns:
        dict with detailed backtest results
    """
    if len(history) < test_size + 50:
        raise ValueError(
            f"Not enough data: {len(history)} draws, need at least {test_size + 50}"
        )

    # For bingo (high-frequency), limit training data
    max_train = None
    if max_num >= 80:
        max_train = 10000

    split = len(history) - test_size
    train_history = history[:split]
    test_draws = history[split:]

    # Limit training data if needed
    train_data = train_history
    if max_train and len(train_data) > max_train:
        train_data = train_data[-max_train:]

    logger.info(
        "[backtest] {} | train={}, test={}, max_num={}, pick={}",
        model_type, len(train_data), test_size, max_num, pick_count,
    )

    # Initial training
    model = _create_model(model_type, max_num, pick_count)
    model.train(train_data)

    # Walk-forward evaluation
    results = []
    hit_counts = []
    per_number_hits = Counter()  # track which numbers we predict correctly
    per_number_predicted = Counter()  # track which numbers we predict

    for i, actual in enumerate(test_draws):
        # Context = all data up to this point
        context = train_history + test_draws[:i]
        if max_train and len(context) > max_train:
            context = context[-max_train:]

        # Retrain periodically if requested
        if retrain_every and i > 0 and i % retrain_every == 0:
            logger.info("[backtest] Retraining at step {}/{}", i, test_size)
            model = _create_model(model_type, max_num, pick_count)
            retrain_data = train_history + test_draws[:i]
            if max_train and len(retrain_data) > max_train:
                retrain_data = retrain_data[-max_train:]
            model.train(retrain_data)

        # Get probabilities and pick top-k
        try:
            probs = model.get_probabilities(context)
            top_k_indices = np.argsort(probs)[-pick_count:]
            predicted = sorted(idx + 1 for idx in top_k_indices)
        except Exception as e:
            logger.warning("[backtest] Prediction failed at step {}: {}", i, e)
            predicted = []

        actual_set = set(actual)
        predicted_set = set(predicted)
        hits = len(actual_set & predicted_set)
        hit_counts.append(hits)

        # Track per-number accuracy
        for n in predicted:
            per_number_predicted[n] += 1
            if n in actual_set:
                per_number_hits[n] += 1

        results.append({
            "step": i,
            "predicted": predicted,
            "actual": actual,
            "hits": hits,
        })

    # Aggregate metrics
    hit_array = np.array(hit_counts)
    hit_dist = Counter(hit_counts)

    # Expected hits by random chance
    expected_random = pick_count * pick_count / max_num

    # Compute rolling averages (windows of 20)
    window = min(20, len(hit_counts))
    rolling_avg = []
    for i in range(len(hit_counts) - window + 1):
        rolling_avg.append(round(float(np.mean(hit_counts[i:i + window])), 4))

    # Best/worst streaks
    best_streak = _longest_streak(hit_counts, lambda h: h > 0)
    worst_streak = _longest_streak(hit_counts, lambda h: h == 0)

    # Top predicted numbers accuracy
    top_numbers = []
    for num in sorted(per_number_predicted, key=per_number_predicted.get, reverse=True)[:10]:
        total_pred = int(per_number_predicted[num])
        total_hit = int(per_number_hits.get(num, 0))
        top_numbers.append({
            "number": int(num),
            "times_predicted": total_pred,
            "times_hit": total_hit,
            "accuracy": round(total_hit / max(total_pred, 1), 4),
        })

    return {
        "model_type": model_type,
        "test_size": int(test_size),
        "train_size": int(len(train_data)),
        "max_num": int(max_num),
        "pick_count": int(pick_count),
        "total_hits": int(hit_array.sum()),
        "average_hits": round(float(hit_array.mean()), 4),
        "median_hits": round(float(np.median(hit_array)), 1),
        "std_hits": round(float(hit_array.std()), 4),
        "max_hits": int(hit_array.max()),
        "min_hits": int(hit_array.min()),
        "expected_random": round(float(expected_random), 4),
        "lift_vs_random": round(
            float(hit_array.mean()) / max(expected_random, 0.001), 4
        ),
        "hit_distribution": {str(int(k)): int(v) for k, v in sorted(hit_dist.items())},
        "hit_rate_nonzero": round(
            float(sum(1 for h in hit_counts if h > 0) / max(len(hit_counts), 1)), 4
        ),
        "best_hit_streak": int(best_streak),
        "worst_miss_streak": int(worst_streak),
        "rolling_avg_last5": rolling_avg[-5:] if rolling_avg else [],
        "top_predicted_numbers": top_numbers,
        "details": results,
    }


def _longest_streak(values: list[int], condition) -> int:
    """Find the longest consecutive streak where condition is True."""
    max_streak = 0
    current = 0
    for v in values:
        if condition(v):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def run_comparison_backtest(
    history: list[list[int]],
    max_num: int,
    pick_count: int,
    model_types: list[str] | None = None,
    test_size: int = 100,
) -> dict:
    """Run backtests for multiple models and compare results.

    Returns comparison summary with all model results side-by-side.
    """
    if model_types is None:
        model_types = ["frequency", "lstm", "dqn", "ensemble"]

    results = {}
    for mt in model_types:
        try:
            logger.info("[backtest] Running {} backtest...", mt)
            results[mt] = run_backtest(
                history, max_num, pick_count,
                model_type=mt, test_size=test_size,
            )
        except Exception as e:
            logger.warning("[backtest] {} failed: {}", mt, e)
            results[mt] = {"error": str(e)}

    # Build comparison summary
    summary = []
    for mt, r in results.items():
        if "error" in r:
            summary.append({"model": mt, "error": r["error"]})
        else:
            summary.append({
                "model": mt,
                "avg_hits": r["average_hits"],
                "std_hits": r["std_hits"],
                "max_hits": r["max_hits"],
                "expected_random": r["expected_random"],
                "lift_vs_random": r["lift_vs_random"],
                "hit_rate_nonzero": r["hit_rate_nonzero"],
            })

    return {
        "test_size": test_size,
        "total_draws": len(history),
        "comparison": summary,
        "details": {mt: r for mt, r in results.items() if "error" not in r},
    }
