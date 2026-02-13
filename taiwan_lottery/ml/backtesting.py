"""Walk-forward backtesting engine for lottery prediction models.

Trains a model on a training window, then evaluates prediction accuracy
on each subsequent draw using all data available up to that point.
"""

from collections import Counter

import numpy as np
from loguru import logger
from scipy import stats as sp_stats

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
    bias_boost: "np.ndarray | None" = None,
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
    if bias_boost is not None and hasattr(model, "bias_boost"):
        model.bias_boost = bias_boost
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
            if bias_boost is not None and hasattr(model, "bias_boost"):
                model.bias_boost = bias_boost
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
    # For bingo: 20 numbers drawn from 80, so draw_count != pick_count
    if max_num >= 80:
        draw_count = 20  # bingo draws 20 numbers per period
        expected_random = pick_count * draw_count / max_num
    else:
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

    # Monte Carlo baseline simulation
    mc = _monte_carlo_baseline(test_draws, max_num, pick_count)

    # Statistical significance tests
    sig = _statistical_significance(hit_array, expected_random)

    # Quarterly performance analysis
    quarterly = _quarterly_performance(hit_counts)

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
        # Phase 1 additions
        "monte_carlo_avg": mc["random_avg_hits"],
        "monte_carlo_std": mc["random_std"],
        "p_value": sig["p_value"],
        "effect_size": sig["effect_size"],
        "confidence_interval_95": sig["confidence_interval_95"],
        "quarterly_performance": quarterly,
        "is_significant": sig["p_value"] < 0.05,
        "details": results,
    }


def _monte_carlo_baseline(
    test_draws: list[list[int]],
    max_num: int,
    pick_count: int,
    n_simulations: int = 10000,
) -> dict:
    """Simulate random picking to establish a baseline for comparison.

    For each test draw, randomly pick `pick_count` numbers `n_simulations` times
    and compute how many hits a random strategy would get on average.
    """
    rng = np.random.default_rng(42)
    all_nums = np.arange(1, max_num + 1)

    # For each simulation run, compute total hits across all test draws
    sim_total_hits = np.zeros(n_simulations, dtype=np.int64)
    for actual in test_draws:
        actual_set = set(actual)
        for s in range(n_simulations):
            random_pick = rng.choice(all_nums, size=pick_count, replace=False)
            sim_total_hits[s] += len(actual_set & set(random_pick))

    # Average hits per draw for each simulation
    sim_avg_per_draw = sim_total_hits / len(test_draws)

    return {
        "random_avg_hits": round(float(sim_avg_per_draw.mean()), 4),
        "random_std": round(float(sim_avg_per_draw.std()), 4),
        "random_hit_distribution": {
            str(int(k)): int(v)
            for k, v in sorted(Counter(sim_total_hits).items())
        },
    }


def _statistical_significance(hit_array: np.ndarray, expected_random: float) -> dict:
    """Compute statistical significance of model hits vs random expectation.

    Uses one-sample t-test and Cohen's d effect size.
    """
    n = len(hit_array)
    model_mean = float(hit_array.mean())
    model_std = float(hit_array.std(ddof=1)) if n > 1 else 0.0

    # One-sample t-test: is model mean significantly different from expected_random?
    if n > 1 and model_std > 0:
        t_stat, p_two = sp_stats.ttest_1samp(hit_array, expected_random)
        # One-sided p-value (model > random)
        p_value = float(p_two / 2) if t_stat > 0 else 1.0 - float(p_two / 2)
    else:
        p_value = 1.0

    # Cohen's d effect size
    if model_std > 0:
        effect_size = (model_mean - expected_random) / model_std
    else:
        effect_size = 0.0

    # 95% confidence interval for model mean
    if n > 1:
        se = model_std / np.sqrt(n)
        ci_low = model_mean - 1.96 * se
        ci_high = model_mean + 1.96 * se
    else:
        ci_low = ci_high = model_mean

    return {
        "p_value": round(p_value, 6),
        "effect_size": round(effect_size, 4),
        "confidence_interval_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
    }


def _quarterly_performance(hit_counts: list[int]) -> list[float]:
    """Split test period into 4 quarters and compute average hits for each."""
    n = len(hit_counts)
    if n < 4:
        return [round(float(np.mean(hit_counts)), 4)] if hit_counts else []

    quarter_size = n // 4
    quarters = []
    for q in range(4):
        start = q * quarter_size
        end = start + quarter_size if q < 3 else n
        q_hits = hit_counts[start:end]
        quarters.append(round(float(np.mean(q_hits)), 4))
    return quarters


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
                "p_value": r.get("p_value"),
                "is_significant": r.get("is_significant"),
                "effect_size": r.get("effect_size"),
            })

    return {
        "test_size": test_size,
        "total_draws": len(history),
        "comparison": summary,
        "details": {mt: r for mt, r in results.items() if "error" not in r},
    }
