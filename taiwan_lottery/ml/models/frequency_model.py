"""Frequency-based statistical baseline model."""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from taiwan_lottery.ml.models.base_model import BaseLotteryModel
from taiwan_lottery.ml.features.feature_engineer import FeatureEngineer


class FrequencyModel(BaseLotteryModel):
    """Statistical baseline: weighted frequency + gap + hot momentum + zone + tail."""

    model_type = "frequency"

    def __init__(self, max_num: int, pick_count: int):
        self.max_num = max_num
        self.pick_count = pick_count
        self.freq_weights: np.ndarray | None = None
        self.gap_weights: np.ndarray | None = None
        self.hot_momentum: np.ndarray | None = None
        self.zone_balance: np.ndarray | None = None
        self.tail_affinity: np.ndarray | None = None
        self.mean_reversion: np.ndarray | None = None
        self.pair_affinity: dict = {}
        self.bias_boost: np.ndarray | None = None  # set by BiasDetector integration
        self._feature_weights = [0.40, 0.20, 0.15, 0.10, 0.08, 0.07]
        self._is_trained = False

    def train(self, history: list[list[int]], **kwargs) -> dict:
        fe = FeatureEngineer(self.max_num, self.pick_count)
        windows = [10, 30, 50, 100, len(history)]
        decay_weights = [0.35, 0.25, 0.20, 0.12, 0.08]

        # 1. Weighted frequency
        self.freq_weights = np.zeros(self.max_num, dtype=np.float64)
        for window, weight in zip(windows, decay_weights):
            recent = history[-window:] if len(history) >= window else history
            counter = Counter()
            for nums in recent:
                counter.update(nums)
            total = max(len(recent), 1)
            for num in range(1, self.max_num + 1):
                self.freq_weights[num - 1] += weight * counter.get(num, 0) / total

        # 2. Gap-based weight (overdue numbers get a bonus)
        self.gap_weights = np.zeros(self.max_num, dtype=np.float64)
        total_draws = len(history)
        for num in range(1, self.max_num + 1):
            last_seen = -1
            gaps = []
            for i, nums in enumerate(history):
                if num in nums:
                    if last_seen >= 0:
                        gaps.append(i - last_seen)
                    last_seen = i
            current_gap = total_draws - 1 - last_seen if last_seen >= 0 else total_draws
            avg_gap = sum(gaps) / len(gaps) if gaps else total_draws
            self.gap_weights[num - 1] = min(current_gap / max(avg_gap, 1), 3.0)

        # 3. Hot/Cold momentum — use streak as momentum signal
        hot_cold = fe.compute_hot_cold_features(history)  # (max_num, 3)
        # Combine: hot numbers get boost, cold numbers get bonus from mean reversion
        self.hot_momentum = np.zeros(self.max_num, dtype=np.float64)
        for num in range(self.max_num):
            is_hot, is_cold, streak = hot_cold[num]
            # Hot numbers: boost proportional to streak
            if is_hot:
                self.hot_momentum[num] = 1.0 + min(streak / 10.0, 1.0)
            # Cold numbers: smaller bonus (contrarian)
            elif is_cold:
                self.hot_momentum[num] = 0.8 + min(streak / 20.0, 0.5)
            else:
                self.hot_momentum[num] = 1.0

        # 4. Zone balance — numbers in under-represented zones get a bonus
        n_zones = 5
        zone_size = self.max_num / n_zones
        recent_50 = history[-50:] if len(history) >= 50 else history
        zone_counts = np.zeros(n_zones, dtype=np.float64)
        for nums in recent_50:
            for n in nums:
                z = min(int((n - 1) / zone_size), n_zones - 1)
                zone_counts[z] += 1
        zone_total = max(zone_counts.sum(), 1)
        zone_freq = zone_counts / zone_total
        expected_zone = 1.0 / n_zones

        self.zone_balance = np.zeros(self.max_num, dtype=np.float64)
        for num in range(1, self.max_num + 1):
            z = min(int((num - 1) / zone_size), n_zones - 1)
            # Under-represented zone => bonus
            self.zone_balance[num - 1] = max(expected_zone / max(zone_freq[z], 0.01), 0.5)

        # 5. Tail digit affinity
        tail_hist = fe.compute_tail_history_features(history, window=100)  # (10,)
        tail_total = max(tail_hist.sum(), 1)
        tail_freq = tail_hist / tail_total
        expected_tail = 0.1  # uniform across 10 digits

        self.tail_affinity = np.zeros(self.max_num, dtype=np.float64)
        for num in range(1, self.max_num + 1):
            digit = num % 10
            # Favor historically more frequent tail digits
            self.tail_affinity[num - 1] = tail_freq[digit] / max(expected_tail, 0.01)

        # 6. Mean reversion
        self.mean_reversion = fe.compute_mean_reversion_score(history).astype(np.float64)
        # Shift to positive range: positive = below average recently
        mr_min = self.mean_reversion.min()
        if mr_min < 0:
            self.mean_reversion = self.mean_reversion - mr_min
        mr_total = max(self.mean_reversion.sum(), 1e-9)
        self.mean_reversion = self.mean_reversion / mr_total * self.max_num

        # Pair affinity (unchanged)
        pair_counter = Counter()
        for nums in history:
            for pair in combinations(sorted(nums), 2):
                pair_counter[pair] += 1
        self.pair_affinity = {
            f"{a}-{b}": count / total_draws
            for (a, b), count in pair_counter.most_common(200)
        }

        # Auto-tune weights using recent validation if enough data
        if len(history) > 100:
            self._auto_tune_weights(history)

        self._is_trained = True
        return {
            "total_draws": total_draws,
            "model_type": self.model_type,
            "top5_freq": [
                int(i + 1) for i in np.argsort(self.freq_weights)[-5:][::-1]
            ],
            "feature_weights": dict(zip(
                ["freq", "gap", "hot_momentum", "zone_balance", "tail_affinity", "mean_reversion"],
                [round(w, 4) for w in self._feature_weights],
            )),
        }

    def _auto_tune_weights(self, history: list[list[int]]) -> None:
        """Simple grid search on last 30 draws to tune feature weights."""
        val_size = 30
        train_h = history[:-val_size]
        val_draws = history[-val_size:]

        # Temporarily compute features on train subset
        temp_model = FrequencyModel(self.max_num, self.pick_count)
        temp_model.freq_weights = self.freq_weights
        temp_model.gap_weights = self.gap_weights
        temp_model.hot_momentum = self.hot_momentum
        temp_model.zone_balance = self.zone_balance
        temp_model.tail_affinity = self.tail_affinity
        temp_model.mean_reversion = self.mean_reversion
        temp_model.bias_boost = self.bias_boost
        temp_model._is_trained = True

        best_hits = -1
        best_weights = self._feature_weights[:]

        # Test a few weight combinations
        candidates = [
            [0.40, 0.20, 0.15, 0.10, 0.08, 0.07],
            [0.35, 0.25, 0.15, 0.10, 0.08, 0.07],
            [0.45, 0.15, 0.15, 0.10, 0.08, 0.07],
            [0.35, 0.20, 0.20, 0.10, 0.08, 0.07],
            [0.30, 0.20, 0.20, 0.15, 0.08, 0.07],
            [0.40, 0.15, 0.15, 0.10, 0.10, 0.10],
        ]

        for weights in candidates:
            temp_model._feature_weights = weights
            total_hits = 0
            for i, actual in enumerate(val_draws):
                context = train_h + val_draws[:i]
                probs = temp_model.get_probabilities(context)
                top_k = set(int(idx + 1) for idx in np.argsort(probs)[-self.pick_count:])
                total_hits += len(top_k & set(actual))

            if total_hits > best_hits:
                best_hits = total_hits
                best_weights = weights[:]

        self._feature_weights = best_weights

    def get_probabilities(self, history: list[list[int]]) -> np.ndarray:
        if not self._is_trained:
            self.train(history)

        w = self._feature_weights
        # Normalize each component to same scale before combining
        def _norm(arr):
            total = arr.sum()
            return arr / total if total > 0 else np.ones_like(arr) / len(arr)

        combined = (
            w[0] * _norm(self.freq_weights)
            + w[1] * _norm(self.gap_weights)
            + w[2] * _norm(self.hot_momentum)
            + w[3] * _norm(self.zone_balance)
            + w[4] * _norm(self.tail_affinity)
            + w[5] * _norm(self.mean_reversion)
        )

        # Apply bias boost if available
        if self.bias_boost is not None:
            combined = combined * self.bias_boost

        # Normalize to probabilities
        total = combined.sum()
        if total > 0:
            combined = combined / total
        return combined.astype(np.float32)

    def predict(self, history: list[list[int]], n_sets: int = 1) -> list[list[int]]:
        probs = self.get_probabilities(history)
        results = []

        for _ in range(n_sets):
            # Sample without replacement using probabilities
            selected = []
            remaining_probs = probs.copy()

            for _ in range(self.pick_count):
                # Normalize remaining probabilities
                total = remaining_probs.sum()
                if total > 0:
                    p = remaining_probs / total
                else:
                    p = np.ones(self.max_num) / self.max_num

                idx = np.random.choice(self.max_num, p=p)
                selected.append(idx + 1)
                remaining_probs[idx] = 0  # Remove selected

            results.append(sorted(selected))

        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_num": self.max_num,
            "pick_count": self.pick_count,
            "freq_weights": self.freq_weights.tolist() if self.freq_weights is not None else None,
            "gap_weights": self.gap_weights.tolist() if self.gap_weights is not None else None,
            "hot_momentum": self.hot_momentum.tolist() if self.hot_momentum is not None else None,
            "zone_balance": self.zone_balance.tolist() if self.zone_balance is not None else None,
            "tail_affinity": self.tail_affinity.tolist() if self.tail_affinity is not None else None,
            "mean_reversion": self.mean_reversion.tolist() if self.mean_reversion is not None else None,
            "pair_affinity": self.pair_affinity,
            "feature_weights": self._feature_weights,
        }
        path.write_text(json.dumps(data))

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.max_num = data["max_num"]
        self.pick_count = data["pick_count"]
        if data.get("freq_weights"):
            self.freq_weights = np.array(data["freq_weights"], dtype=np.float64)
        if data.get("gap_weights"):
            self.gap_weights = np.array(data["gap_weights"], dtype=np.float64)
        if data.get("hot_momentum"):
            self.hot_momentum = np.array(data["hot_momentum"], dtype=np.float64)
        if data.get("zone_balance"):
            self.zone_balance = np.array(data["zone_balance"], dtype=np.float64)
        if data.get("tail_affinity"):
            self.tail_affinity = np.array(data["tail_affinity"], dtype=np.float64)
        if data.get("mean_reversion"):
            self.mean_reversion = np.array(data["mean_reversion"], dtype=np.float64)
        self.pair_affinity = data.get("pair_affinity", {})
        self._feature_weights = data.get("feature_weights", [0.40, 0.20, 0.15, 0.10, 0.08, 0.07])
        self._is_trained = True
