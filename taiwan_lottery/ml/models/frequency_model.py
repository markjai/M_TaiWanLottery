"""Frequency-based statistical baseline model."""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from taiwan_lottery.ml.models.base_model import BaseLotteryModel


class FrequencyModel(BaseLotteryModel):
    """Statistical baseline: weighted frequency + gap + pair affinity."""

    model_type = "frequency"

    def __init__(self, max_num: int, pick_count: int):
        self.max_num = max_num
        self.pick_count = pick_count
        self.freq_weights: np.ndarray | None = None
        self.gap_weights: np.ndarray | None = None
        self.pair_affinity: dict = {}
        self._is_trained = False

    def train(self, history: list[list[int]], **kwargs) -> dict:
        windows = [10, 30, 50, 100, len(history)]
        decay_weights = [0.35, 0.25, 0.20, 0.12, 0.08]

        # Weighted frequency
        self.freq_weights = np.zeros(self.max_num, dtype=np.float64)
        for window, weight in zip(windows, decay_weights):
            recent = history[-window:] if len(history) >= window else history
            counter = Counter()
            for nums in recent:
                counter.update(nums)
            total = max(len(recent), 1)
            for num in range(1, self.max_num + 1):
                self.freq_weights[num - 1] += weight * counter.get(num, 0) / total

        # Gap-based weight (overdue numbers get a bonus)
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
            # Overdue bonus: gap_ratio > 1 means the number is overdue
            self.gap_weights[num - 1] = min(current_gap / max(avg_gap, 1), 3.0)

        # Pair affinity
        pair_counter = Counter()
        for nums in history:
            for pair in combinations(sorted(nums), 2):
                pair_counter[pair] += 1
        self.pair_affinity = {
            f"{a}-{b}": count / total_draws
            for (a, b), count in pair_counter.most_common(200)
        }

        self._is_trained = True
        return {
            "total_draws": total_draws,
            "model_type": self.model_type,
            "top5_freq": [
                int(i + 1) for i in np.argsort(self.freq_weights)[-5:][::-1]
            ],
        }

    def get_probabilities(self, history: list[list[int]]) -> np.ndarray:
        if not self._is_trained:
            self.train(history)

        # Combine frequency and gap weights
        combined = 0.65 * self.freq_weights + 0.35 * self.gap_weights
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
            "pair_affinity": self.pair_affinity,
        }
        path.write_text(json.dumps(data))

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.max_num = data["max_num"]
        self.pick_count = data["pick_count"]
        if data["freq_weights"]:
            self.freq_weights = np.array(data["freq_weights"], dtype=np.float64)
        if data["gap_weights"]:
            self.gap_weights = np.array(data["gap_weights"], dtype=np.float64)
        self.pair_affinity = data.get("pair_affinity", {})
        self._is_trained = True
