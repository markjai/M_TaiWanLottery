"""Feature engineering for lottery ML models."""

import math
from collections import Counter
from itertools import combinations

import numpy as np


class FeatureEngineer:
    """Extract features from lottery draw history for ML models."""

    def __init__(self, max_num: int, pick_count: int):
        self.max_num = max_num
        self.pick_count = pick_count

    # ── frequency features ────────────────────────────────────────────

    def compute_frequency_features(
        self, history: list[list[int]], windows: list[int] = (10, 30, 50, 100)
    ) -> np.ndarray:
        """Compute frequency features for each number across different windows.

        Returns shape: (max_num, len(windows))
        """
        features = np.zeros((self.max_num, len(windows)), dtype=np.float32)

        for w_idx, window in enumerate(windows):
            recent = history[-window:] if len(history) >= window else history
            counter = Counter()
            for nums in recent:
                counter.update(nums)
            total = len(recent)
            for num in range(1, self.max_num + 1):
                features[num - 1, w_idx] = counter.get(num, 0) / max(total, 1)

        return features

    # ── gap / 遺漏 features ──────────────────────────────────────────

    def compute_gap_features(self, history: list[list[int]]) -> np.ndarray:
        """Compute gap (遺漏) features: current_gap, avg_gap, gap_ratio.

        Returns shape: (max_num, 3)
        """
        total = len(history)
        features = np.zeros((self.max_num, 3), dtype=np.float32)

        for num in range(1, self.max_num + 1):
            last_seen = -1
            gaps = []

            for i, nums in enumerate(history):
                if num in nums:
                    if last_seen >= 0:
                        gaps.append(i - last_seen)
                    last_seen = i

            current_gap = total - 1 - last_seen if last_seen >= 0 else total
            avg_gap = sum(gaps) / len(gaps) if gaps else float(total)
            gap_ratio = current_gap / avg_gap if avg_gap > 0 else 0

            features[num - 1] = [current_gap, avg_gap, gap_ratio]

        return features

    # ── AC 值 (Arithmetic Complexity) ────────────────────────────────

    def compute_ac_value(self, numbers: list[int]) -> int:
        """Compute AC value — number of distinct differences between all pairs.

        AC = distinct_diffs - (n - 1), measures arithmetic spread complexity.
        High AC = numbers are spread randomly; Low AC = arithmetic pattern.
        """
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)
        if n < 2:
            return 0
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(sorted_nums[j] - sorted_nums[i])
        return len(diffs) - (n - 1)

    # ── tail digit distribution ──────────────────────────────────────

    def compute_tail_features(self, numbers: list[int]) -> np.ndarray:
        """Count tail digits (0‑9) of the drawn numbers.

        Returns shape: (10,) — count of each tail digit.
        """
        counts = np.zeros(10, dtype=np.float32)
        for n in numbers:
            counts[n % 10] += 1
        return counts

    def compute_tail_history_features(
        self, history: list[list[int]], window: int = 50
    ) -> np.ndarray:
        """Average tail digit distribution over recent draws.

        Returns shape: (10,)
        """
        recent = history[-window:] if len(history) >= window else history
        if not recent:
            return np.zeros(10, dtype=np.float32)
        acc = np.zeros(10, dtype=np.float32)
        for nums in recent:
            acc += self.compute_tail_features(nums)
        return acc / len(recent)

    # ── consecutive number analysis ──────────────────────────────────

    def compute_consecutive_features(self, numbers: list[int]) -> np.ndarray:
        """Detailed consecutive number features.

        Returns: [consecutive_pairs, max_consecutive_run, has_triple]
        """
        sorted_nums = sorted(numbers)
        pairs = 0
        max_run = 1
        current_run = 1
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i + 1] - sorted_nums[i] == 1:
                pairs += 1
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        has_triple = 1.0 if max_run >= 3 else 0.0
        return np.array([pairs, max_run, has_triple], dtype=np.float32)

    # ── zone / sector distribution ───────────────────────────────────

    def compute_zone_features(self, numbers: list[int], n_zones: int = 5) -> np.ndarray:
        """Split number range into n zones and count distribution.

        Returns shape: (n_zones,)
        """
        zone_size = self.max_num / n_zones
        counts = np.zeros(n_zones, dtype=np.float32)
        for n in numbers:
            zone_idx = min(int((n - 1) / zone_size), n_zones - 1)
            counts[zone_idx] += 1
        return counts

    # ── hot / cold streak features ───────────────────────────────────

    def compute_hot_cold_features(
        self, history: list[list[int]], hot_window: int = 10, cold_window: int = 50
    ) -> np.ndarray:
        """For each number: [is_hot, is_cold, streak_length].

        hot = appeared >= 2x expected in hot_window.
        cold = appeared <= 0.5x expected in cold_window.
        streak = consecutive draws since last appearance change.

        Returns shape: (max_num, 3)
        """
        expected_rate = self.pick_count / self.max_num
        features = np.zeros((self.max_num, 3), dtype=np.float32)

        recent_hot = history[-hot_window:] if len(history) >= hot_window else history
        recent_cold = history[-cold_window:] if len(history) >= cold_window else history

        hot_counter = Counter()
        for nums in recent_hot:
            hot_counter.update(nums)
        cold_counter = Counter()
        for nums in recent_cold:
            cold_counter.update(nums)

        for num in range(1, self.max_num + 1):
            hot_freq = hot_counter.get(num, 0) / max(len(recent_hot), 1)
            cold_freq = cold_counter.get(num, 0) / max(len(recent_cold), 1)

            is_hot = 1.0 if hot_freq >= expected_rate * 2 else 0.0
            is_cold = 1.0 if cold_freq <= expected_rate * 0.5 else 0.0

            # Streak: how many recent consecutive draws it appeared / didn't appear
            streak = 0
            if history:
                last_appeared = num in history[-1]
                streak = 1
                for draw in reversed(history[:-1]):
                    if (num in draw) == last_appeared:
                        streak += 1
                    else:
                        break

            features[num - 1] = [is_hot, is_cold, streak]

        return features

    # ── sum / range distribution features ────────────────────────────

    def compute_sum_range_features(self, history: list[list[int]], window: int = 50) -> np.ndarray:
        """Statistical features of sum and span distributions over recent draws.

        Returns: [sum_mean, sum_std, span_mean, span_std, sum_trend]
        """
        recent = history[-window:] if len(history) >= window else history
        if len(recent) < 2:
            return np.zeros(5, dtype=np.float32)

        sums = [sum(nums) for nums in recent]
        spans = [max(nums) - min(nums) if nums else 0 for nums in recent]

        # Trend: linear slope of sum over time (positive = increasing)
        x = np.arange(len(sums), dtype=np.float32)
        y = np.array(sums, dtype=np.float32)
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        return np.array([
            np.mean(sums), np.std(sums),
            np.mean(spans), np.std(spans),
            slope,
        ], dtype=np.float32)

    # ── aggregate features (enhanced) ────────────────────────────────

    def compute_aggregate_features(self, numbers: list[int]) -> np.ndarray:
        """Compute aggregate features for a single draw.

        Returns: [sum, odd_count, even_count, span, consecutive_pairs,
                  low_count, mid_count, high_count, ac_value,
                  max_consecutive_run, tail_diversity]
        (11 dims)
        """
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)

        total_sum = sum(sorted_nums)
        odd_count = sum(1 for x in sorted_nums if x % 2 == 1)
        even_count = n - odd_count
        span = sorted_nums[-1] - sorted_nums[0] if n > 0 else 0

        # Consecutive pairs
        consecutive = 0
        max_run = 1
        current_run = 1
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i + 1] - sorted_nums[i] == 1:
                consecutive += 1
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        # Low / Mid / High distribution
        third = self.max_num / 3
        low = sum(1 for x in sorted_nums if x <= third)
        mid = sum(1 for x in sorted_nums if third < x <= 2 * third)
        high = sum(1 for x in sorted_nums if x > 2 * third)

        # AC value
        ac = self.compute_ac_value(numbers)

        # Tail digit diversity (how many distinct tail digits)
        tail_diversity = len(set(n % 10 for n in numbers))

        return np.array(
            [total_sum, odd_count, even_count, span, consecutive,
             low, mid, high, ac, max_run, tail_diversity],
            dtype=np.float32,
        )

    # ── time features ────────────────────────────────────────────────

    def compute_time_features(self, day_of_week: int, month: int) -> np.ndarray:
        """Sin/cos encoding for temporal features.

        Returns: [dow_sin, dow_cos, month_sin, month_cos]
        """
        dow_sin = math.sin(2 * math.pi * day_of_week / 7)
        dow_cos = math.cos(2 * math.pi * day_of_week / 7)
        month_sin = math.sin(2 * math.pi * (month - 1) / 12)
        month_cos = math.cos(2 * math.pi * (month - 1) / 12)
        return np.array([dow_sin, dow_cos, month_sin, month_cos], dtype=np.float32)

    # ── pair affinity ────────────────────────────────────────────────

    def compute_pair_affinity(
        self, history: list[list[int]], top_n: int = 50
    ) -> dict[tuple[int, int], float]:
        """Compute pair co-occurrence affinity scores."""
        pair_counter = Counter()
        total = len(history)

        for nums in history:
            for pair in combinations(sorted(nums), 2):
                pair_counter[pair] += 1

        affinity = {}
        for pair, count in pair_counter.most_common(top_n):
            affinity[pair] = count / total

        return affinity

    # ── multi-hot encoding ───────────────────────────────────────────

    def build_multi_hot(self, numbers: list[int]) -> np.ndarray:
        """Convert a draw to a multi-hot vector of shape (max_num,)."""
        vec = np.zeros(self.max_num, dtype=np.float32)
        for n in numbers:
            if 1 <= n <= self.max_num:
                vec[n - 1] = 1.0
        return vec

    # ── sequence features for Transformer / LSTM ─────────────────────

    def build_sequence_features(
        self, history: list[list[int]], seq_len: int = 20
    ) -> np.ndarray:
        """Build sequence of multi-hot + enhanced aggregate features.

        Returns shape: (seq_len, max_num + 11)
        """
        recent = history[-seq_len:] if len(history) >= seq_len else history

        # Zero-pad if shorter (instead of repeating first draw)
        pad_len = seq_len - len(recent)

        features = []
        for draw in recent:
            multi_hot = self.build_multi_hot(draw)
            agg = self.compute_aggregate_features(draw)
            features.append(np.concatenate([multi_hot, agg]))

        feature_dim = self.max_num + 11  # enhanced aggregate has 11 dims
        # Zero-pad at the beginning
        if pad_len > 0:
            padding = [np.zeros(feature_dim, dtype=np.float32)] * pad_len
            features = padding + features

        return np.array(features, dtype=np.float32)

    # ── context features for enhanced Transformer input ──────────────

    def precompute_context_features(
        self, history: list[list[int]], context_window: int = 100
    ) -> np.ndarray:
        """Precompute per-timestep context features for the entire history.

        For each timestep t, computes context from history[max(0,t-context_window):t].

        Returns shape: (len(history), context_dim) where context_dim = max_num*2 + max_num + max_num + 10 + 5
            = frequency(2 windows) * max_num + gap_ratio * max_num + is_hot * max_num + tail_hist(10) + sum_range(5)
            = max_num * 4 + 15   (for 2-window freq: max_num*2, gap: max_num, hot: max_num)
        """
        n = len(history)
        # context_dim: freq(2 windows)*max_num + gap_ratio*max_num + is_hot*max_num + tail_hist(10) + sum_range(5)
        context_dim = self.max_num * 4 + 15
        result = np.zeros((n, context_dim), dtype=np.float32)

        for t in range(n):
            start = max(0, t - context_window)
            ctx_history = history[start:t] if t > 0 else [history[0]]

            offset = 0
            # Frequency with 2 windows (short=10, long=50)
            freq = self.compute_frequency_features(ctx_history, windows=[10, 50])  # (max_num, 2)
            result[t, offset:offset + self.max_num * 2] = freq.flatten()
            offset += self.max_num * 2

            # Gap ratio
            gap = self.compute_gap_features(ctx_history)  # (max_num, 3)
            result[t, offset:offset + self.max_num] = gap[:, 2]  # gap_ratio column
            offset += self.max_num

            # Is hot
            hot_cold = self.compute_hot_cold_features(ctx_history)  # (max_num, 3)
            result[t, offset:offset + self.max_num] = hot_cold[:, 0]  # is_hot column
            offset += self.max_num

            # Tail histogram
            tail_hist = self.compute_tail_history_features(ctx_history, window=50)  # (10,)
            result[t, offset:offset + 10] = tail_hist
            offset += 10

            # Sum/range stats
            sum_range = self.compute_sum_range_features(ctx_history, window=50)  # (5,)
            result[t, offset:offset + 5] = sum_range
            offset += 5

        return result

    def build_enhanced_sequence_features(
        self, history: list[list[int]], seq_len: int = 30
    ) -> np.ndarray:
        """Build enhanced sequence: multi_hot + context + aggregate per timestep.

        For inference use (single sequence, not batch).

        Returns shape: (seq_len, max_num + context_dim + 11)
            where context_dim = max_num * 4 + 15
            total = max_num * 5 + 26  (e.g. 80*5+26=426 for bingo)
        """
        context_dim = self.max_num * 4 + 15

        # Take last seq_len draws
        recent = history[-seq_len:] if len(history) >= seq_len else history
        pad_len = seq_len - len(recent)

        # Compute context features for the recent portion
        # We need history up to each point for context
        features = []
        start_idx = len(history) - len(recent)
        for i, draw in enumerate(recent):
            multi_hot = self.build_multi_hot(draw)                        # (max_num,)
            agg = self.compute_aggregate_features(draw)                   # (11,)

            # Context: use history up to this point
            t = start_idx + i
            ctx_start = max(0, t - 100)
            ctx_history = history[ctx_start:t] if t > 0 else [history[0]]

            freq = self.compute_frequency_features(ctx_history, windows=[10, 50]).flatten()
            gap_ratio = self.compute_gap_features(ctx_history)[:, 2]
            is_hot = self.compute_hot_cold_features(ctx_history)[:, 0]
            tail_hist = self.compute_tail_history_features(ctx_history, window=50)
            sum_range = self.compute_sum_range_features(ctx_history, window=50)

            context = np.concatenate([freq, gap_ratio, is_hot, tail_hist, sum_range])
            features.append(np.concatenate([multi_hot, context, agg]))

        feature_dim = self.max_num + context_dim + 11
        if pad_len > 0:
            padding = [np.zeros(feature_dim, dtype=np.float32)] * pad_len
            features = padding + features

        return np.array(features, dtype=np.float32)

    # ── global context features ──────────────────────────────────────

    def build_global_context(self, history: list[list[int]]) -> np.ndarray:
        """Build a global context vector summarizing full history state.

        Includes: frequency(4w) flat + gap(3) flat + hot_cold(3) flat
                + sum_range(5) + tail_hist(10)
        Returns shape: (max_num*10 + 15,)
        """
        freq = self.compute_frequency_features(history).flatten()       # max_num * 4
        gap = self.compute_gap_features(history).flatten()              # max_num * 3
        hot_cold = self.compute_hot_cold_features(history).flatten()    # max_num * 3
        sum_range = self.compute_sum_range_features(history)            # 5
        tail_hist = self.compute_tail_history_features(history)         # 10

        return np.concatenate([freq, gap, hot_cold, sum_range, tail_hist]).astype(np.float32)
