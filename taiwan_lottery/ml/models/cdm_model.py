"""Compound-Dirichlet-Multinomial (CDM) prediction model.

Based on arXiv:2403.12836 — uses Bayesian inference with a Dirichlet prior
and Multinomial likelihood to derive a posterior predictive distribution.

Core prediction formula:
    Pred(x_j) = M * (alpha_j + n_j) / sum(alpha_j + n_j)
"""

import json
from pathlib import Path

import numpy as np

from taiwan_lottery.ml.features.feature_engineer import FeatureEngineer
from taiwan_lottery.ml.models.base_model import BaseLotteryModel

# Euler-Mascheroni constant
_EULER_GAMMA = 0.57721566490153286


class CDMModel(BaseLotteryModel):
    """CDM (Compound-Dirichlet-Multinomial) lottery prediction model."""

    model_type = "cdm"

    def __init__(self, max_num: int, pick_count: int):
        self.max_num = max_num
        self.pick_count = pick_count
        self.alpha: np.ndarray | None = None
        self.estimation_method: str = "feature"
        self.alpha_scale: float = 2000.0
        self._feature_weights: list = [0.30, 0.25, 0.20, 0.10, 0.08, 0.07]
        # [freq, gap, hot_momentum, zone_balance, tail_affinity, mean_reversion]
        self._is_trained = False

    def train(self, history: list[list[int]], **kwargs) -> dict:
        method = kwargs.get("estimation_method", "feature")
        if "alpha_scale" in kwargs:
            self.alpha_scale = float(kwargs["alpha_scale"])
        if "feature_weights" in kwargs:
            self._feature_weights = list(kwargs["feature_weights"])

        n = len(history)
        K = self.max_num
        M = self.pick_count

        # Build binary matrix X[n, K]: X[i,j]=1 if number j+1 appeared in draw i
        X = np.zeros((n, K), dtype=np.float64)
        for i, draw in enumerate(history):
            for num in draw:
                if 1 <= num <= K:
                    X[i, num - 1] = 1

        # Column sums: n_j = how many times number j+1 appeared
        n_j = X.sum(axis=0)  # shape (K,)

        actual_method = method

        if method == "feature":
            self.alpha = self._estimate_feature(history, K, M)
        elif method == "mle":
            try:
                self.alpha = self._estimate_mle(X, n_j, n, K, M)
            except Exception:
                actual_method = "mom"
                self.alpha = self._estimate_mom(n_j, n)
        elif method == "diagonal":
            if n == K:
                self.alpha = np.diag(X).copy()
                # Ensure no zero alphas
                self.alpha = np.maximum(self.alpha, 1e-6)
            else:
                actual_method = "mom"
                self.alpha = self._estimate_mom(n_j, n)
        else:
            actual_method = "mom"
            self.alpha = self._estimate_mom(n_j, n)

        self.estimation_method = actual_method
        self._is_trained = True

        alpha_0 = float(self.alpha.sum())
        return {
            "model_type": self.model_type,
            "method": actual_method,
            "alpha_0": round(alpha_0, 4),
            "alpha_range": [round(float(self.alpha.min()), 6),
                            round(float(self.alpha.max()), 6)],
            "total_draws": n,
        }

    @staticmethod
    def _estimate_mle(
        X: np.ndarray, n_j: np.ndarray, n: int, K: int, M: int,
    ) -> np.ndarray:
        """MLE estimation of Dirichlet parameters.

        alpha_0 = n*(K-1)*gamma / [n*sum(f_j*ln(f_j)) - sum(f_j * sum_i(ln(X_ij)))]
        alpha_j = alpha_0 * f_j
        """
        eps = 1e-10
        f_j = n_j / max(n * M, 1)
        f_j = np.maximum(f_j, eps)  # avoid log(0)

        # sum of f_j * ln(f_j)
        term1 = np.sum(f_j * np.log(f_j))

        # For each column j, compute sum_i(ln(X[i,j])) — X is binary so
        # ln(0)=-inf, ln(1)=0. We use ln(max(X,eps)) to handle zeros.
        log_X = np.log(np.maximum(X, eps))  # (n, K)
        sum_log_X = log_X.sum(axis=0)  # (K,)

        # sum of f_j * sum_i(ln(X_ij))
        term2 = np.sum(f_j * sum_log_X)

        denominator = n * term1 - term2
        if abs(denominator) < eps or denominator <= 0:
            raise ValueError("MLE denominator too small or non-positive")

        alpha_0 = n * (K - 1) * _EULER_GAMMA / denominator
        if alpha_0 <= 0 or not np.isfinite(alpha_0):
            raise ValueError(f"Invalid alpha_0: {alpha_0}")

        alpha = alpha_0 * f_j
        alpha = np.maximum(alpha, eps)
        return alpha

    @staticmethod
    def _estimate_mom(n_j: np.ndarray, n: int) -> np.ndarray:
        """Method of Moments: alpha_j = n_j / n."""
        eps = 1e-6
        alpha = n_j / max(n, 1)
        return np.maximum(alpha, eps)

    def _estimate_feature(
        self, history: list[list[int]], K: int, M: int,
    ) -> np.ndarray:
        """Feature-weighted Dirichlet prior: α_j = α₀ × feature_score_j.

        Combines 6 normalized feature vectors using self._feature_weights,
        then scales by self.alpha_scale to produce the Dirichlet α vector.
        """
        fe = FeatureEngineer(K, M)
        w = self._feature_weights
        eps = 1e-8

        def _normalize(v: np.ndarray) -> np.ndarray:
            """Normalize a vector to sum=1, handle all-zero gracefully."""
            v = np.maximum(v, 0.0)
            s = v.sum()
            if s > eps:
                return v / s
            return np.ones(K, dtype=np.float64) / K

        # 1. freq — multi-window decay-weighted frequency
        freq_raw = fe.compute_frequency_features(history, windows=[10, 30, 50, 100])
        # Decay weights: recent windows matter more
        decay = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        freq_score = (freq_raw * decay).sum(axis=1)  # shape (K,)

        # 2. gap — gap_ratio: higher ratio = more overdue = higher prior
        gap_features = fe.compute_gap_features(history)  # (K, 3)
        gap_score = np.minimum(gap_features[:, 2], 3.0)  # cap at 3.0

        # 3. hot_momentum — streak-based momentum score
        hot_cold = fe.compute_hot_cold_features(history)  # (K, 3)
        is_hot = hot_cold[:, 0]    # binary
        is_cold = hot_cold[:, 1]   # binary
        streak = hot_cold[:, 2]    # length
        # Hot numbers get a boost proportional to streak, cold get a mild boost
        momentum = np.ones(K, dtype=np.float64)
        hot_mask = is_hot > 0.5
        cold_mask = is_cold > 0.5
        momentum[hot_mask] = 1.0 + streak[hot_mask] / 10.0
        momentum[cold_mask] = 0.8 + streak[cold_mask] / 20.0

        # 4. zone_balance — under-represented zones get bonus
        zone_balance = np.ones(K, dtype=np.float64)
        n_zones = 5
        zone_size = K / n_zones
        if len(history) >= 10:
            recent_50 = history[-50:] if len(history) >= 50 else history
            zone_counts = np.zeros(n_zones, dtype=np.float64)
            total_nums = 0
            for nums in recent_50:
                for num in nums:
                    if 1 <= num <= K:
                        z = min(int((num - 1) / zone_size), n_zones - 1)
                        zone_counts[z] += 1
                        total_nums += 1
            if total_nums > 0:
                zone_freq = zone_counts / total_nums
                expected = 1.0 / n_zones
                for num in range(1, K + 1):
                    z = min(int((num - 1) / zone_size), n_zones - 1)
                    if zone_freq[z] > eps:
                        zone_balance[num - 1] = max(expected / zone_freq[z], 0.5)
                    else:
                        zone_balance[num - 1] = 2.0

        # 5. tail_affinity — map tail digit distribution back to numbers
        tail_hist = fe.compute_tail_history_features(history, window=100)  # (10,)
        tail_score = np.zeros(K, dtype=np.float64)
        for num in range(1, K + 1):
            tail_score[num - 1] = max(float(tail_hist[num % 10]), eps)

        # 6. mean_reversion — positive = below average, expected to rise
        mr_raw = fe.compute_mean_reversion_score(history)  # (K,)
        # Shift to all-positive: add abs(min) + small offset
        mr_min = float(mr_raw.min())
        mean_reversion = mr_raw.astype(np.float64) - mr_min + 0.01

        # Combine: weighted sum of normalized features
        components = [freq_score, gap_score, momentum,
                      zone_balance, tail_score, mean_reversion]
        feature_score = np.zeros(K, dtype=np.float64)
        for i, comp in enumerate(components):
            weight = w[i] if i < len(w) else 0.0
            feature_score += weight * _normalize(comp.astype(np.float64))

        # Normalize feature_score to sum=1, then scale
        feature_score = _normalize(feature_score)
        alpha = self.alpha_scale * feature_score
        return np.maximum(alpha, eps)

    def get_probabilities(self, history: list[list[int]]) -> np.ndarray:
        if not self._is_trained or self.alpha is None:
            self.train(history)

        K = self.max_num
        M = self.pick_count

        # Recompute n_j from current history (walk-forward)
        n_j = np.zeros(K, dtype=np.float64)
        for draw in history:
            for num in draw:
                if 1 <= num <= K:
                    n_j[num - 1] += 1

        # Posterior predictive: pred_j = M * (alpha_j + n_j) / sum(alpha_j + n_j)
        posterior = self.alpha + n_j
        total = posterior.sum()
        if total > 0:
            probs = M * posterior / total
        else:
            probs = np.ones(K) / K

        # Normalize to probability distribution (sum=1)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum

        return probs.astype(np.float32)

    def predict(self, history: list[list[int]], n_sets: int = 1) -> list[list[int]]:
        probs = self.get_probabilities(history)
        results = []

        for _ in range(n_sets):
            selected = []
            remaining_probs = probs.copy()

            for _ in range(self.pick_count):
                total = remaining_probs.sum()
                if total > 0:
                    p = remaining_probs / total
                else:
                    p = np.ones(self.max_num) / self.max_num

                idx = np.random.choice(self.max_num, p=p)
                selected.append(idx + 1)
                remaining_probs[idx] = 0

            results.append(sorted(selected))

        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_num": self.max_num,
            "pick_count": self.pick_count,
            "estimation_method": self.estimation_method,
            "alpha": self.alpha.tolist() if self.alpha is not None else None,
            "alpha_scale": self.alpha_scale,
            "feature_weights": self._feature_weights,
        }
        path.write_text(json.dumps(data))

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self.max_num = data["max_num"]
        self.pick_count = data["pick_count"]
        self.estimation_method = data.get("estimation_method", "mom")
        self.alpha_scale = data.get("alpha_scale", 2000.0)
        self._feature_weights = data.get("feature_weights",
                                         [0.30, 0.25, 0.20, 0.10, 0.08, 0.07])
        if data.get("alpha"):
            self.alpha = np.array(data["alpha"], dtype=np.float64)
        self._is_trained = True
