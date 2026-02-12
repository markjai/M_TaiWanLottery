"""Ensemble model — combines frequency, LSTM, and DQN predictions."""

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from taiwan_lottery.ml.models.base_model import BaseLotteryModel
from taiwan_lottery.ml.models.frequency_model import FrequencyModel
from taiwan_lottery.ml.models.lstm_model import LSTMModel
from taiwan_lottery.ml.models.dqn_model import DQNModel


class EnsembleModel(BaseLotteryModel):
    """Weighted ensemble of frequency, LSTM, and DQN models.

    Weights are optimized using scipy to maximize diversity-weighted hit rate.
    """

    model_type = "ensemble"

    def __init__(self, max_num: int, pick_count: int, device: str = "cpu"):
        self.max_num = max_num
        self.pick_count = pick_count
        self.device = device

        self.frequency_model = FrequencyModel(max_num, pick_count)
        self.lstm_model = LSTMModel(max_num, pick_count, device)
        self.dqn_model = DQNModel(max_num, pick_count, device)

        # Default weights: [freq, lstm, dqn]
        self.weights = np.array([0.4, 0.35, 0.25], dtype=np.float64)
        self._models_trained = {"frequency": False, "lstm": False, "dqn": False}

    def train(self, history: list[list[int]], **kwargs) -> dict:
        metrics = {}

        # Train each sub-model
        try:
            m = self.frequency_model.train(history)
            metrics["frequency"] = m
            self._models_trained["frequency"] = True
        except Exception as e:
            metrics["frequency_error"] = str(e)

        try:
            m = self.lstm_model.train(history, **kwargs)
            metrics["lstm"] = m
            self._models_trained["lstm"] = "error" not in m
        except Exception as e:
            metrics["lstm_error"] = str(e)

        try:
            m = self.dqn_model.train(history, **kwargs)
            metrics["dqn"] = m
            self._models_trained["dqn"] = "error" not in m
        except Exception as e:
            metrics["dqn_error"] = str(e)

        # Optimize weights using validation set
        if sum(self._models_trained.values()) >= 2:
            self._optimize_weights(history)

        metrics["ensemble_weights"] = self.weights.tolist()
        return metrics

    def _optimize_weights(self, history: list[list[int]]):
        """Optimize ensemble weights on the last 15% of data.

        Uses grid search over weight combinations (exhaustive for 3 models),
        with Nelder-Mead as fallback refinement.
        """
        split = int(len(history) * 0.85)
        val_history = history[split:]
        train_history = history[:split]

        if len(val_history) < 5:
            return

        # Limit validation samples for high-frequency games (bingo)
        max_val_samples = 500
        if len(val_history) > max_val_samples:
            val_history = val_history[-max_val_samples:]
            train_history = history[:len(history) - max_val_samples]

        def compute_score(w):
            w = np.abs(w)
            w_sum = w.sum()
            if w_sum == 0:
                return 0.0
            w = w / w_sum
            total_hits = 0

            for i in range(len(val_history)):
                context = train_history + val_history[:i] if i > 0 else train_history
                # Limit context window for efficiency
                if len(context) > 1000:
                    context = context[-1000:]
                target = set(val_history[i])

                probs = self._weighted_probs(context, w)
                top_k = np.argsort(probs)[-self.pick_count:]
                predicted = set(idx + 1 for idx in top_k)
                total_hits += len(predicted & target)

            return total_hits / len(val_history)

        # Grid search: step=0.05 for 3 weights summing to 1.0
        best_score = -1.0
        best_weights = self.weights.copy()
        step = 0.05
        for w0 in np.arange(0.05, 0.95, step):
            for w1 in np.arange(0.05, 0.95 - w0, step):
                w2 = 1.0 - w0 - w1
                if w2 < 0.05:
                    continue
                w = np.array([w0, w1, w2])
                score = compute_score(w)
                if score > best_score:
                    best_score = score
                    best_weights = w.copy()

        # Refine with Nelder-Mead from best grid point
        try:
            result = minimize(
                lambda w: -compute_score(w),
                best_weights,
                method="Nelder-Mead",
                options={"maxiter": 300},
            )
            w = np.abs(result.x)
            refined_score = compute_score(w)
            if refined_score > best_score:
                best_weights = w / w.sum()
            else:
                best_weights = best_weights / best_weights.sum()
        except Exception:
            best_weights = best_weights / best_weights.sum()

        self.weights = best_weights

    def _weighted_probs(
        self, history: list[list[int]], weights: np.ndarray | None = None
    ) -> np.ndarray:
        """Get weighted probability from all available models."""
        if weights is None:
            weights = self.weights

        probs = np.zeros(self.max_num, dtype=np.float64)
        total_weight = 0

        models = [
            ("frequency", self.frequency_model),
            ("lstm", self.lstm_model),
            ("dqn", self.dqn_model),
        ]

        for i, (name, model) in enumerate(models):
            if not self._models_trained.get(name, False):
                continue
            try:
                p = model.get_probabilities(history)
                probs += weights[i] * p
                total_weight += weights[i]
            except Exception:
                continue

        if total_weight > 0:
            probs = probs / total_weight

        return probs.astype(np.float32)

    def get_probabilities(self, history: list[list[int]]) -> np.ndarray:
        return self._weighted_probs(history)

    def predict(self, history: list[list[int]], n_sets: int = 1) -> list[list[int]]:
        probs = self.get_probabilities(history)
        results = []

        for _ in range(n_sets):
            p = probs.copy()
            selected = []
            for _ in range(self.pick_count):
                total = p.sum()
                if total > 0:
                    normalized = p / total
                else:
                    normalized = np.ones(self.max_num) / self.max_num
                idx = np.random.choice(self.max_num, p=normalized)
                selected.append(idx + 1)
                p[idx] = 0

            results.append(sorted(selected))

        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save ensemble metadata + sub-models
        import json
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps({
            "max_num": self.max_num,
            "pick_count": self.pick_count,
            "weights": self.weights.tolist(),
            "models_trained": self._models_trained,
        }))

        base = path.parent / path.stem
        if self._models_trained.get("frequency"):
            self.frequency_model.save(base.with_name(f"{base.name}_freq.json"))
        if self._models_trained.get("lstm"):
            self.lstm_model.save(base.with_name(f"{base.name}_lstm.pt"))
        if self._models_trained.get("dqn"):
            self.dqn_model.save(base.with_name(f"{base.name}_dqn.pt"))

    def load(self, path: Path) -> None:
        import json
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())

        self.max_num = meta["max_num"]
        self.pick_count = meta["pick_count"]
        self.weights = np.array(meta["weights"])
        self._models_trained = meta["models_trained"]

        base = path.parent / path.stem
        if self._models_trained.get("frequency"):
            self.frequency_model = FrequencyModel(self.max_num, self.pick_count)
            self.frequency_model.load(base.with_name(f"{base.name}_freq.json"))
        if self._models_trained.get("lstm"):
            self.lstm_model = LSTMModel(self.max_num, self.pick_count, self.device)
            self.lstm_model.load(base.with_name(f"{base.name}_lstm.pt"))
        if self._models_trained.get("dqn"):
            self.dqn_model = DQNModel(self.max_num, self.pick_count, self.device)
            self.dqn_model.load(base.with_name(f"{base.name}_dqn.pt"))
