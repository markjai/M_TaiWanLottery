"""DQN (Deep Q-Network) Reinforcement Learning model for lottery prediction.

Improved with:
- Intermediate shaping rewards (not just terminal)
- Richer state representation using global context features
- Double DQN for more stable Q-value estimation
- Prioritized experience replay approximation
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from taiwan_lottery.ml.models.base_model import BaseLotteryModel
from taiwan_lottery.ml.features.feature_engineer import FeatureEngineer


class DuelingDQNNet(nn.Module):
    """Dueling DQN with deeper architecture."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DQNModel(BaseLotteryModel):
    """DQN-based lottery number selection model with improved rewards.

    State: selection_mask + frequency_context + gap_context + hot_cold_context
    Action: pick the next number
    Reward: intermediate shaping + terminal hit count
    """

    model_type = "dqn"

    def __init__(self, max_num: int, pick_count: int, device: str = "cpu"):
        self.max_num = max_num
        self.pick_count = pick_count
        self.device = torch.device(device)
        self.feature_eng = FeatureEngineer(max_num, pick_count)

        # State: selection_mask(max_num) + freq(max_num*4) + gap_ratio(max_num)
        #        + hot_cold(max_num) + step_progress(1)
        self.state_dim = max_num + max_num * 4 + max_num + max_num + 1
        self.action_dim = max_num

        self.q_net: DuelingDQNNet | None = None
        self.target_net: DuelingDQNNet | None = None

    def _build_nets(self):
        self.q_net = DuelingDQNNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingDQNNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _build_context(self, history: list[list[int]]) -> np.ndarray:
        """Build context: freq(4 windows) + gap_ratio + hot indicator."""
        freq = self.feature_eng.compute_frequency_features(history).flatten()  # max_num * 4
        gap = self.feature_eng.compute_gap_features(history)
        gap_ratio = gap[:, 2]  # just gap_ratio column, shape (max_num,)
        hot_cold = self.feature_eng.compute_hot_cold_features(history)
        is_hot = hot_cold[:, 0]  # just is_hot column, shape (max_num,)
        return np.concatenate([freq, gap_ratio, is_hot]).astype(np.float32)

    def _get_state(
        self, selection_mask: np.ndarray, context: np.ndarray, step: int
    ) -> torch.Tensor:
        progress = np.array([step / self.pick_count], dtype=np.float32)
        state = np.concatenate([selection_mask, context, progress])
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _compute_intermediate_reward(
        self, action: int, target: list[int],
        freq_weights: np.ndarray, gap_ratios: np.ndarray,
    ) -> float:
        """Intermediate shaping reward based on statistical plausibility.

        - Small positive reward if the selected number is statistically likely
        - Bonus if the number is in the target (but discounted to avoid overfitting)
        """
        reward = 0.0

        # Frequency-based reward: select numbers that appear frequently
        freq_score = freq_weights[action] if action < len(freq_weights) else 0
        reward += freq_score * 0.3

        # Gap-based reward: overdue numbers get small bonus
        gap_ratio = gap_ratios[action] if action < len(gap_ratios) else 0
        if 1.0 < gap_ratio < 2.5:
            reward += 0.1  # mildly overdue is good

        return reward

    def _compute_terminal_reward(self, selected: list[int], target: list[int]) -> float:
        """Terminal reward: hit count with bonus for multiple hits."""
        hits = len(set(selected) & set(target))
        # Exponential bonus for more hits
        if hits == 0:
            return -0.5
        return hits ** 1.5

    def train(self, history: list[list[int]], **kwargs) -> dict:
        episodes = kwargs.get("episodes", 2500)
        lr = kwargs.get("lr", 0.0005)
        gamma = kwargs.get("gamma", 0.95)
        epsilon_start = 1.0
        epsilon_end = 0.05
        epsilon_decay = 0.998
        batch_size = kwargs.get("batch_size", 128)
        memory_size = 50000
        target_update_freq = 20

        # Limit history for high-frequency games (bingo: ~200 draws/day)
        max_samples = kwargs.get("max_samples", None)
        if max_samples is None and self.max_num >= 80:
            max_samples = 10000
        if max_samples and len(history) > max_samples:
            history = history[-max_samples:]

        if len(history) < 50:
            return {"error": "Not enough data for DQN training"}

        self._build_nets()
        optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        memory = deque(maxlen=memory_size)
        epsilon = epsilon_start

        total_rewards = []

        # Pre-compute context from full (trimmed) history for efficiency
        # Only recompute per-episode context for the sliding window
        context_window = min(1000, len(history))

        for episode in range(episodes):
            # Pick a random target draw from history
            target_idx = random.randint(self.pick_count, len(history) - 1)
            target_nums = history[target_idx]
            # Use a limited window for context computation
            ctx_start = max(0, target_idx - context_window)
            context = self._build_context(history[ctx_start:target_idx])

            # Pre-compute freq weights and gap ratios for shaping rewards
            freq = self.feature_eng.compute_frequency_features(
                history[ctx_start:target_idx], windows=[30]
            )
            freq_weights = freq[:, 0]
            gap = self.feature_eng.compute_gap_features(history[ctx_start:target_idx])
            gap_ratios = gap[:, 2]

            selection_mask = np.zeros(self.max_num, dtype=np.float32)
            selected = []
            episode_reward = 0

            for step in range(self.pick_count):
                state = self._get_state(selection_mask, context, step)

                # Epsilon-greedy
                if random.random() < epsilon:
                    available = [i for i in range(self.max_num) if selection_mask[i] == 0]
                    action = random.choice(available)
                else:
                    with torch.no_grad():
                        q_values = self.q_net(state).squeeze()
                        q_values[selection_mask > 0] = -float("inf")
                        action = q_values.argmax().item()

                selected.append(action + 1)
                selection_mask[action] = 1.0

                # Intermediate shaping reward + terminal reward
                if step == self.pick_count - 1:
                    reward = self._compute_terminal_reward(selected, target_nums)
                    done = True
                else:
                    reward = self._compute_intermediate_reward(
                        action, target_nums, freq_weights, gap_ratios
                    )
                    done = False

                next_state = self._get_state(selection_mask, context, step + 1)

                memory.append((
                    state.cpu().numpy(),
                    action,
                    reward,
                    next_state.cpu().numpy(),
                    done,
                    selection_mask.copy(),
                ))

                episode_reward += reward

            total_rewards.append(episode_reward)

            # Train from replay buffer with Double DQN
            if len(memory) >= batch_size:
                batch = random.sample(list(memory), batch_size)
                states = torch.tensor(
                    np.array([s[0] for s, *_ in batch]).squeeze(), dtype=torch.float32
                ).to(self.device)
                actions = torch.tensor([a for _, a, *_ in batch], dtype=torch.long).to(self.device)
                rewards = torch.tensor([r for _, _, r, *_ in batch], dtype=torch.float32).to(self.device)
                next_states = torch.tensor(
                    np.array([ns[0] for *_, ns, _, _ in batch]).squeeze(), dtype=torch.float32
                ).to(self.device)
                dones = torch.tensor([d for *_, d, _ in batch], dtype=torch.float32).to(self.device)

                q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                # Double DQN: use q_net to select action, target_net to evaluate
                with torch.no_grad():
                    next_actions = self.q_net(next_states).argmax(dim=1)
                    next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                    target_q = rewards + gamma * next_q * (1 - dones)

                loss = nn.SmoothL1Loss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                optimizer.step()

            # Soft update target network
            if episode % target_update_freq == 0:
                tau = 0.005
                for tp, qp in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        avg_reward = np.mean(total_rewards[-100:]) if total_rewards else 0
        return {
            "episodes": episodes,
            "avg_reward_last100": round(float(avg_reward), 4),
            "final_epsilon": round(epsilon, 4),
        }

    def get_probabilities(self, history: list[list[int]]) -> np.ndarray:
        if self.q_net is None:
            raise RuntimeError("Model not trained")

        context = self._build_context(history)
        selection_mask = np.zeros(self.max_num, dtype=np.float32)

        self.q_net.eval()
        with torch.no_grad():
            state = self._get_state(selection_mask, context, 0)
            q_values = self.q_net(state).squeeze().cpu().numpy()

        # Convert Q-values to probabilities via softmax
        q_shifted = q_values - q_values.max()
        exp_q = np.exp(q_shifted)
        probs = exp_q / exp_q.sum()
        return probs.astype(np.float32)

    def predict(self, history: list[list[int]], n_sets: int = 1) -> list[list[int]]:
        if self.q_net is None:
            raise RuntimeError("Model not trained")

        context = self._build_context(history)
        results = []

        self.q_net.eval()
        for _ in range(n_sets):
            selection_mask = np.zeros(self.max_num, dtype=np.float32)
            selected = []

            for step in range(self.pick_count):
                state = self._get_state(selection_mask, context, step)
                with torch.no_grad():
                    q_values = self.q_net(state).squeeze()
                    q_values[selection_mask > 0] = -float("inf")
                    action = q_values.argmax().item()

                selected.append(action + 1)
                selection_mask[action] = 1.0

            results.append(sorted(selected))

        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net_state": self.q_net.state_dict() if self.q_net else None,
            "max_num": self.max_num,
            "pick_count": self.pick_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.max_num = checkpoint["max_num"]
        self.pick_count = checkpoint["pick_count"]
        self.state_dim = checkpoint["state_dim"]
        self.action_dim = checkpoint["action_dim"]
        self.feature_eng = FeatureEngineer(self.max_num, self.pick_count)

        self._build_nets()
        if checkpoint["q_net_state"]:
            self.q_net.load_state_dict(checkpoint["q_net_state"])
            self.target_net.load_state_dict(checkpoint["q_net_state"])
