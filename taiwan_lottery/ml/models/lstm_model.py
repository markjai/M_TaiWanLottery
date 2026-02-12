"""Transformer Encoder model for lottery number prediction.

Replaces the original BiLSTM+Attention with a Transformer Encoder architecture
that better captures long-range dependencies in draw sequences. Uses Top-K
constrained sampling for valid probability output.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from taiwan_lottery.ml.models.base_model import BaseLotteryModel
from taiwan_lottery.ml.features.feature_engineer import FeatureEngineer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LotteryTransformerNet(nn.Module):
    """Transformer Encoder for lottery prediction.

    Architecture:
        Input projection → Positional Encoding → Transformer Encoder (4 layers)
        → Global context pooling → FC head → per-number logits
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input features to model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-scale pooling: combine CLS-like (last token), mean, max
        self.pool_proj = nn.Linear(d_model * 3, d_model)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)             # (batch, seq_len, d_model)
        x = self.pos_encoder(x)            # (batch, seq_len, d_model)
        x = self.transformer(x)            # (batch, seq_len, d_model)

        # Multi-scale pooling
        last_token = x[:, -1, :]                    # (batch, d_model)
        mean_pool = x.mean(dim=1)                   # (batch, d_model)
        max_pool = x.max(dim=1).values              # (batch, d_model)
        pooled = torch.cat([last_token, mean_pool, max_pool], dim=-1)
        pooled = self.pool_proj(pooled)             # (batch, d_model)

        return self.head(pooled)                    # (batch, output_dim) — raw logits


class LSTMModel(BaseLotteryModel):
    """Transformer-based lottery prediction model.

    Kept as LSTMModel class name for backward compatibility with
    model registry and training pipeline.
    """

    model_type = "lstm"

    def __init__(self, max_num: int, pick_count: int, device: str = "cpu"):
        self.max_num = max_num
        self.pick_count = pick_count
        self.device = torch.device(device)
        self.seq_len = 30  # increased from 20 for more context
        self.feature_eng = FeatureEngineer(max_num, pick_count)
        # Enhanced: multi-hot(max_num) + context(max_num*4+15) + aggregate(11)
        # = max_num*5 + 26  (e.g. 80*5+26=426 for bingo)
        self.input_dim = max_num * 5 + 26
        self.net: LotteryTransformerNet | None = None

    def _build_net(self) -> LotteryTransformerNet:
        net = LotteryTransformerNet(
            input_dim=self.input_dim,
            output_dim=self.max_num,
            d_model=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.2,
        ).to(self.device)
        return net

    def _prepare_data(
        self, history: list[list[int]], max_samples: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data: sliding window sequences → multi-hot targets.

        Uses enhanced features (multi-hot + precomputed context + aggregate).
        For high-frequency games (bingo), use max_samples to limit data size.
        """
        from loguru import logger

        # Limit history to most recent draws if max_samples is set
        if max_samples and len(history) > max_samples + self.seq_len:
            history = history[-(max_samples + self.seq_len):]

        # Precompute context features for entire history at once
        logger.info("Precomputing context features for {} draws...", len(history))
        context_all = self.feature_eng.precompute_context_features(history, context_window=100)

        X_list, y_list = [], []

        for i in range(self.seq_len, len(history)):
            # Build enhanced sequence for positions [i-seq_len, i)
            seq_features = []
            for t in range(i - self.seq_len, i):
                multi_hot = self.feature_eng.build_multi_hot(history[t])     # (max_num,)
                context = context_all[t]                                     # (context_dim,)
                agg = self.feature_eng.compute_aggregate_features(history[t])  # (11,)
                seq_features.append(np.concatenate([multi_hot, context, agg]))

            X_list.append(np.array(seq_features, dtype=np.float32))
            y_list.append(self.feature_eng.build_multi_hot(history[i]))

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32)
        logger.info("Prepared {} training samples with input_dim={}", len(X), X.shape[-1])
        return X, y

    def train(self, history: list[list[int]], **kwargs) -> dict:
        epochs = kwargs.get("epochs", 120)
        lr = kwargs.get("lr", 0.0003)
        batch_size = kwargs.get("batch_size", 64)
        weight_decay = kwargs.get("weight_decay", 1e-4)
        warmup_epochs = kwargs.get("warmup_epochs", 5)

        # Limit samples for high-frequency games (bingo: ~200 draws/day)
        max_samples = kwargs.get("max_samples", None)
        if max_samples is None and self.max_num >= 80:
            max_samples = 15000  # ~75 days of bingo data

        if len(history) < self.seq_len + 10:
            return {"error": "Not enough data for training"}

        self.net = self._build_net()
        self.net.train()

        X, y = self._prepare_data(history, max_samples=max_samples)

        # Time-series split: 80% train, 10% val, 10% test
        n = len(X)
        train_end = int(n * 0.80)
        val_end = int(n * 0.90)
        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

        # Move to device
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        X_test, y_test = X_test.to(self.device), y_test.to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2
        )

        # Focal-like BCE loss: emphasize harder-to-predict numbers
        pos_weight = torch.tensor(
            [(self.max_num - self.pick_count) / self.pick_count]
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Warmup: linearly increase lr
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * warmup_factor

            # Mini-batch training
            self.net.train()
            indices = torch.randperm(len(X_train))
            total_loss = 0
            n_batches = 0

            for start in range(0, len(X_train), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_x = X_train[batch_idx]
                batch_y = y_train[batch_idx]

                optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = criterion(logits, batch_y)

                # Label smoothing
                smooth_target = batch_y * 0.9 + 0.1 / self.max_num
                loss += 0.1 * criterion(logits, smooth_target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if epoch >= warmup_epochs:
                scheduler.step()

            train_losses.append(total_loss / max(n_batches, 1))

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(X_val)
                val_loss = criterion(val_logits, y_val).item()

                # Compute hit rate on validation set
                val_probs = torch.sigmoid(val_logits)
                val_hits = 0
                for i in range(len(X_val)):
                    top_k = val_probs[i].topk(self.pick_count).indices
                    predicted = set((top_k + 1).cpu().numpy())
                    actual = set(
                        j + 1 for j in range(self.max_num) if y_val[i, j] > 0.5
                    )
                    val_hits += len(predicted & actual)
                avg_val_hits = val_hits / max(len(X_val), 1)

            val_losses.append(val_loss)

            # Early stopping on val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

        if best_state:
            self.net.load_state_dict(best_state)

        # Evaluate on test set
        self.net.eval()
        test_hits = 0
        with torch.no_grad():
            test_logits = self.net(X_test)
            test_probs = torch.sigmoid(test_logits)
            for i in range(len(X_test)):
                top_k = test_probs[i].topk(self.pick_count).indices
                predicted = set((top_k + 1).cpu().numpy())
                actual = set(
                    j + 1 for j in range(self.max_num) if y_test[i, j] > 0.5
                )
                test_hits += len(predicted & actual)
        avg_test_hits = test_hits / max(len(X_test), 1)

        return {
            "architecture": "transformer_encoder",
            "epochs_trained": epoch + 1,
            "best_val_loss": round(best_val_loss, 6),
            "avg_val_hits": round(avg_val_hits, 4),
            "avg_test_hits": round(avg_test_hits, 4),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        }

    def _get_inference_features(self, history: list[list[int]]) -> torch.Tensor:
        """Get input features for inference, adapting to model's input_dim."""
        if self.input_dim == self.max_num + 11:
            # Legacy model: multi-hot + aggregate only
            seq = self.feature_eng.build_sequence_features(history, seq_len=self.seq_len)
        else:
            # Enhanced model: multi-hot + context + aggregate
            seq = self.feature_eng.build_enhanced_sequence_features(history, seq_len=self.seq_len)
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

    def get_probabilities(self, history: list[list[int]]) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model not trained")

        self.net.eval()
        x = self._get_inference_features(history)

        with torch.no_grad():
            logits = self.net(x).cpu()
            probs = torch.sigmoid(logits).numpy()[0]

        # Normalize to valid probability distribution
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs

    def predict(self, history: list[list[int]], n_sets: int = 1) -> list[list[int]]:
        """Generate predictions using Top-K constrained sampling.

        Uses temperature-scaled softmax over logits, then samples
        without replacement for diversity across n_sets.
        """
        if self.net is None:
            raise RuntimeError("Model not trained")

        self.net.eval()
        x = self._get_inference_features(history)

        with torch.no_grad():
            logits = self.net(x).cpu().numpy()[0]

        results = []
        for _ in range(n_sets):
            # Temperature-scaled sampling (temperature > 1 = more diversity)
            temperature = 1.2
            scaled = logits / temperature
            # Softmax for proper probability distribution
            exp_scaled = np.exp(scaled - scaled.max())
            probs = exp_scaled / exp_scaled.sum()

            # Sample without replacement
            selected = []
            p = probs.copy()
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
        torch.save({
            "model_state": self.net.state_dict() if self.net else None,
            "max_num": self.max_num,
            "pick_count": self.pick_count,
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "architecture": "transformer_encoder",
        }, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.max_num = checkpoint["max_num"]
        self.pick_count = checkpoint["pick_count"]
        self.input_dim = checkpoint["input_dim"]
        self.seq_len = checkpoint.get("seq_len", 30)
        self.feature_eng = FeatureEngineer(self.max_num, self.pick_count)

        self.net = self._build_net()
        if checkpoint["model_state"]:
            self.net.load_state_dict(checkpoint["model_state"])
