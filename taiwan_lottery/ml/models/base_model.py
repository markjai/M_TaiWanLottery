"""Base model abstract class for lottery prediction models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseLotteryModel(ABC):
    """Abstract base for all lottery prediction models."""

    model_type: str = ""

    @abstractmethod
    def train(self, history: list[list[int]], **kwargs) -> dict:
        """Train the model on historical draw data.

        Args:
            history: List of sorted number lists, ordered chronologically.

        Returns:
            dict of training metrics
        """
        ...

    @abstractmethod
    def predict(self, history: list[list[int]], n_sets: int = 1) -> list[list[int]]:
        """Generate predicted number sets.

        Args:
            history: Historical draws for context
            n_sets: Number of prediction sets to generate

        Returns:
            List of predicted number lists
        """
        ...

    @abstractmethod
    def get_probabilities(self, history: list[list[int]]) -> np.ndarray:
        """Get probability distribution over all numbers.

        Returns:
            Array of shape (max_num,) with probability for each number
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        ...
