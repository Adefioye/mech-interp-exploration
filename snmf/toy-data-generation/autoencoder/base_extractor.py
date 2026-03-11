from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FeatureExtractionResult:
    """Common fit output for all feature extraction techniques."""

    # TODO: Verify the shapes later and make it consistent with nmf and sparse-nmf methods.
    # NOTE: n_components = number of extracted features, which is the "rank" of the factorization.
    learned_features: np.ndarray  # shape: (d_hidden, n_components)
    coefficients: np.ndarray      # shape: (n_components, num_samples)
    reconstruction_loss: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseFeatureExtractor(ABC):
    """Base interface for toy-data feature extractors."""

    @property
    @abstractmethod
    def method_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fit(self, activations: np.ndarray) -> FeatureExtractionResult:
        """Fit on activations with shape (num_samples, d_hidden)."""
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return serializable config for logging."""
        raise NotImplementedError
