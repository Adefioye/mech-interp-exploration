from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FeatureExtractionResult:
    """Common fit output for all feature extraction techniques."""

    # NOTE: n_components = number of learned features (factorization rank).
    learned_features: np.ndarray  # shape: (d_hidden, n_components)
    coefficients: np.ndarray      # shape: (n_components, num_samples)
    reconstruction_loss: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_factorization_shapes(
    activations: np.ndarray,
    learned_features: np.ndarray,
    coefficients: np.ndarray,
    n_components: int,
    method_name: str,
) -> None:
    """Validate extractor output shape convention for all methods."""
    x = np.asarray(activations)
    if x.ndim != 2:
        raise ValueError(f"[{method_name}] activations must be 2D, got shape {x.shape}.")

    num_samples, d_hidden = x.shape
    expected_lf_shape = (d_hidden, n_components)
    expected_coeff_shape = (n_components, num_samples)

    if learned_features.shape != expected_lf_shape:
        raise ValueError(
            f"[{method_name}] learned_features shape mismatch: "
            f"expected {expected_lf_shape}, got {learned_features.shape}."
        )
    if coefficients.shape != expected_coeff_shape:
        raise ValueError(
            f"[{method_name}] coefficients shape mismatch: "
            f"expected {expected_coeff_shape}, got {coefficients.shape}."
        )


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
