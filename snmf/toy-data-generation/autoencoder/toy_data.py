from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from tqdm import tqdm

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ToyDataConfig:
    num_samples: int = 100_000
    feature_dim: int = 256
    num_ground_truth_features: int = 512
    num_active_features: float = 5.0
    decay_rate: float = 0.99
    random_seed: int = 42


def generate_toy_data(cfg: ToyDataConfig) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate toy dataset with sparse nonnegative coefficients."""
    assert cfg.num_samples > 0, f"num_samples must be positive, got {cfg.num_samples}."
    assert cfg.feature_dim > 0, f"feature_dim must be positive, got {cfg.feature_dim}."
    assert (
        cfg.num_ground_truth_features > 0
    ), f"num_ground_truth_features must be positive, got {cfg.num_ground_truth_features}."
    assert (
        cfg.num_active_features >= 0
    ), f"num_active_features must be nonnegative, got {cfg.num_active_features}."
    assert cfg.decay_rate > 0, f"decay_rate must be positive, got {cfg.decay_rate}."

    rng = np.random.default_rng(cfg.random_seed)
    g = cfg.num_ground_truth_features

    ground_truth_features: FloatArray = rng.standard_normal(size=(cfg.feature_dim, g))
    col_norms: FloatArray = np.linalg.norm(ground_truth_features, axis=0)
    col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    ground_truth_features = ground_truth_features / col_norms

    cov_seed: FloatArray = rng.standard_normal(size=(g, g))
    covariance: FloatArray = cov_seed @ cov_seed.T
    mean = np.zeros(g, dtype=np.float64)
    gaussian_samples: FloatArray = rng.multivariate_normal(mean, covariance, size=cfg.num_samples)
    correlated_feature_probs: FloatArray = norm.cdf(gaussian_samples)

    sparse_coefficients: FloatArray = np.zeros((cfg.num_samples, g), dtype=np.float64)
    dataset: FloatArray = np.zeros((cfg.num_samples, cfg.feature_dim), dtype=np.float64)
    feature_indices: FloatArray = np.arange(g, dtype=np.float64)

    for i in tqdm(range(cfg.num_samples), desc="Generating toy data"):
        decayed_feature_probs: FloatArray = np.power(
            correlated_feature_probs[i],
            feature_indices * cfg.decay_rate,
        )
        mean_prob = float(np.mean(decayed_feature_probs))
        if mean_prob <= 0.0:
            raise ValueError("Mean decayed probability became non-positive.")

        probs: FloatArray = decayed_feature_probs / mean_prob
        probs = cfg.num_active_features * probs / g
        probs = np.clip(probs, 0.0, 1.0)

        active_mask: FloatArray = rng.binomial(1, probs).astype(np.float64)
        amplitudes: FloatArray = rng.uniform(0.0, 1.0, g)
        activations: FloatArray = active_mask * amplitudes

        sparse_coefficients[i] = activations
        dataset[i] = ground_truth_features @ activations

    return dataset, ground_truth_features, sparse_coefficients
