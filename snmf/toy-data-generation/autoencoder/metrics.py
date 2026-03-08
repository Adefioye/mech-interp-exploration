from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def mean_max_cosine_similarity(
    learned_features: FloatArray,
    ground_truth_features: FloatArray,
) -> float:
    """Mean over ground-truth atoms of their best cosine with learned atoms."""
    gt_norm = np.linalg.norm(ground_truth_features, axis=0)[:, None]
    lf_norm = np.linalg.norm(learned_features, axis=0)[None, :]
    denom = np.maximum(gt_norm * lf_norm, 1e-12)
    cos_sims = (ground_truth_features.T @ learned_features) / denom
    largest_cosine_similarity = np.max(cos_sims, axis=1)
    return float(np.mean(largest_cosine_similarity))
