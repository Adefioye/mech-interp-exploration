from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass

import numpy as np

from autoencoder.base_extractor import BaseFeatureExtractor, FeatureExtractionResult


@dataclass(frozen=True)
class SklearnNMFExtractorConfig:
    n_components: int
    max_iter: int = 500
    tol: float = 1e-4
    init: str = "nndsvda"
    solver: str = "cd"
    beta_loss: str = "frobenius"
    alpha_w: float = 0.0
    alpha_h: float = 0.0
    l1_ratio: float = 0.0
    random_state: int = 42
    shift_to_nonnegative: bool = True


class SklearnNMFExtractor(BaseFeatureExtractor):
    def __init__(self, config: SklearnNMFExtractorConfig):
        self.config = config

    @property
    def method_name(self) -> str:
        return "sklearn_nmf"

    def get_config(self) -> dict[str, object]:
        return asdict(self.config)

    def fit(self, activations: np.ndarray) -> FeatureExtractionResult:
        try:
            from sklearn.decomposition import NMF
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scikit-learn is required for sklearn_nmf method. "
                "Install with: pip install scikit-learn"
            ) from exc

        cfg = self.config
        x = np.asarray(activations, dtype=np.float64)

        shift_value = 0.0
        x_fit = x
        if cfg.shift_to_nonnegative:
            min_val = float(np.min(x))
            if min_val < 0.0:
                shift_value = -min_val
                x_fit = x + shift_value

        nmf_kwargs = {
            "n_components": cfg.n_components,
            "max_iter": cfg.max_iter,
            "tol": cfg.tol,
            "init": cfg.init,
            "solver": cfg.solver,
            "beta_loss": cfg.beta_loss,
            "l1_ratio": cfg.l1_ratio,
            "random_state": cfg.random_state,
        }

        sig = inspect.signature(NMF.__init__)
        if "alpha_W" in sig.parameters:
            nmf_kwargs["alpha_W"] = cfg.alpha_w
        if "alpha_H" in sig.parameters:
            nmf_kwargs["alpha_H"] = cfg.alpha_h
        if "alpha" in sig.parameters and "alpha_W" not in sig.parameters:
            nmf_kwargs["alpha"] = cfg.alpha_w

        model = NMF(**nmf_kwargs)
        coefficients = model.fit_transform(x_fit)  # W: (n_samples, n_components)
        components = model.components_             # H: (n_components, d_hidden)

        reconstruction = coefficients @ components
        reconstruction_loss = float(np.linalg.norm(x_fit - reconstruction, ord="fro") ** 2)

        learned_features = components.T  # (d_hidden, n_components)

        return FeatureExtractionResult(
            learned_features=learned_features.astype(np.float64, copy=False),
            coefficients=coefficients.astype(np.float64, copy=False),
            reconstruction_loss=reconstruction_loss,
            metadata={
                "shift_value": shift_value,
                "n_iter": getattr(model, "n_iter_", None),
                "reconstruction_err": float(getattr(model, "reconstruction_err_", np.nan)),
            },
        )
