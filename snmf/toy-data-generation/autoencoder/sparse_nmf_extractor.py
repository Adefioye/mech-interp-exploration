from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch as t

from autoencoder.base_extractor import (
    BaseFeatureExtractor,
    FeatureExtractionResult,
    validate_factorization_shapes,
)
from autoencoder.common import resolve_device, resolve_dtype, set_seed
from autoencoder.io_utils import count_negative_elements


@dataclass(frozen=True)
class SparseNMFConfig:
    n_components: int
    max_iter: int = 200
    beta: float = 2.0
    s_w: float | None = None
    s_h: float | None = None
    seed: int = 42
    dtype: str = "float32"
    device: str = "cuda"
    verbose: bool = False


class SparseNMFExtractor(BaseFeatureExtractor):
    def __init__(self, config: SparseNMFConfig):
        self.config = config

    @property
    def method_name(self) -> str:
        return "sparse_nmf"

    def get_config(self) -> dict[str, object]:
        return asdict(self.config)

    @staticmethod
    def _validate_sparsity_target(value: float | None, name: str) -> None:
        if value is None:
            return
        if not (0.0 < value < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got {value}.")

    def fit(self, activations: np.ndarray) -> FeatureExtractionResult:
        cfg = self.config
        set_seed(cfg.seed)

        try:
            from torchnmf.nmf import NMF
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torchnmf is required for sparse_nmf method. "
                "Install with: pip install torchnmf"
            ) from exc

        if isinstance(activations, t.Tensor):
            x = activations.detach().to(dtype=t.float32, device="cpu").numpy()
        else:
            x = np.asarray(activations, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D activations, got shape {x.shape}.")
        
        negative_count = count_negative_elements(x)

        if negative_count > 0:
            print(
                f"[{self.method_name}] Warning: Input contains {negative_count} negative elements. "
                "NMF requires non-negative inputs. Shifting input to be non-negative."
            )
            self.shift = -x.min()
            x += self.shift

        x_fit = x
        x_fit = t.tensor(x_fit, dtype=resolve_dtype(cfg.dtype))

        self._validate_sparsity_target(cfg.s_w, "s_w")
        self._validate_sparsity_target(cfg.s_h, "s_h")

        # torchnmf's sparse_fit has device compatibility issues, so we use CPU for fitting

        # Per torchnmf docs for NMF: V ≈ H W^T with W:(C,R), H:(N,R).
        try:
            model = NMF(x_fit.shape, rank=cfg.n_components)
        except TypeError:
            model = NMF(x_fit.shape, cfg.n_components)

        if not hasattr(model, "sparse_fit"):
            raise RuntimeError("torchnmf NMF model does not expose sparse_fit().")

        n_iter = model.sparse_fit(
            x_fit,
            beta=cfg.beta,
            max_iter=cfg.max_iter,
            verbose=cfg.verbose,
            sW=cfg.s_w,
            sH=cfg.s_h,
        )

        if model.W is None or model.H is None:
            raise RuntimeError("torchnmf returned empty factors (W/H).")

        with t.no_grad():
            x_hat = model()
            reconstruction_loss = t.norm(x_fit - x_hat, p="fro").pow(2).item()

        learned_features = model.W.detach().cpu().numpy().astype(np.float64, copy=False)
        # Return coefficients as (n_components, num_samples).
        coefficients = model.H.detach().cpu().numpy().T.astype(np.float64, copy=False)

        validate_factorization_shapes(
            activations=x,
            learned_features=learned_features,
            coefficients=coefficients,
            n_components=cfg.n_components,
            method_name=self.method_name,
        )

        return FeatureExtractionResult(
            learned_features=learned_features,
            coefficients=coefficients,
            reconstruction_loss=float(reconstruction_loss),
            metadata={
                "beta": cfg.beta,
                "n_iter": int(n_iter),
                "s_w": cfg.s_w,
                "s_h": cfg.s_h,
                "n_learned_features": int(learned_features.shape[1]),
            },
        )
