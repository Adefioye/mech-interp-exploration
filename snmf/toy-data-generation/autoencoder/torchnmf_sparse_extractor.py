from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass

import numpy as np
import torch as t

from autoencoder.base_extractor import BaseFeatureExtractor, FeatureExtractionResult
from autoencoder.common import resolve_device, resolve_dtype, set_seed


@dataclass(frozen=True)
class TorchNMFExtractorConfig:
    n_components: int
    max_iter: int = 500
    beta: float = 2.0
    l1_strength: float = 0.0
    seed: int = 42
    dtype: str = "float32"
    device: str = "cpu"
    shift_to_nonnegative: bool = True
    verbose: bool = False


class TorchNMFSparseExtractor(BaseFeatureExtractor):
    def __init__(self, config: TorchNMFExtractorConfig):
        self.config = config

    @property
    def method_name(self) -> str:
        return "torchnmf_sparse_nmf"

    def get_config(self) -> dict[str, object]:
        return asdict(self.config)

    @staticmethod
    def _to_numpy_2d(x: object) -> np.ndarray | None:
        if isinstance(x, t.Tensor) and x.ndim == 2:
            return x.detach().cpu().numpy().astype(np.float64, copy=False)
        return None

    @staticmethod
    def _extract_factors(model: object) -> tuple[np.ndarray, np.ndarray]:
        cand_w = TorchNMFSparseExtractor._to_numpy_2d(getattr(model, "W", None))
        cand_h = TorchNMFSparseExtractor._to_numpy_2d(getattr(model, "H", None))
        if cand_w is None or cand_h is None:
            raise RuntimeError("Could not extract W/H from torchnmf model after fit().")
        return cand_w, cand_h

    @staticmethod
    def _align_factors(
        x_fit: np.ndarray,
        w: np.ndarray,
        h: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return coefficients (N,K), basis (K,D), and reconstruction SSE."""
        n, d = x_fit.shape
        candidates: list[tuple[np.ndarray, np.ndarray]] = []

        # candidate: W as coefficients, H as basis
        if w.shape[0] == n:
            if h.shape[0] == w.shape[1] and h.shape[1] == d:
                candidates.append((w, h))
            if h.shape[1] == w.shape[1] and h.shape[0] == d:
                candidates.append((w, h.T))

        # candidate: H as coefficients, W as basis
        if h.shape[0] == n:
            if w.shape[0] == h.shape[1] and w.shape[1] == d:
                candidates.append((h, w))
            if w.shape[1] == h.shape[1] and w.shape[0] == d:
                candidates.append((h, w.T))

        if not candidates:
            raise RuntimeError(
                f"Unable to align torchnmf factors with data shape {x_fit.shape}, "
                f"W shape {w.shape}, H shape {h.shape}."
            )

        best_coeff = candidates[0][0]
        best_basis = candidates[0][1]
        best_sse = float(np.linalg.norm(x_fit - best_coeff @ best_basis, ord="fro") ** 2)

        for coeff, basis in candidates[1:]:
            sse = float(np.linalg.norm(x_fit - coeff @ basis, ord="fro") ** 2)
            if sse < best_sse:
                best_coeff = coeff
                best_basis = basis
                best_sse = sse

        return best_coeff, best_basis, best_sse

    def fit(self, activations: np.ndarray) -> FeatureExtractionResult:
        cfg = self.config
        set_seed(cfg.seed)

        try:
            from torchnmf.nmf import NMF
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torchnmf is required for torchnmf_sparse_nmf method. "
                "Install with: pip install torchnmf"
            ) from exc

        x = np.asarray(activations, dtype=np.float64)
        shift_value = 0.0
        x_fit = x
        if cfg.shift_to_nonnegative:
            min_val = float(np.min(x))
            if min_val < 0.0:
                shift_value = -min_val
                x_fit = x + shift_value

        device = resolve_device(cfg.device)
        dtype = resolve_dtype(cfg.dtype)
        x_t = t.tensor(x_fit, dtype=dtype, device=device)

        # Constructor compatibility across torchnmf versions.
        try:
            model = NMF(x_t.shape, rank=cfg.n_components)
        except TypeError:
            model = NMF(x_t.shape, cfg.n_components)

        if not hasattr(model, "fit"):
            raise RuntimeError("torchnmf NMF model does not expose fit().")

        fit_sig = inspect.signature(model.fit)
        fit_kwargs: dict[str, object] = {}
        if "max_iter" in fit_sig.parameters:
            fit_kwargs["max_iter"] = cfg.max_iter
        if "beta" in fit_sig.parameters:
            fit_kwargs["beta"] = cfg.beta
        if "verbose" in fit_sig.parameters:
            fit_kwargs["verbose"] = cfg.verbose

        # Prefer sparse/L1 regularization if the library version supports it.
        if cfg.l1_strength > 0.0:
            if "alpha" in fit_sig.parameters:
                fit_kwargs["alpha"] = cfg.l1_strength
            if "l1_ratio" in fit_sig.parameters:
                fit_kwargs["l1_ratio"] = 1.0

        model.fit(x_t, **fit_kwargs)

        w, h = self._extract_factors(model)
        coefficients, basis_kd, sse = self._align_factors(x_fit=x_fit, w=w, h=h)
        learned_features = basis_kd.T

        return FeatureExtractionResult(
            learned_features=learned_features.astype(np.float64, copy=False),
            coefficients=coefficients.astype(np.float64, copy=False),
            reconstruction_loss=sse,
            metadata={
                "shift_value": shift_value,
            },
        )
