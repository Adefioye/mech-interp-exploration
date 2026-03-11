from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch as t

from autoencoder.base_extractor import (
    BaseFeatureExtractor,
    FeatureExtractionResult,
    validate_factorization_shapes,
)
from autoencoder.common import set_seed


def positive_part(x: t.Tensor) -> t.Tensor:
    return 0.5 * (x.abs() + x)


def negative_part(x: t.Tensor) -> t.Tensor:
    return 0.5 * (x.abs() - x)


@t.no_grad()
def fix_scale_inplace(z: t.Tensor, y: t.Tensor, eps: float = 1e-8) -> None:
    col_norms = y.norm(dim=0, keepdim=True).clamp_min(eps)
    y.div_(col_norms)
    z.mul_(col_norms.squeeze(0))


@t.no_grad()
def init_svd(a: t.Tensor, n_components: int, eps: float = 1e-8) -> tuple[t.Tensor, t.Tensor]:
    d_hidden, n = a.shape
    rank = min(d_hidden, n, n_components)

    u, s, vh = t.linalg.svd(a, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]

    sroot = s.sqrt()
    z = u * sroot.unsqueeze(0)
    y = (sroot.unsqueeze(1) * vh).T.clamp_min(eps)

    if rank < n_components:
        z_pad = t.randn((d_hidden, n_components - rank), device=a.device, dtype=a.dtype)
        y_pad = t.rand((n, n_components - rank), device=a.device, dtype=a.dtype).clamp_min(eps)
        z = t.cat([z, z_pad], dim=1)
        y = t.cat([y, y_pad], dim=1)

    return z, y


@t.no_grad()
def init_knn(
    a: t.Tensor,
    n_components: int,
    n_iter: int = 15,
    eps: float = 1e-8,
    chunk_size: int = 10_000,
) -> tuple[t.Tensor, t.Tensor]:
    d_hidden, n = a.shape
    x = a.T
    device = a.device

    if n_components <= n:
        perm = t.randperm(n, device=device)
        centres = x[perm[:n_components]].clone()
    else:
        rand_idx = t.randint(0, n, (n_components,), device=device)
        centres = x[rand_idx].clone()

    labels = t.empty(n, dtype=t.long, device=device)

    for _ in range(n_iter):
        c2 = (centres * centres).sum(dim=1).unsqueeze(0)

        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            block = x[start:end]
            x2 = (block * block).sum(dim=1, keepdim=True)
            dot = block @ centres.T
            dist2 = x2 + c2 - 2.0 * dot
            labels[start:end] = dist2.argmin(dim=1)

        counts = t.bincount(labels, minlength=n_components).unsqueeze(1)
        sums = t.zeros((n_components, d_hidden), device=device, dtype=a.dtype)
        sums.scatter_add_(0, labels.view(-1, 1).expand(-1, d_hidden), x)
        centres = sums / counts.clamp_min(1)

        empty = (counts.squeeze(1) == 0).nonzero(as_tuple=False).view(-1)
        if empty.numel() > 0:
            rand_idx = t.randint(0, n, (empty.numel(),), device=device)
            centres[empty] = x[rand_idx]

    z = centres.T
    y = t.zeros((n, n_components), device=device, dtype=a.dtype)
    y[t.arange(n, device=device), labels] = 1.0
    y.clamp_min_(eps)
    return z, y


@dataclass(frozen=True)
class SemiNMFExtractorConfig:
    n_components: int
    max_iter: int = 500
    tol: float = 1e-6
    patience: int = 30
    closed_form_eqn_reg: float = 1e-4
    sparsity_reg: float = 0.1
    verbose_every: int = 25
    seed: int = 42
    init: str = "random"  # random | svd | knn
    knn_iters: int = 20
    knn_chunk_size: int = 5_000
    dtype: t.dtype = t.float32
    device: t.device = t.device("mps")


class SemiNMFExtractor(BaseFeatureExtractor):
    def __init__(self, config: SemiNMFExtractorConfig):
        self.config = config

    @property
    def method_name(self) -> str:
        return "semi_nmf"

    def get_config(self) -> dict[str, object]:
        cfg = asdict(self.config)
        cfg["dtype"] = str(self.config.dtype)
        cfg["device"] = str(self.config.device)
        return cfg

    @t.no_grad()
    def fit(self, activations: np.ndarray) -> FeatureExtractionResult:
        cfg = self.config
        set_seed(cfg.seed)

        x = np.asarray(activations, dtype=np.float64)
        a = t.tensor(x, dtype=cfg.dtype, device=cfg.device).T  # (d_hidden, n)

        d_hidden, n = a.shape
        k = cfg.n_components
        i_k = t.eye(k, device=cfg.device, dtype=cfg.dtype)

        if cfg.init == "random":
            y = t.rand((n, k), device=cfg.device, dtype=cfg.dtype).clamp_min(1e-8)
            z = t.randn((d_hidden, k), device=cfg.device, dtype=cfg.dtype)
        elif cfg.init == "svd":
            z, y = init_svd(a, n_components=k, eps=1e-8)
        elif cfg.init == "knn":
            z, y = init_knn(
                a,
                n_components=k,
                n_iter=cfg.knn_iters,
                eps=1e-8,
                chunk_size=cfg.knn_chunk_size,
            )
        else:
            raise ValueError(f"Unsupported init '{cfg.init}'.")

        best_loss = float("inf")
        best_iter = -1
        best_z: t.Tensor | None = None
        best_y: t.Tensor | None = None
        no_improve = 0

        for it in range(cfg.max_iter):
            yty = y.T @ y
            z = t.linalg.solve(yty + cfg.closed_form_eqn_reg * i_k, (a @ y).T).T
            fix_scale_inplace(z, y)

            p = a.T @ z
            q = z.T @ z
            p_plus, p_minus = positive_part(p), negative_part(p)
            q_plus, q_minus = positive_part(q), negative_part(q)

            numer = p_plus + (y @ q_minus)
            denom = p_minus + (y @ q_plus) + cfg.sparsity_reg
            y = y * t.sqrt(numer / (denom + 1e-8))
            y = y.clamp_min(1e-8)

            a_hat = z @ y.T
            loss = t.norm(a - a_hat, p="fro").pow(2).item()

            if loss < (best_loss - cfg.tol):
                best_loss = loss
                best_iter = it
                best_z = z.detach().clone()
                best_y = y.detach().clone()
                no_improve = 0
            else:
                no_improve += 1

            if cfg.verbose_every > 0 and (it % cfg.verbose_every == 0 or no_improve == 1):
                print(
                    f"[SemiNMF] iter={it:4d} loss={loss:.6e} "
                    f"best={best_loss:.6e} no_improve={no_improve}"
                )

            if no_improve >= cfg.patience:
                print(
                    f"[SemiNMF] early stop at iter={it} "
                    f"(best_iter={best_iter}, best_loss={best_loss:.6e})"
                )
                break

        assert best_z is not None and best_y is not None
        learned_features = best_z.detach().cpu().numpy().astype(np.float64, copy=False)
        # Return coefficients as (n_components, num_samples).
        coefficients = best_y.detach().cpu().numpy().T.astype(np.float64, copy=False)
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
            reconstruction_loss=best_loss,
            metadata={
                "best_iter": best_iter,
                "n_components": cfg.n_components,
                "n_learned_features": int(learned_features.shape[1]),
            },
        )
