#!/usr/bin/env python3
"""Run toy-data Semi-NMF experiments with the fit() method only.

This script includes:
- Toy data generation
- SemiNMF model with fit() only (no fit_batched)
- Mean max cosine similarity scoring against ground-truth features
- Optional result appending to JSON/JSONL files
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch as t
import torch.nn as nn
from numpy.typing import NDArray
from scipy.stats import norm
from tqdm import tqdm

FloatArray = NDArray[np.float64]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def positive_part(x: t.Tensor) -> t.Tensor:
    return 0.5 * (x.abs() + x)


def negative_part(x: t.Tensor) -> t.Tensor:
    return 0.5 * (x.abs() - x)


@t.no_grad()
def fix_scale_inplace(z: t.Tensor, y: t.Tensor, eps: float = 1e-8) -> None:
    """Normalize columns of y and compensate in z so z @ y.T is unchanged."""
    col_norms = y.norm(dim=0, keepdim=True).clamp_min(eps)
    y.div_(col_norms)
    z.mul_(col_norms.squeeze(0))


@t.no_grad()
def init_svd(a: t.Tensor, k: int, eps: float = 1e-8) -> tuple[t.Tensor, t.Tensor]:
    """SVD-based initialization for Semi-NMF (single variant)."""
    d_hidden, n = a.shape
    rank = min(d_hidden, n, k)

    u, s, vh = t.linalg.svd(a, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]

    sroot = s.sqrt()
    z = u * sroot.unsqueeze(0)                      # (d_hidden, rank)
    y = (sroot.unsqueeze(1) * vh).T.clamp_min(eps)  # (n, rank), forced nonnegative

    # If k exceeds truncated SVD rank, pad remaining factors randomly.
    if rank < k:
        z_pad = t.randn((d_hidden, k - rank), device=a.device, dtype=a.dtype)
        y_pad = t.rand((n, k - rank), device=a.device, dtype=a.dtype).clamp_min(eps)
        z = t.cat([z, z_pad], dim=1)
        y = t.cat([y, y_pad], dim=1)

    return z, y


@t.no_grad()
def init_knn(
    a: t.Tensor,
    k: int,
    n_iter: int = 15,
    eps: float = 1e-8,
    chunk_size: int = 10_000,
) -> tuple[t.Tensor, t.Tensor]:
    """K-means/KNN-style initialization for Semi-NMF."""
    d_hidden, n = a.shape
    x = a.T  # (n, d_hidden)
    device = a.device

    if k <= n:
        perm = t.randperm(n, device=device)
        centres = x[perm[:k]].clone()
    else:
        rand_idx = t.randint(0, n, (k,), device=device)
        centres = x[rand_idx].clone()

    labels = t.empty(n, dtype=t.long, device=device)

    for _ in range(n_iter):
        c2 = (centres * centres).sum(dim=1).unsqueeze(0)  # (1, k)

        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            block = x[start:end]  # (b, d_hidden)
            x2 = (block * block).sum(dim=1, keepdim=True)  # (b, 1)
            dot = block @ centres.T  # (b, k)
            dist2 = x2 + c2 - 2.0 * dot
            labels[start:end] = dist2.argmin(dim=1)

        counts = t.bincount(labels, minlength=k).unsqueeze(1)  # (k, 1)
        sums = t.zeros((k, d_hidden), device=device, dtype=a.dtype)
        sums.scatter_add_(0, labels.view(-1, 1).expand(-1, d_hidden), x)

        centres = sums / counts.clamp_min(1)

        empty = (counts.squeeze(1) == 0).nonzero(as_tuple=False).view(-1)
        if empty.numel() > 0:
            rand_idx = t.randint(0, n, (empty.numel(),), device=device)
            centres[empty] = x[rand_idx]

    z = centres.T  # (d_hidden, k)
    y = t.zeros((n, k), device=device, dtype=a.dtype)
    y[t.arange(n, device=device), labels] = 1.0
    y.clamp_min_(eps)
    return z, y


@dataclass(frozen=True)
class ToyDataConfig:
    num_samples: int = 100_000
    feature_dim: int = 256
    num_ground_truth_features: int = 512
    num_active_features: float = 5.0
    decay_rate: float = 0.99
    random_seed: int = 42


def generate_toy_data(cfg: ToyDataConfig) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate toy data A = C @ F^T where C is sparse nonnegative."""
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

    # 1) Dense signed ground-truth feature dictionary (unit-norm columns).
    ground_truth_features: FloatArray = rng.standard_normal(size=(cfg.feature_dim, g))
    col_norms: FloatArray = np.linalg.norm(ground_truth_features, axis=0)
    col_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    ground_truth_features = ground_truth_features / col_norms

    # 2) Correlated latent probabilities via Gaussian copula-like sampling.
    cov_seed: FloatArray = rng.standard_normal(size=(g, g))
    covariance: FloatArray = cov_seed @ cov_seed.T
    mean = np.zeros(g, dtype=np.float64)
    gaussian_samples: FloatArray = rng.multivariate_normal(mean, covariance, size=cfg.num_samples)
    correlated_feature_probs: FloatArray = norm.cdf(gaussian_samples)

    # 3) Build sparse nonnegative coefficients and observed dataset.
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


@dataclass
class SemiNMFResult:
    z: t.Tensor
    y: t.Tensor
    loss_history: list[float]
    best_iter: int
    best_loss: float


@dataclass(frozen=True)
class SemiNMFConfig:
    k_scale: float = 2.0
    max_iter: int = 300
    tol: float = 1e-6
    patience: int = 30
    closed_form_eqn_reg: float = 1e-6
    sparsity_reg: float = 0.0
    verbose_every: int = 25
    seed: int = 42
    dtype: str = "float32"
    init: str = "random"
    knn_iters: int = 15
    knn_chunk_size: int = 10_000


class SemiNMF(nn.Module):
    """Semi-NMF: A ≈ Z @ Y.T with Y >= 0 and Z unconstrained."""

    def __init__(self, k: int, device: t.device, dtype: t.dtype = t.float32):
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        self.k = k
        self.device = device
        self.dtype = dtype

    @t.no_grad()
    def fit(self, a: t.Tensor, cfg: SemiNMFConfig) -> SemiNMFResult:
        """Fit using full-batch alternating updates."""
        set_seed(cfg.seed)

        a = a.to(device=self.device, dtype=self.dtype)
        d_hidden, n = a.shape
        k = self.k

        if cfg.init == "random":
            y = t.rand((n, k), device=self.device, dtype=self.dtype).clamp_min(1e-8)
            z = t.randn((d_hidden, k), device=self.device, dtype=self.dtype)
        elif cfg.init == "svd":
            z, y = init_svd(a, k=k, eps=1e-8)
        elif cfg.init == "knn":
            z, y = init_knn(
                a,
                k=k,
                n_iter=cfg.knn_iters,
                eps=1e-8,
                chunk_size=cfg.knn_chunk_size,
            )
        else:
            raise ValueError(f"Unsupported init '{cfg.init}'. Use random, svd, or knn.")

        best_loss = float("inf")
        best_iter = -1
        best_z: t.Tensor | None = None
        best_y: t.Tensor | None = None
        no_improve = 0
        history: list[float] = []

        i_k = t.eye(k, device=self.device, dtype=self.dtype)

        for it in range(cfg.max_iter):
            # 1) Closed-form Z update.
            yty = y.T @ y
            z = t.linalg.solve(
                yty + cfg.closed_form_eqn_reg * i_k,
                (a @ y).T,
            ).T

            # Keep factor scaling stable.
            fix_scale_inplace(z, y)

            # 2) Multiplicative Y update (nonnegative) + optional L1 sparsity.
            p = a.T @ z
            q = z.T @ z
            p_plus, p_minus = positive_part(p), negative_part(p)
            q_plus, q_minus = positive_part(q), negative_part(q)

            numer = p_plus + (y @ q_minus)
            denom = p_minus + (y @ q_plus) + cfg.sparsity_reg

            y = y * t.sqrt(numer / (denom + 1e-8))
            y = y.clamp_min(1e-8)

            # 3) Track loss and early stopping.
            a_hat = z @ y.T
            loss = t.norm(a - a_hat, p="fro").pow(2).item()
            history.append(loss)

            if loss < best_loss - cfg.tol:
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
        return SemiNMFResult(
            z=best_z,
            y=best_y,
            loss_history=history,
            best_iter=best_iter,
            best_loss=best_loss,
        )


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


def resolve_device(device: str) -> t.device:
    if device == "cpu":
        return t.device("cpu")
    if device == "cuda":
        return t.device("cuda" if t.cuda.is_available() else "cpu")
    if device == "mps":
        return t.device("mps" if t.backends.mps.is_available() else "cpu")
    # auto
    if t.cuda.is_available():
        return t.device("cuda")
    if t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")


def resolve_dtype(dtype: str) -> t.dtype:
    if dtype == "float32":
        return t.float32
    if dtype == "float64":
        return t.float64
    raise ValueError(f"Unsupported dtype '{dtype}'. Use float32 or float64.")


def append_result(path: Path, record: dict[str, Any]) -> None:
    """Append a run record to .json (array) or .jsonl (one JSON per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        if path.exists():
            existing = json.loads(path.read_text())
            if not isinstance(existing, list):
                raise ValueError(f"{path} exists but is not a JSON array.")
            existing.append(record)
            path.write_text(json.dumps(existing, indent=2))
        else:
            path.write_text(json.dumps([record], indent=2))
        return

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def default_results_file() -> Path:
    """Default JSONL output path under this directory's results/ folder."""
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return script_dir / "results" / f"snmf_fit_{timestamp}.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run toy-data Semi-NMF fit experiment and log similarity.",
    )

    # Toy data args.
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--num-ground-truth-features", type=int, default=512)
    parser.add_argument("--num-active-features", type=float, default=5.0)
    parser.add_argument("--decay-rate", type=float, default=0.99)
    parser.add_argument("--data-seed", type=int, default=42)

    # Semi-NMF args.
    parser.add_argument("--k-scale", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--closed-form-eqn-reg", type=float, default=1e-4)
    parser.add_argument("--sparsity-reg", type=float, default=0.1)
    parser.add_argument("--verbose-every", type=int, default=25)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--init", choices=["random", "svd", "knn"], default="random")
    parser.add_argument("--knn-iters", type=int, default=20)
    parser.add_argument("--knn-chunk-size", type=int, default=5_000)

    # Output/logging args.
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=(
            "Output path. If omitted, auto-saves to "
            "snmf/toy-data-generation/results/snmf_fit_<UTC timestamp>.jsonl."
        ),
    )
    parser.add_argument("--print-json", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    start_time = time.time()

    toy_cfg = ToyDataConfig(
        num_samples=args.num_samples,
        feature_dim=args.feature_dim,
        num_ground_truth_features=args.num_ground_truth_features,
        num_active_features=args.num_active_features,
        decay_rate=args.decay_rate,
        random_seed=args.data_seed,
    )
    g = toy_cfg.num_ground_truth_features

    model_cfg = SemiNMFConfig(
        k_scale=args.k_scale,
        max_iter=args.max_iter,
        tol=args.tol,
        patience=args.patience,
        closed_form_eqn_reg=args.closed_form_eqn_reg,
        sparsity_reg=args.sparsity_reg,
        verbose_every=args.verbose_every,
        seed=args.model_seed,
        dtype=args.dtype,
        init=args.init,
        knn_iters=args.knn_iters,
        knn_chunk_size=args.knn_chunk_size,
    )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    k = int(round(model_cfg.k_scale * g))
    if k <= 0:
        raise ValueError(f"Computed k must be positive, got {k}.")

    print(f"Using device={device}, dtype={dtype}, init={model_cfg.init}, G={g}, K={k}")
    dataset, ground_truth_features, _ = generate_toy_data(toy_cfg)

    # Semi-NMF fit expects A as (d_hidden, N).
    # Dataset is currently (N, d_hidden) so we transpose it here before feeding to model.
    a = t.tensor(dataset, dtype=dtype, device=device).T
    model = SemiNMF(k=k, device=device, dtype=dtype)
    result = model.fit(a, model_cfg)

    learned_features = result.z.detach().cpu().numpy().astype(np.float64, copy=False)
    similarity = mean_max_cosine_similarity(learned_features, ground_truth_features)
    elapsed = time.time() - start_time

    run_record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mean_max_cosine_similarity": similarity,
        "elapsed_seconds": elapsed,
        "num_samples": toy_cfg.num_samples,
        "k_scale": model_cfg.k_scale,
        "closed_form_eqn_reg": model_cfg.closed_form_eqn_reg,
        "sparsity_reg": model_cfg.sparsity_reg,
        "init": model_cfg.init,
    }

    results_file = args.results_file if args.results_file is not None else default_results_file()
    append_result(results_file, run_record)
    print(f"Appended result to {results_file}")

    print(f"Mean max cosine similarity: {similarity:.6f}")
    if args.print_json:
        print(json.dumps(run_record, indent=2))


if __name__ == "__main__":
    main()
