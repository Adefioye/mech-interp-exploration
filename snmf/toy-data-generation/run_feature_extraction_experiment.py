#!/usr/bin/env python3
"""Run toy-data feature extraction experiments across multiple techniques.

Techniques supported:
- semi_nmf
- nmf
- sparse_nmf
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoencoder.common import resolve_device, resolve_dtype
from autoencoder.factory import build_extractor
from autoencoder.io_utils import append_result, default_results_file
from autoencoder.metrics import mean_max_cosine_similarity
from autoencoder.semi_nmf_extractor import SemiNMFConfig
from autoencoder.nmf_extractor import NMFConfig
from autoencoder.sparse_nmf_extractor import SparseNMFConfig
from autoencoder.toy_data import ToyDataConfig, generate_toy_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run toy-data feature extraction with selectable method.",
    )

    # Method selection.
    parser.add_argument(
        "--method",
        choices=["semi_nmf", "nmf", "sparse_nmf"],
        default="semi_nmf",
    )

    # Shared sizing.
    parser.add_argument("--k-scale", type=float, default=2.0)
    parser.add_argument("--n-components", type=int, default=None)

    # Toy data.
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--d-hidden", type=int, default=256)
    parser.add_argument("--num-ground-truth-features", type=int, default=512)
    parser.add_argument("--num-active-features", type=float, default=5.0)
    parser.add_argument("--decay-rate", type=float, default=0.99)
    parser.add_argument("--data-seed", type=int, default=42)

    # Shared runtime.
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="mps")

    # Semi-NMF-specific.
    parser.add_argument("--semi-nmf-max-iter", type=int, default=500)
    parser.add_argument("--semi-nmf-tol", type=float, default=1e-6)
    parser.add_argument("--semi-nmf-patience", type=int, default=30)
    parser.add_argument("--semi-nmf-closed-form-eqn-reg", type=float, default=1e-4)
    parser.add_argument("--semi-nmf-sparsity-reg", type=float, default=0.1)
    parser.add_argument("--semi-nmf-verbose-every", type=int, default=25)
    parser.add_argument("--semi-nmf-seed", type=int, default=42)
    parser.add_argument("--semi-nmf-init", choices=["random", "svd", "knn"], default="random")
    parser.add_argument("--semi-nmf-knn-iters", type=int, default=20)
    parser.add_argument("--semi-nmf-knn-chunk-size", type=int, default=5_000)

    # sklearn NMF-specific.
    parser.add_argument("--nmf-max-iter", type=int, default=500)
    parser.add_argument("--nmf-tol", type=float, default=1e-4)
    parser.add_argument("--nmf-init", default="nndsvda")
    parser.add_argument("--nmf-solver", choices=["cd", "mu"], default="cd")
    parser.add_argument("--nmf-beta-loss", default="frobenius")
    parser.add_argument("--nmf-alpha-w", type=float, default=0.0)
    parser.add_argument("--nmf-alpha-h", type=float, default=0.0)
    parser.add_argument("--nmf-l1-ratio", type=float, default=0.0)
    parser.add_argument("--nmf-random-state", type=int, default=42)
    parser.add_argument("--nmf-no-shift", action="store_true")

    # torchnmf sparse NMF-specific.
    parser.add_argument("--sparse-nmf-max-iter", type=int, default=500)
    parser.add_argument("--sparse-nmf-beta", type=float, default=2.0)
    parser.add_argument("--sparse-nmf-s-w", type=float, default=None)
    parser.add_argument("--sparse-nmf-s-h", type=float, default=None)
    parser.add_argument("--sparse-nmf-l1-strength", type=float, default=0.0)
    parser.add_argument("--sparse-nmf-seed", type=int, default=42)
    parser.add_argument("--sparse-nmf-verbose", action="store_true")
    parser.add_argument("--sparse-nmf-no-shift", action="store_true")

    # Logging.
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=(
            "Output path. If omitted, auto-saves to "
            "snmf/toy-data-generation/results/<method>_fit_<UTC timestamp>.jsonl."
        ),
    )
    parser.add_argument("--print-json", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    start_time = time.time()

    toy_cfg = ToyDataConfig(
        num_samples=args.num_samples,
        d_hidden=args.d_hidden,
        num_ground_truth_features=args.num_ground_truth_features,
        num_active_features=args.num_active_features,
        decay_rate=args.decay_rate,
        random_seed=args.data_seed,
    )
    g = toy_cfg.num_ground_truth_features

    n_components = args.n_components
    if n_components is None:
        n_components = int(round(args.k_scale * g))
    if n_components <= 0:
        raise ValueError(f"Computed n_components must be positive, got {n_components}.")

    dtype_t = resolve_dtype(args.dtype)
    device_t = resolve_device(args.device)

    semi_cfg = SemiNMFConfig(
        n_components=n_components,
        max_iter=args.semi_nmf_max_iter,
        tol=args.semi_nmf_tol,
        patience=args.semi_nmf_patience,
        closed_form_eqn_reg=args.semi_nmf_closed_form_eqn_reg,
        sparsity_reg=args.semi_nmf_sparsity_reg,
        verbose_every=args.semi_nmf_verbose_every,
        seed=args.semi_nmf_seed,
        init=args.semi_nmf_init,
        knn_iters=args.semi_nmf_knn_iters,
        knn_chunk_size=args.semi_nmf_knn_chunk_size,
        dtype=dtype_t,
        device=device_t,
    )

    nmf_cfg = NMFConfig(
        n_components=n_components,
        max_iter=args.nmf_max_iter,
        tol=args.nmf_tol,
        init=args.nmf_init,
        solver=args.nmf_solver,
        beta_loss=args.nmf_beta_loss,
        alpha_w=args.nmf_alpha_w,
        alpha_h=args.nmf_alpha_h,
        l1_ratio=args.nmf_l1_ratio,
        random_state=args.nmf_random_state,
        shift_to_nonnegative=not args.nmf_no_shift,
    )

    sparse_nmf_cfg = SparseNMFConfig(
        n_components=n_components,
        max_iter=args.sparse_nmf_max_iter,
        beta=args.sparse_nmf_beta,
        s_w=args.sparse_nmf_s_w,
        s_h=args.sparse_nmf_s_h,
        l1_strength=args.sparse_nmf_l1_strength,
        seed=args.sparse_nmf_seed,
        dtype=args.dtype,
        device=args.device,
        shift_to_nonnegative=not args.sparse_nmf_no_shift,
        verbose=args.sparse_nmf_verbose,
    )

    print(
        f"Running method={args.method} device={device_t} dtype={dtype_t} "
        f"G={g} K={n_components}"
    )

    dataset, ground_truth_features, _ = generate_toy_data(toy_cfg)
    extractor = build_extractor(
        args.method,
        semi_cfg=semi_cfg,
        nmf_cfg=nmf_cfg,
        sparse_nmf_cfg=sparse_nmf_cfg,
    )

    fit_result = extractor.fit(dataset)
    similarity = mean_max_cosine_similarity(
        fit_result.learned_features,
        ground_truth_features,
    )
    elapsed = time.time() - start_time

    run_record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": extractor.method_name,
        "mean_max_cosine_similarity": similarity,
        "elapsed_seconds": elapsed,
        "n_components": n_components,
        "k_scale": args.k_scale,
        "toy_data_config": asdict(toy_cfg),
        "method_config": extractor.get_config(),
        "reconstruction_loss": fit_result.reconstruction_loss,
        "fit_metadata": fit_result.metadata,
    }

    results_file = (
        args.results_file
        if args.results_file is not None
        else default_results_file(prefix=f"{extractor.method_name}_fit")
    )
    append_result(results_file, run_record)
    print(f"Appended result to {results_file}")

    print(f"Mean max cosine similarity: {similarity:.6f}")
    if args.print_json:
        print(json.dumps(run_record, indent=2))


if __name__ == "__main__":
    main()
