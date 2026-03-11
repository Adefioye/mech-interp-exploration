from __future__ import annotations

from autoencoder.base_extractor import BaseFeatureExtractor
from autoencoder.semi_nmf_extractor import SemiNMFExtractor, SemiNMFConfig
from autoencoder.nmf_extractor import NMF, NMFConfig
from autoencoder.sparse_nmf_extractor import SparseNMFConfig, SparseNMFExtractor


def build_extractor(
    method: str,
    *,
    semi_cfg: SemiNMFConfig,
    nmf_cfg: NMFConfig,
    sparse_nmf_cfg: SparseNMFConfig,
) -> BaseFeatureExtractor:
    if method == "semi_nmf":
        return SemiNMFExtractor(semi_cfg)
    if method == "nmf":
        return NMF(nmf_cfg)
    if method == "sparse_nmf":
        return SparseNMFExtractor(sparse_nmf_cfg)
    raise ValueError(
        f"Unknown method '{method}'. Use one of: semi_nmf, nmf, sparse_nmf."
    )
