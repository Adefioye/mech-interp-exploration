from __future__ import annotations

from autoencoder.base_extractor import BaseFeatureExtractor
from autoencoder.semi_nmf_extractor import SemiNMFExtractor, SemiNMFExtractorConfig
from autoencoder.sklearn_nmf_extractor import SklearnNMFExtractor, SklearnNMFExtractorConfig
from autoencoder.torchnmf_sparse_extractor import TorchNMFExtractorConfig, TorchNMFSparseExtractor


def build_extractor(
    method: str,
    *,
    semi_cfg: SemiNMFExtractorConfig,
    sklearn_cfg: SklearnNMFExtractorConfig,
    torchnmf_cfg: TorchNMFExtractorConfig,
) -> BaseFeatureExtractor:
    if method == "semi_nmf":
        return SemiNMFExtractor(semi_cfg)
    if method == "sklearn_nmf":
        return SklearnNMFExtractor(sklearn_cfg)
    if method == "torchnmf_sparse_nmf":
        return TorchNMFSparseExtractor(torchnmf_cfg)
    raise ValueError(
        f"Unknown method '{method}'. Use one of: semi_nmf, sklearn_nmf, torchnmf_sparse_nmf."
    )
