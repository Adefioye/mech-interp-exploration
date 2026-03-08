from __future__ import annotations

import random

import numpy as np
import torch as t


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def resolve_device(device: str) -> t.device:
    if device == "cpu":
        return t.device("cpu")
    if device == "cuda":
        return t.device("cuda" if t.cuda.is_available() else "cpu")
    if device == "mps":
        return t.device("mps" if t.backends.mps.is_available() else "cpu")
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
