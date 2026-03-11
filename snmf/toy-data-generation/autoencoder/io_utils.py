from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch as t


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


def default_results_file(prefix: str = "feature_extraction") -> Path:
    script_dir = Path(__file__).resolve().parent.parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return script_dir / "results" / f"{prefix}_{timestamp}.jsonl"


def count_negative_elements(data: t.Tensor | np.ndarray | list[Any]) -> int:
    """Return the total number of negative values in tensor/ndarray/list input."""
    if isinstance(data, t.Tensor):
        return int((data < 0).sum().item())

    if isinstance(data, (np.ndarray, list)):
        arr = np.asarray(data)
        try:
            return int(np.sum(arr < 0))
        except TypeError as exc:
            raise TypeError("Input must contain numeric values.") from exc

    raise TypeError(f"Expected torch.Tensor, np.ndarray, or list; got {type(data)!r}.")
