#!/usr/bin/env python3
"""Select top-k experiment configs by mean_max_cosine_similarity.

Reads a results file (.jsonl or .json), sorts runs by
`mean_max_cosine_similarity` descending, and writes top-k to a new .jsonl file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEY = "mean_max_cosine_similarity"


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {path}.")
        return [r for r in payload if isinstance(r, dict)]

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {line_num} in {path} is not a JSON object."
                )
            records.append(obj)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def metric_value(record: dict[str, Any]) -> float:
    value = record.get(METRIC_KEY, float("-inf"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a JSONL file containing top-k configs by similarity.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input results file (.jsonl or .json).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output file (.jsonl).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top runs to keep.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be positive, got {args.top_k}.")

    records = load_records(args.input_file)
    if not records:
        raise ValueError(f"No valid records found in {args.input_file}.")

    sorted_records = sorted(records, key=metric_value, reverse=True)
    top_records = sorted_records[: args.top_k]

    # Add rank and explicit metric to make downstream review easier.
    ranked: list[dict[str, Any]] = []
    for idx, rec in enumerate(top_records, start=1):
        with_rank = dict(rec)
        with_rank["rank"] = idx
        with_rank["rank_metric"] = METRIC_KEY
        with_rank["rank_metric_value"] = metric_value(rec)
        ranked.append(with_rank)

    write_jsonl(args.output_file, ranked)

    best = ranked[0]["rank_metric_value"]
    print(
        f"Wrote top {len(ranked)} / {len(records)} records to {args.output_file} "
        f"(best {METRIC_KEY}={best:.6f})."
    )


if __name__ == "__main__":
    main()

