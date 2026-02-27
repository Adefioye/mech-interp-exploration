import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml


def load_metrics(metrics_path: Path) -> Dict:
    return json.loads(metrics_path.read_text())


def maybe_load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def build_override_string(entry: Dict) -> str:
    hyperparameters = entry["hyperparameters"]
    return (
        f"train.lr={hyperparameters['lr']} "
        f"sae.hidden_dim={hyperparameters['hidden_dim']} "
        f"sae.l1_lambda={hyperparameters['l1_lambda']}"
    )


def collect_runs(multirun_dir: Path) -> List[Dict]:
    runs = []
    for metrics_path in sorted(multirun_dir.rglob("metrics.json")):
        metrics = load_metrics(metrics_path)
        run_dir = metrics_path.parent
        config = maybe_load_config(run_dir / "resolved_config.yaml")
        entry = {
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "config_path": str(run_dir / "resolved_config.yaml"),
            "best_val_recon_loss": metrics["best_val_recon_loss"],
            "best_epoch": metrics["best_epoch"],
            "hyperparameters": metrics["hyperparameters"],
            "val_metrics": metrics["val_metrics"],
            "test_metrics": metrics["test_metrics"],
            "resolved_config": config,
        }
        entry["override_string"] = build_override_string(entry)
        runs.append(entry)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best Hydra sweep run for vanilla SAE.")
    parser.add_argument(
        "--multirun-dir",
        required=True,
        help="Path to a Hydra multirun directory that contains metrics.json files.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output path for the best-run summary JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top runs to save in the summary.",
    )
    args = parser.parse_args()

    multirun_dir = Path(args.multirun_dir).resolve()
    if not multirun_dir.exists():
        raise FileNotFoundError(f"Multirun directory not found: {multirun_dir}")

    runs = collect_runs(multirun_dir)
    if not runs:
        raise FileNotFoundError(f"No metrics.json files found under: {multirun_dir}")

    ranked_runs = sorted(runs, key=lambda entry: entry["best_val_recon_loss"])
    top_k = max(1, min(args.top_k, len(ranked_runs)))
    best_run = ranked_runs[0]
    top_runs = ranked_runs[:top_k]
    summary = {
        "selection_metric": "val/recon_loss",
        "multirun_dir": str(multirun_dir),
        "num_runs": len(ranked_runs),
        "top_k": top_k,
        "best_run": {
            **best_run,
            "suggested_test_command": (
                "PYTHONPATH=. python experiments/vanilla_sae/train.py "
                f"{best_run['override_string']} evaluation.run_test=true"
            ),
        },
        "top_runs": top_runs,
        "ranked_runs": ranked_runs,
    }

    output_file = (
        Path(args.output_file).resolve()
        if args.output_file
        else multirun_dir / "best_run_summary.json"
    )
    output_file.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Best run: {best_run['run_dir']}")
    print(f"Best val/recon_loss: {best_run['best_val_recon_loss']:.6f}")
    print(f"Top {top_k} runs:")
    for i, run in enumerate(top_runs, start=1):
        print(
            f"  {i}. loss={run['best_val_recon_loss']:.6f} "
            f"overrides={run['override_string']}"
        )
    print(f"Summary written to: {output_file}")


if __name__ == "__main__":
    main()
