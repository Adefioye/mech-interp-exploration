import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import hydra
import numpy as np
import requests
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

# NOTE: only useful when not running with PYTHONPATH= . <python command>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_utils.concept_dataset import SupervisedConceptDataset
from llm_utils.activation_generator import ActivationGenerator
from sae import SAE


def log(message: str) -> None:
    print(f"[vanilla_sae] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_num_workers(value, cap: int) -> int:
    if isinstance(value, int):
        return max(0, value)

    if value is None:
        return 0

    normalized = str(value).strip().lower()
    if normalized == "auto":
        cpu_count = os.cpu_count() or 1
        return min(max(1, cpu_count - 1), max(0, cap))

    return max(0, int(normalized))


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def sanitize_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def ensure_dataset(dataset_path: Path, dataset_url: str) -> None:
    if dataset_path.exists():
        return

    if not dataset_url:
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}' and no dataset_url was provided."
        )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading dataset to {dataset_path}")
    response = requests.get(dataset_url, timeout=60)
    response.raise_for_status()
    dataset_path.write_bytes(response.content)


def activation_cache_path(cfg: DictConfig, dataset_path: Path) -> Path:
    cache_dir = resolve_path(cfg.data.activation_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{sanitize_fragment(dataset_path.stem)}"
        f"__{sanitize_fragment(cfg.model_name)}"
        f"__{sanitize_fragment(cfg.factorization_mode)}"
        f"__layer{cfg.layer}.pt"
    )
    return cache_dir / filename


def load_or_generate_activations(
    cfg: DictConfig,
    dataset_path: Path,
    model_device: torch.device,
) -> Tuple[torch.Tensor, Path]:
    cache_path = activation_cache_path(cfg, dataset_path)
    if cache_path.exists() and not cfg.data.force_regenerate_activations:
        log(f"Loading cached activations from {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        return payload["activations"].float(), cache_path

    dataset = SupervisedConceptDataset(str(dataset_path))
    generator = ActivationGenerator(
        cfg.model_name,
        model_device=str(model_device),
        data_device=cfg.train.data_device,
        mode=cfg.factorization_mode,
    )
    activations, _ = generator.generate_multiple_layer_activations_and_freq(
        dataset,
        [cfg.layer],
        batch_size=cfg.data.activation_batch_size,
    )
    activation_tensor = activations[0].float().cpu()
    torch.save(
        {
            "activations": activation_tensor,
            "metadata": {
                "dataset_path": str(dataset_path),
                "model_name": cfg.model_name,
                "factorization_mode": cfg.factorization_mode,
                "layer": cfg.layer,
            },
        },
        cache_path,
    )
    log(f"Saved activation cache to {cache_path}")
    return activation_tensor, cache_path


def split_activations(activations: torch.Tensor, cfg: DictConfig) -> Dict[str, torch.Tensor]:
    train_ratio = float(cfg.data.train_ratio)
    val_ratio = float(cfg.data.val_ratio)
    test_ratio = float(cfg.data.test_ratio)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {ratio_sum:.4f}"
        )

    total_samples = activations.size(0)
    if total_samples < 3:
        raise ValueError("Need at least 3 activation samples to form train/val/test splits.")

    train_idx = int(total_samples * train_ratio)
    val_idx = int(total_samples * val_ratio) + train_idx

    if train_idx == 0 or val_idx == 0:
        raise ValueError(
            "Split ratios produced an empty split. Adjust train/val/test ratios or use more data."
        )

    return {
        "train": activations[:train_idx],
        "val": activations[train_idx:val_idx],
        "test": activations[val_idx:],
    }


def build_loader(
    activations: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    dataset = TensorDataset(activations)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_training(cfg: DictConfig) -> Dict:
    set_seed(int(cfg.seed))

    train_device = resolve_device(cfg.train.device)
    model_device = resolve_device(cfg.train.model_device)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_path(cfg.data.dataset_path)
    ensure_dataset(dataset_path, cfg.data.dataset_url)

    log(f"Using dataset: {dataset_path}")
    activations, cache_path = load_or_generate_activations(cfg, dataset_path, model_device)
    splits = split_activations(activations, cfg)

    pin_memory = train_device.type == "cuda"
    num_workers = resolve_num_workers(cfg.train.num_workers, int(cfg.train.num_workers_cap))
    persistent_workers = num_workers > 0
    train_loader = build_loader(
        splits["train"],
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = build_loader(
        splits["val"],
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = build_loader(
        splits["test"],
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    sae = SAE(
        input_dim=activations.size(1),
        hidden_dim=int(cfg.sae.hidden_dim),
        l1_lambda=float(cfg.sae.l1_lambda),
        eps=float(cfg.sae.eps),
    )

    checkpoint_path = output_dir / "best_model.pt" if cfg.train.save_best_checkpoint else None
    wandb_run_name = (
        f"lr={cfg.train.lr}_hidden={cfg.sae.hidden_dim}_l1={cfg.sae.l1_lambda}"
    )
    fit_result = sae.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(cfg.train.epochs),
        lr=float(cfg.train.lr),
        device=train_device,
        patience=int(cfg.train.patience) if cfg.train.patience is not None else None,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        use_wandb=bool(cfg.logging.use_wandb),
        wandb_project=cfg.logging.wandb_project,
        wandb_entity=cfg.logging.wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_config=OmegaConf.to_container(cfg, resolve=True),
    )

    # We are re-evaluating so we get easy access to metrics on validation data on the best model.
    val_metrics = sae.evaluate(val_loader, device=train_device)
    test_metrics = None
    if cfg.evaluation.run_test:
        test_metrics = sae.evaluate(test_loader, device=train_device)

    metrics = {
        "best_epoch": fit_result["best_epoch"],
        "best_val_recon_loss": fit_result["best_val_recon_loss"],
        "selection_metric": cfg.evaluation.selection_metric,
        "selection_value": fit_result["best_val_recon_loss"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": fit_result["checkpoint_path"],
        "activation_cache_path": str(cache_path),
        "input_dim": int(activations.size(1)),
        "num_samples": int(activations.size(0)),
        "split_sizes": {name: int(tensor.size(0)) for name, tensor in splits.items()},
        "hyperparameters": {
            "lr": float(cfg.train.lr),
            "hidden_dim": int(cfg.sae.hidden_dim),
            "l1_lambda": float(cfg.sae.l1_lambda),
        },
        "runtime": {
            "device": str(train_device),
            "model_device": str(model_device),
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
        },
        "history": fit_result["history"],
    }
    save_json(output_dir / "metrics.json", metrics)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    log(
        "Completed run with "
        f"best_val_recon_loss={metrics['best_val_recon_loss']:.6f}"
    )
    if test_metrics is not None:
        log(f"Test recon loss={test_metrics['recon_loss']:.6f}")

    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")
    run_training(cfg)


if __name__ == "__main__":
    main()
