import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm


class SAE(nn.Module):
    """
    Sparse autoencoder with an L1 penalty on hidden activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_lambda: float,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.l1_lambda = float(l1_lambda)
        self.eps = float(eps)

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.encoder.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.encoder.bias, -bound, bound)

        nn.init.xavier_uniform_(self.decoder.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.decoder.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.decoder.bias, -bound, bound)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b_d = self.decoder.bias
        return F.relu(self.encoder(x - b_d))

    def decode(self, activations: torch.Tensor) -> torch.Tensor:
        decoder_weight = self._normalize_columns(self.decoder.weight)
        return activations @ decoder_weight.T + self.decoder.bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        activations = self.encode(x)
        reconstruction = self.decode(activations)
        return reconstruction, activations

    def _normalize_columns(self, weights: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.norm(weights, dim=0, keepdim=True).clamp_min(self.eps)
        return weights / norms

    @staticmethod
    def _unwrap_batch(batch: Any) -> torch.Tensor:
        return batch[0] if isinstance(batch, (list, tuple)) else batch

    def _compute_losses(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction, activations = self(x)
        recon_loss = F.mse_loss(reconstruction, x, reduction="mean")
        l1_loss = activations.abs().mean()
        total_loss = recon_loss + self.l1_lambda * l1_loss
        return {
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            "total_loss": total_loss,
        }

    def evaluate(self, data_loader, device: Optional[torch.device] = None) -> Dict[str, float]:
        device = device or next(self.parameters()).device
        self.eval()

        total_recon = 0.0
        total_l1 = 0.0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                x = self._unwrap_batch(batch).to(device)
                losses = self._compute_losses(x)
                batch_size = x.size(0)

                total_recon += losses["recon_loss"].item() * batch_size
                total_l1 += losses["l1_loss"].item() * batch_size
                total_loss += losses["total_loss"].item() * batch_size
                total_samples += batch_size

        divisor = max(1, total_samples)
        return {
            "recon_loss": total_recon / divisor,
            "l1_loss": total_l1 / divisor,
            "total_loss": total_loss / divisor,
            "samples": total_samples,
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 1,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        patience: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        use_wandb: bool = True,
        wandb_project: str = "vanilla-sae",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if val_loader is None:
            raise ValueError("val_loader must be provided for model selection.")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        checkpoint_target = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint_target is not None:
            checkpoint_target.parent.mkdir(parents=True, exist_ok=True)

        wandb_run = None
        if use_wandb:
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=wandb_config,
            )
            wandb.watch(self, log="gradients", log_freq=100)

        best_val_recon = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_state_dict = None
        history = []

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_start = time.time()
            train_recon = 0.0
            train_l1 = 0.0
            train_total = 0.0
            train_samples = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs}",
                leave=False,
            )
            for batch in progress:
                x = self._unwrap_batch(batch).to(device)
                optimizer.zero_grad()

                losses = self._compute_losses(x)
                losses["total_loss"].backward()
                optimizer.step()

                batch_size = x.size(0)
                train_recon += losses["recon_loss"].item() * batch_size
                train_l1 += losses["l1_loss"].item() * batch_size
                train_total += losses["total_loss"].item() * batch_size
                train_samples += batch_size

            train_divisor = max(1, train_samples)
            train_metrics = {
                "recon_loss": train_recon / train_divisor,
                "l1_loss": train_l1 / train_divisor,
                "total_loss": train_total / train_divisor,
                "samples": train_samples,
            }
            val_metrics = self.evaluate(val_loader, device=device)
            epoch_duration = time.time() - epoch_start

            is_best = val_metrics["recon_loss"] < best_val_recon
            if is_best:
                best_val_recon = val_metrics["recon_loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state_dict = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.state_dict().items()
                }
                if checkpoint_target is not None:
                    torch.save(best_state_dict, checkpoint_target)
            else:
                epochs_without_improvement += 1

            epoch_metrics = {
                "epoch": epoch,
                "train/recon_loss": train_metrics["recon_loss"],
                "train/l1_loss": train_metrics["l1_loss"],
                "train/total_loss": train_metrics["total_loss"],
                "val/recon_loss": val_metrics["recon_loss"],
                "val/l1_loss": val_metrics["l1_loss"],
                "val/total_loss": val_metrics["total_loss"],
                "time/epoch_sec": epoch_duration,
            }
            history.append(epoch_metrics)

            if wandb_run is not None:
                wandb.log(epoch_metrics, step=epoch)

            print(
                f"[fit] epoch={epoch}/{epochs} "
                f"train_total={train_metrics['total_loss']:.6f} "
                f"val_recon={val_metrics['recon_loss']:.6f} "
                f"best_val_recon={best_val_recon:.6f} "
                f"time={epoch_duration:.2f}s",
                flush=True,
            )

            if patience is not None and epochs_without_improvement >= patience:
                print(
                    f"[fit] early stopping triggered after {epoch} epochs "
                    f"(patience={patience}).",
                    flush=True,
                )
                break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        if wandb_run is not None:
            wandb_run.summary["best_epoch"] = best_epoch
            wandb_run.summary["best_val_recon_loss"] = best_val_recon
            wandb_run.finish()

        return {
            "best_epoch": best_epoch,
            "best_val_recon_loss": best_val_recon,
            "checkpoint_path": str(checkpoint_target) if checkpoint_target else None,
            "history": history,
        }
