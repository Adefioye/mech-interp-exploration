import math
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm


class SAE(nn.Module):
    """
    Sparse Autoencoder

    Encoder:  a = W_e (x - b_d) + b_e
    Decoder:  x̂ = W_d^T a + b_d

    Forward returns:
        recon, activations   # shapes: (batch_size, input_dim), (batch_size, hidden_dim)

    Args
    ----
    input_dim : int
        input dimensionality
    hidden_dim : int
        number of dictionary atoms (latent units)
    eps : float
        Numerical stability for normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_lambda: float,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.l1_lambda = l1_lambda
        self.eps = float(eps)

        # Initialize necessary weights and biases
        # self.encoder: Project from input dimension to hidden dimension
        # self.decoder: Project from hidden dimension, k to input dimension, d

        # Encoder (# Implement me!)
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        # Decoder (# Implement me!)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        self.init_parameters()

    def init_parameters(self) -> None:
        # Kaiming init for encoder (good with ReLU), xavier for decoder.
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.encoder.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.encoder.bias, -bound, bound)

        nn.init.xavier_uniform_(self.decoder.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.decoder.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.decoder.bias, -bound, bound)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return activations a (after ReLU nonlinearity), shape (B, H)."""
        b_d = self.decoder.bias
        output = self.encoder(x - b_d) # We didn't add b_e as it is already inside of self.encoder
        return F.relu(output)

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        """Decode activations to reconstruction, shape (B, D)."""
        # Normalize decoder weights before decoding
        W = self._normalize_columns(self.decoder.weight) # (D, H)
        output = a @ W.T + self.decoder.bias # (B, H) @ (H, D) -> (B, D)
        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (batch_size, input_dim)
        Returns: (recon, a)
            recon: the reconstructed activations `x_hat` (batch_size, input_dim)
            a: the encoder feature activations `a` (batch_size, hidden_dim)
        """
        # Implement me!
        a = self.encode(x)
        recon = self.decode(a)
        return (recon, a)

    def _normalize_columns(self, W: torch.Tensor) -> torch.Tensor:
        # W: (D, H) where each column is an atom; normalize each column
        norms = torch.linalg.norm(W, dim=0, keepdim=True).clamp_min(self.eps)
        return W / norms

    def fit(self, train_loader, val_loader, epochs: int = 1, lr: float = 1e-3, device=None):
        self.train()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        run = wandb.init(project="my-sae", name=f"{lr}-{epochs}-{l1_lambda}-{hidden_dim}", config={
            "lr": lr,
            "epochs": epochs,
            "l1_lambda": self.l1_lambda,
            "model": "SAE",
            "input_dim": self.encoder.in_features,
            "hidden_dim": self.encoder.out_features,
        })
        wandb.watch(self, log="gradients", log_freq=100)

        for epoch in range(epochs):
            
            self.train()

            epoch_start = time.time()
            train_loss, train_n = 0.0, 0
            print(f"[fit] Epoch {epoch+1}/{epochs} started.")

            ##### Train Loss #####
            for batch_idx, batch in tqdm(enumerate(train_loader)):
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)

                opt.zero_grad()

                # Implement the training loop!
                # 1. Make a the forward pass to compute the `recon` and activations `a`.
                # 2. Compute the MSE loss between `recon` and `x`.
                # 3. Compute the L1 loss on the activations `a`.
                # 4. Compute the total `loss`.
                recon, act = self(x)
                recon_loss = F.mse_loss(recon, x, reduction="mean")
                l1 = act.abs().mean()
                loss = recon_loss + self.l1_lambda * l1

                loss.backward()
                opt.step()

                train_loss += loss.item() * x.size(0)
                train_n += x.size(0)

            train_epoch_loss = train_loss / max(1, train_n)
            dur = time.time() - epoch_start
            ############################

            ##### Validation Loss #####
            self.eval()
            val_loss, val_n = 0.0, 0
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)

                recon, act = self(x)

                loss = F.mse_loss(recon, x, reduction="mean")

                val_loss += loss.item() * x.size(0)
                val_n += x.size(0)

            val_loss = val_loss / max(val_n, 1)
            ############################

            # history.append((train_epoch_loss, val_loss))

            print(f"[fit] Epoch {epoch+1}/{epochs} completed. "
                f"AvgTrainLoss={train_epoch_loss:.6f} | TrainSamples={train_n} | AvgValLoss={val_loss:.6f} | ValSamples={val_n} | Time={dur:.2f}s")


            wandb.log({
                "train/epoch_loss": train_epoch_loss,
                "val/epoch_loss": val_loss,
                "train/epoch": epoch + 1,
                "train/samples": n,
                "time/epoch_sec": dur,
            }, step=epoch + 1)

        run.finish()
