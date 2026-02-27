# Vanilla SAE Training

This experiment now uses Hydra to run single trainings and multirun grid searches for:

- `train.lr`
- `sae.hidden_dim`
- `sae.l1_lambda`

Hyperparameters are selected by the lowest `val/recon_loss`. The data split is `train/val/test`, where:

- `train` fits the SAE
- `val` selects the best epoch and best hyperparameters
- `test` is reserved for final evaluation after selection

## Setup

From the `snmf-mlp` directory:

```bash
pip install -r requirements.txt
```

Optional if you want Weights & Biases logging:

```bash
wandb login
```

## Single Run

```bash
cd snmf-mlp
PYTHONPATH=. python experiments/vanilla_sae/train.py
```

Useful overrides:

```bash
PYTHONPATH=. python experiments/vanilla_sae/train.py \
  train.lr=5e-4 \
  sae.hidden_dim=512 \
  sae.l1_lambda=1e-4 \
  evaluation.run_test=true
```

`train.num_workers` supports `auto`, `0`, or an explicit integer. `auto` resolves to a capped CPU-worker count and is intended for `DataLoader` workers, not GPU selection.

## Grid Search

The provided bash script launches a Hydra multirun sweep and then selects the best run by validation reconstruction loss.

```bash
cd snmf-mlp
experiments/vanilla_sae/run_sweep.sh
```

You can override the search space through environment variables:

```bash
cd snmf-mlp
LR_VALUES=1e-4,3e-4,5e-4 \
HIDDEN_DIM_VALUES=256,512,1024 \
L1_LAMBDA_VALUES=1e-5,1e-4,1e-3 \
experiments/vanilla_sae/run_sweep.sh
```

## Outputs

- Single runs write to `experiments/vanilla_sae/outputs/...`
- Sweeps write to `experiments/vanilla_sae/multirun/...`
- Each run stores:
  - `metrics.json`
  - `resolved_config.yaml`
  - `best_model.pt` when checkpoint saving is enabled
- Generated activations are cached in `experiments/vanilla_sae/cache/` so sweeps do not recompute them for every SAE hyperparameter combination

## Best Hyperparameter Selection

After a sweep, `select_best.py` writes `best_run_summary.json` into the multirun directory and prints the best override string.

Manual usage:

```bash
cd snmf-mlp
PYTHONPATH=. python experiments/vanilla_sae/select_best.py \
  --multirun-dir experiments/vanilla_sae/multirun/<date>/<time>
```

The summary also prints a suggested command for final test evaluation with `evaluation.run_test=true`.

## Dataset

Default dataset path:

```text
data/final_dataset_20_concepts.json
```

You can point to another dataset with:

```bash
PYTHONPATH=. python experiments/vanilla_sae/train.py \
  data.dataset_path=data/your_dataset.json \
  data.dataset_url=https://your-host/path/to/your_dataset.json
```
