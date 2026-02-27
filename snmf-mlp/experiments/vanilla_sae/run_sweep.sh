#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

LR_VALUES="${LR_VALUES:-1e-4,5e-4}"
HIDDEN_DIM_VALUES="${HIDDEN_DIM_VALUES:-256,512}"
L1_LAMBDA_VALUES="${L1_LAMBDA_VALUES:-1e-4,1e-3}"
SWEEP_STAMP="$(date +"%Y-%m-%d/%H-%M-%S")"
SWEEP_DIR="experiments/vanilla_sae/multirun/${SWEEP_STAMP}"

PYTHONPATH=. python experiments/vanilla_sae/train.py -m \
  hydra.sweep.dir="${SWEEP_DIR}" \
  train.lr="${LR_VALUES}" \
  sae.hidden_dim="${HIDDEN_DIM_VALUES}" \
  sae.l1_lambda="${L1_LAMBDA_VALUES}" \
  evaluation.run_test=false

PYTHONPATH=. python experiments/vanilla_sae/select_best.py \
  --multirun-dir "${SWEEP_DIR}"
