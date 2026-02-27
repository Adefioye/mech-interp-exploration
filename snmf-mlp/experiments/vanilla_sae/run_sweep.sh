#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

LR_VALUES="${LR_VALUES:-1e-4,5e-4}"
HIDDEN_DIM_VALUES="${HIDDEN_DIM_VALUES:-256,512}"
L1_LAMBDA_VALUES="${L1_LAMBDA_VALUES:-1e-4,1e-3}"
TOP_K="${TOP_K:-3}"
SWEEP_STAMP="$(date +"%Y-%m-%d/%H-%M-%S")"
SWEEP_DIR="experiments/vanilla_sae/multirun/${SWEEP_STAMP}"
SUMMARY_DIR="experiments/vanilla_sae/reports"
SUMMARY_FILE="${SUMMARY_DIR}/best_run_summary_${SWEEP_STAMP//\//_}.json"

mkdir -p "${SUMMARY_DIR}"

PYTHONPATH=. python experiments/vanilla_sae/train.py -m \
  hydra.sweep.dir="${SWEEP_DIR}" \
  train.lr="${LR_VALUES}" \
  sae.hidden_dim="${HIDDEN_DIM_VALUES}" \
  sae.l1_lambda="${L1_LAMBDA_VALUES}" \
  evaluation.run_test=false

PYTHONPATH=. python experiments/vanilla_sae/select_best.py \
  --multirun-dir "${SWEEP_DIR}" \
  --top-k "${TOP_K}" \
  --output-file "${SUMMARY_FILE}"
