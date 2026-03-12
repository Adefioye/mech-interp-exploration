#!/usr/bin/env bash
set -euo pipefail

# Grid sweep for run_snmf_fit_experiment.py
# Sweeps over:
#   - k_scale
#   - closed_form_eqn_reg
#   - sparsity_reg
#
# Usage:
#   bash snmf/toy-data-generation/sweep_snmf_grid.sh
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   RESULTS_FILE=snmf/toy-data-generation/results/sweep.jsonl
#   DEVICE=cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${SCRIPT_DIR}/run_snmf_fit_experiment.py"
DEVICE="${DEVICE:-cuda}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="${RESULTS_FILE:-${RESULTS_DIR}/snmf_grid_${TIMESTAMP}.jsonl}"

# 5-value sweep grids
K_SCALES=(0.75 1.00 1.25 1.50 2.00)
CLOSED_FORM_EQN_REGS=(1e-8 1e-7 1e-6 1e-5 1e-4)
SPARSITY_REGS=(0.0 1e-4 1e-3 1e-2 1e-1)

TOTAL_RUNS=$(( ${#K_SCALES[@]} * ${#CLOSED_FORM_EQN_REGS[@]} * ${#SPARSITY_REGS[@]} ))
RUN_IDX=0

echo "Starting grid sweep: ${TOTAL_RUNS} runs"
echo "Device: ${DEVICE}"
echo "Results file: ${RESULTS_FILE}"

for K_SCALE in "${K_SCALES[@]}"; do
  for CLOSED_FORM_EQN_REG in "${CLOSED_FORM_EQN_REGS[@]}"; do
    for SPARSITY_REG in "${SPARSITY_REGS[@]}"; do
      RUN_IDX=$((RUN_IDX + 1))
      echo ""
      echo "[${RUN_IDX}/${TOTAL_RUNS}] k_scale=${K_SCALE} closed_form_eqn_reg=${CLOSED_FORM_EQN_REG} sparsity_reg=${SPARSITY_REG}"

      "${PYTHON_BIN}" "${RUN_SCRIPT}" \
        --k-scale "${K_SCALE}" \
        --closed-form-eqn-reg "${CLOSED_FORM_EQN_REG}" \
        --sparsity-reg "${SPARSITY_REG}" \
        --device "${DEVICE}" \
        --results-file "${RESULTS_FILE}"
    done
  done
done

echo ""
echo "Sweep complete."
echo "Saved results to: ${RESULTS_FILE}"
