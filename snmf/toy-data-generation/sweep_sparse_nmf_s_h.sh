#!/usr/bin/env bash
set -euo pipefail

# Sweep run_feature_extraction_experiment.py for sparse_nmf by varying s_h
# over values strictly between 0 and 1.
#
# Usage:
#   bash snmf/toy-data-generation/sweep_sparse_nmf_s_h.sh [extra args...]
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   RESULTS_FILE=snmf/toy-data-generation/results/sparse_nmf_s_h_sweep.jsonl
#   S_H_VALUES_STR="0.05 0.2 0.4 0.6 0.8 0.95"
#   SPARSE_NMF_S_W=0.1
#   DEVICE=cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${SCRIPT_DIR}/run_feature_extraction_experiment.py"
DEVICE="${DEVICE:-cuda}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="${RESULTS_FILE:-${RESULTS_DIR}/sparse_nmf_s_h_sweep_${TIMESTAMP}.jsonl}"

S_H_VALUES_STR="${S_H_VALUES_STR:-0.05 0.2 0.4 0.6 0.8 0.95}"
SPARSE_NMF_S_W="${SPARSE_NMF_S_W:-}"
read -r -a S_H_VALUES <<< "${S_H_VALUES_STR}"

TOTAL_RUNS="${#S_H_VALUES[@]}"
RUN_IDX=0

echo "Starting sparse_nmf s_h sweep: ${TOTAL_RUNS} runs"
echo "s_h values (must satisfy 0 < s_h < 1): ${S_H_VALUES_STR}"
echo "Device: ${DEVICE}"
if [[ -n "${SPARSE_NMF_S_W}" ]]; then
  echo "Fixed s_w: ${SPARSE_NMF_S_W}"
fi
echo "Results file: ${RESULTS_FILE}"

for S_H in "${S_H_VALUES[@]}"; do
  RUN_IDX=$((RUN_IDX + 1))
  echo ""
  echo "[${RUN_IDX}/${TOTAL_RUNS}] s_h=${S_H}"

  CMD=(
    "${PYTHON_BIN}" "${RUN_SCRIPT}"
    --method sparse_nmf
    --sparse-nmf-s-h "${S_H}"
    --results-file "${RESULTS_FILE}"
  )
  if [[ -n "${SPARSE_NMF_S_W}" ]]; then
    CMD+=(--sparse-nmf-s-w "${SPARSE_NMF_S_W}")
  fi
  CMD+=("$@")
  CMD+=(--device "${DEVICE}")

  "${CMD[@]}"
done

echo ""
echo "s_h sweep complete."
echo "Saved results to: ${RESULTS_FILE}"
