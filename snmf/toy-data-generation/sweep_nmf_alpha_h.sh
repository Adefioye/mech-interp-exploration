#!/usr/bin/env bash
set -euo pipefail

# Sweep run_feature_extraction_experiment.py for NMF by fixing l1_ratio
# and varying alpha_H.
#
# Usage:
#   bash snmf/toy-data-generation/sweep_nmf_alpha_h.sh [extra args...]
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   RESULTS_FILE=snmf/toy-data-generation/results/nmf_alpha_h_sweep.jsonl
#   L1_RATIO=1.0
#   ALPHA_W=0.0
#   ALPHA_H_VALUES_STR="1e-4 1e-3 1e-2 1e-1 1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${SCRIPT_DIR}/run_feature_extraction_experiment.py"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="${RESULTS_FILE:-${RESULTS_DIR}/nmf_alpha_h_sweep_${TIMESTAMP}.jsonl}"

L1_RATIO="${L1_RATIO:-1.0}"
ALPHA_W="${ALPHA_W:-0.0}"
ALPHA_H_VALUES_STR="${ALPHA_H_VALUES_STR:-1e-4 1e-3 1e-2 1e-1 1}"
read -r -a ALPHA_H_VALUES <<< "${ALPHA_H_VALUES_STR}"

TOTAL_RUNS="${#ALPHA_H_VALUES[@]}"
RUN_IDX=0

echo "Starting NMF alpha_H sweep: ${TOTAL_RUNS} runs"
echo "Fixed l1_ratio: ${L1_RATIO}"
echo "Fixed alpha_W: ${ALPHA_W}"
echo "alpha_H values: ${ALPHA_H_VALUES_STR}"
echo "Results file: ${RESULTS_FILE}"

for ALPHA_H in "${ALPHA_H_VALUES[@]}"; do
  RUN_IDX=$((RUN_IDX + 1))
  echo ""
  echo "[${RUN_IDX}/${TOTAL_RUNS}] alpha_H=${ALPHA_H}"

  "${PYTHON_BIN}" "${RUN_SCRIPT}" \
    --method nmf \
    --nmf-l1-ratio "${L1_RATIO}" \
    --nmf-alpha-w "${ALPHA_W}" \
    --nmf-alpha-h "${ALPHA_H}" \
    --results-file "${RESULTS_FILE}" \
    "$@"
done

echo ""
echo "NMF alpha_H sweep complete."
echo "Saved results to: ${RESULTS_FILE}"
