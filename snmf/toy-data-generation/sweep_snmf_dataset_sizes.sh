#!/usr/bin/env bash
set -euo pipefail

# Sweep run_snmf_fit_experiment.py over multiple toy dataset sizes.
#
# Usage:
#   bash snmf/toy-data-generation/sweep_snmf_dataset_sizes.sh [extra args...]
#
# Examples:
#   bash snmf/toy-data-generation/sweep_snmf_dataset_sizes.sh --init svd
#   bash snmf/toy-data-generation/sweep_snmf_dataset_sizes.sh --init knn --knn-iters 20
#
# Optional environment overrides:
#   PYTHON_BIN=python
#   RESULTS_FILE=snmf/toy-data-generation/results/dataset_size_sweep.jsonl
#   DATASET_SIZES_STR="5000 10000 25000 50000 100000"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${SCRIPT_DIR}/run_snmf_fit_experiment.py"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="${RESULTS_FILE:-${RESULTS_DIR}/snmf_dataset_size_sweep_${TIMESTAMP}.jsonl}"

DATASET_SIZES_STR="${DATASET_SIZES_STR:-5000 10000 25000 50000}"
read -r -a DATASET_SIZES <<< "${DATASET_SIZES_STR}"

TOTAL_RUNS="${#DATASET_SIZES[@]}"
RUN_IDX=0

echo "Starting dataset-size sweep: ${TOTAL_RUNS} runs"
echo "Dataset sizes: ${DATASET_SIZES_STR}"
echo "Results file: ${RESULTS_FILE}"

for NUM_SAMPLES in "${DATASET_SIZES[@]}"; do
  RUN_IDX=$((RUN_IDX + 1))
  echo ""
  echo "[${RUN_IDX}/${TOTAL_RUNS}] num_samples=${NUM_SAMPLES}"

  "${PYTHON_BIN}" "${RUN_SCRIPT}" \
    --num-samples "${NUM_SAMPLES}" \
    --results-file "${RESULTS_FILE}" \
    "$@"
done

echo ""
echo "Dataset-size sweep complete."
echo "Saved results to: ${RESULTS_FILE}"
