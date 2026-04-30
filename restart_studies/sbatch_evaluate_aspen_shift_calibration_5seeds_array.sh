#!/usr/bin/env bash
#SBATCH --job-name=aspenShift5
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=restart_studies/logs/aspen_shift5_%A_%a.out
#SBATCH --error=restart_studies/logs/aspen_shift5_%A_%a.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"

SEEDS_RAW="${SEEDS:-41,52,63,74,85}"
SEEDS_CSV="$(echo "${SEEDS_RAW}" | tr ' ' ',' | tr -s ',' | sed 's/^,//; s/,$//')"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS_CSV}"

if [[ "${#SEED_LIST[@]}" -ne 5 ]]; then
  echo "ERROR: Expected exactly 5 seeds, got ${#SEED_LIST[@]} from SEEDS='${SEEDS_RAW}'" >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#SEED_LIST[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range for ${#SEED_LIST[@]} seeds" >&2
  exit 2
fi

SEED="${SEED_LIST[$TASK_ID]}"
RUN_BASENAME="${RUN_BASENAME:-prelim_reimpl_cluster}"
MODEL_RUN_NAME="${MODEL_RUN_NAME:-${RUN_BASENAME}_seed${SEED}}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${MODEL_RUN_NAME}_aspen_shift_1M}"

SINGLE_SEED_SCRIPT_CANDIDATES=(
  "${WORKDIR}/restart_studies/sbatch_evaluate_aspen_shift_calibration.sh"
  "${PROJECT_ROOT}/restart_studies/sbatch_evaluate_aspen_shift_calibration.sh"
)
SINGLE_SEED_SCRIPT=""
for c in "${SINGLE_SEED_SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SINGLE_SEED_SCRIPT="${c}"
    break
  fi
done

if [[ -z "${SINGLE_SEED_SCRIPT}" ]]; then
  echo "ERROR: Could not find single-seed Aspen shift sbatch script." >&2
  echo "Checked: ${SINGLE_SEED_SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

mkdir -p "${WORKDIR}/restart_studies/logs"

echo "============================================================"
echo "Aspen Shift Calibration Evaluation (5-seed array task)"
echo "Array Job: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Task ID: ${TASK_ID}"
echo "Seed: ${SEED}"
echo "Model run: ${MODEL_RUN_NAME}"
echo "Output run: ${OUTPUT_RUN_NAME}"
echo "Delegating to: ${SINGLE_SEED_SCRIPT}"
echo "============================================================"

export PROJECT_ROOT
export RUN_BASENAME
export SEED
export MODEL_RUN_NAME
export OUTPUT_RUN_NAME

bash "${SINGLE_SEED_SCRIPT}"
