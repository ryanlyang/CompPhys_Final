#!/usr/bin/env bash
#SBATCH --job-name=rrrSweep16
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-15
#SBATCH --output=restart_studies/logs/rrr_sweep16_%A_%a.out
#SBATCH --error=restart_studies/logs/rrr_sweep16_%A_%a.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"

A_SOURCE="${A_SOURCE:-input_grad}"
SEED="${SEED:-52}"
RUN_BASENAME="${RUN_BASENAME:-rrr_findanother_seed52}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/restart_studies/results}"

LAMBDA_VALUES_RAW="${LAMBDA_VALUES:-1,10,100,1000}"
MASK_FRACS_RAW="${MASK_FRACS:-0.05,0.10,0.20,0.30}"

LAMBDA_VALUES_CSV="$(echo "${LAMBDA_VALUES_RAW}" | tr ' ' ',' | tr -s ',' | sed 's/^,//; s/,$//')"
MASK_FRACS_CSV="$(echo "${MASK_FRACS_RAW}" | tr ' ' ',' | tr -s ',' | sed 's/^,//; s/,$//')"

IFS=',' read -r -a LAMBDA_LIST <<< "${LAMBDA_VALUES_CSV}"
IFS=',' read -r -a MASK_LIST <<< "${MASK_FRACS_CSV}"

if [[ "${#LAMBDA_LIST[@]}" -ne 4 ]]; then
  echo "ERROR: expected 4 lambda values, got ${#LAMBDA_LIST[@]} from '${LAMBDA_VALUES_RAW}'" >&2
  exit 2
fi
if [[ "${#MASK_LIST[@]}" -ne 4 ]]; then
  echo "ERROR: expected 4 mask fractions, got ${#MASK_LIST[@]} from '${MASK_FRACS_RAW}'" >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_CFGS=$(( ${#LAMBDA_LIST[@]} * ${#MASK_LIST[@]} ))
if (( TASK_ID < 0 || TASK_ID >= TOTAL_CFGS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range for ${TOTAL_CFGS} configs" >&2
  exit 2
fi

N_MASK="${#MASK_LIST[@]}"
LIDX=$(( TASK_ID / N_MASK ))
MIDX=$(( TASK_ID % N_MASK ))
LAMBDA_RRR="${LAMBDA_LIST[$LIDX]}"
MASK_FRAC="${MASK_LIST[$MIDX]}"

to_tag() {
  local x="${1}"
  echo "${x}" | sed -e 's/-/m/g' -e 's/\./p/g' -e 's/+//g'
}

RUN_NAME="${RUN_NAME:-${RUN_BASENAME}_${A_SOURCE}_lam$(to_tag "${LAMBDA_RRR}")_mask$(to_tag "${MASK_FRAC}")}"

SINGLE_SCRIPT_CANDIDATES=(
  "${WORKDIR}/restart_studies/sbatch_train_rrr_find_another_single.sh"
  "${PROJECT_ROOT}/restart_studies/sbatch_train_rrr_find_another_single.sh"
)
SINGLE_SCRIPT=""
for c in "${SINGLE_SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SINGLE_SCRIPT="${c}"
    break
  fi
done

if [[ -z "${SINGLE_SCRIPT}" ]]; then
  echo "ERROR: Could not find single-config sbatch script." >&2
  echo "Checked: ${SINGLE_SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

mkdir -p "${WORKDIR}/restart_studies/logs"

echo "============================================================"
echo "RRR Find-Another Sweep (16-config array)"
echo "Array Job: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "Task ID: ${TASK_ID} / $((TOTAL_CFGS - 1))"
echo "A_SOURCE: ${A_SOURCE}"
echo "Lambda: ${LAMBDA_RRR}"
echo "Mask frac: ${MASK_FRAC}"
echo "Run name: ${RUN_NAME}"
echo "Delegating to: ${SINGLE_SCRIPT}"
echo "============================================================"

export PROJECT_ROOT
export OUTPUT_ROOT
export RUN_BASENAME
export RUN_NAME
export SEED
export A_SOURCE
export LAMBDA_RRR
export MASK_FRAC

bash "${SINGLE_SCRIPT}"

